from datetime import date, timedelta
import duckdb
import hydra
import os
import logging
import calendar
from tqdm import tqdm


# Configure logging
LOGGER = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

@hydra.main(config_path="../conf/health", config_name="config", version_base=None)
def main(cfg):
    """
    Preprocess health data for data loader.
    Current implementation is for the LEGO dataset.
    Only zcta daily data is supported (with hardcoded vars)
    """
    
    conn = duckdb.connect()

    LOGGER.info(f"Processing data for icd code {cfg.var}")
    input_files = f"{cfg.input_dir}/{cfg.lego_dir}/{cfg.lego_nm}_*.parquet"
    output_folder = f"{cfg.output_dir}/{cfg.vg_name}/{cfg.var}/"
    os.makedirs(output_folder, exist_ok=True)

    year = cfg.year
    lags = cfg.lags
    LOGGER.info(f"Processing year {year}")

    # get days list for a given year with calendar days
    days_list = [(year, month, day) for month in range(1, 13) for day in range(1, calendar.monthrange(year, month)[1] + 1)]
    for day in tqdm(days_list, desc="Processing days"):
        date_str = f"{day[0]}{day[1]:02d}{day[2]:02d}"
        output_fname = f"{output_folder}/{cfg.var}__{date_str}.parquet"

        # LOGGER.info(f"Processing date {date_str}")
        t = date(day[0], day[1], day[2])
        # Create index of all zcta + date in a relevant time window
        max_lag = max(lags)
        t_max = t + timedelta(days=max_lag)
        
        conn.execute(f"""
        CREATE OR REPLACE TABLE index AS
        SELECT DISTINCT zcta, date
        FROM read_parquet('{input_files}')
        WHERE 
            icd10 = '{cfg.var}' AND 
            date >= DATE '{t}' AND
            date <= DATE '{t_max}'
        ORDER BY zcta, date
        """)

        conn.execute(f"""
        CREATE OR REPLACE TABLE n AS
        SELECT 
            zcta, 
            date, 
            n
        FROM read_parquet('{input_files}')
        WHERE 
            icd10 = '{cfg.var}' AND 
            date = DATE '{t}'
        ORDER BY zcta
        """)

        # Initialize base output (index)
        conn.execute(f"""
        CREATE OR REPLACE TABLE output AS
        SELECT 
            index.zcta, 
            index.date, 
            COALESCE(n.n, 0) AS n
        FROM index
        LEFT JOIN n ON index.zcta = n.zcta AND index.date = n.date
        """)

        # For each lag, create n_lag and join it
        for lag in lags:
            t_lag = t + timedelta(days=lag)
            conn.execute(f"""
            CREATE OR REPLACE TABLE n_{lag} AS
            WITH i_{lag} AS (
            SELECT 
                zcta, 
                date, 
                n AS n_{lag}
            FROM read_parquet('{input_files}')
            WHERE 
                icd10 = '{cfg.var}' AND 
                date >= DATE '{t}' AND
                date <= DATE '{t_lag}'
                )
            SELECT
                zcta, 
                date, 
                SUM(n_{lag}) AS n_{lag}
            FROM i_{lag}
            GROUP BY zcta, date
            ORDER BY zcta, date
            """)

            conn.execute(f"""
            CREATE OR REPLACE TABLE output AS
            SELECT 
                o.*, 
                COALESCE(n_{lag}.n_{lag}, 0) AS n_{lag}
            FROM output o
            LEFT JOIN n_{lag} ON o.zcta = n_{lag}.zcta AND o.date = n_{lag}.date
            """)
        
        # Save final result
        conn.execute(f"""
        COPY output TO '{output_fname}' (FORMAT PARQUET)
        """)

    LOGGER.info("Processing complete.")
    conn.close()
    
if __name__ == "__main__":
    main()