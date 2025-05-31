from datetime import date, timedelta
import duckdb
import hydra
import os
import logging
import calendar

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

    years_list = list(range(cfg.min_year, cfg.max_year + 1))
    for year in years_list:
        LOGGER.info(f"Processing year {year}")

        # get days list for a given year with calendar days
        days_list = [(year, month, day) for month in range(1, 13) for day in range(1, calendar.monthrange(year, month)[1] + 1)]
        for day in days_list:
            date_str = f"{day[0]}{day[1]:02d}{day[2]:02d}"
            output_fname = f"{output_folder}/{cfg.var}__{date_str}.parquet"
            
            LOGGER.info(f"Processing date {date_str}")
            # Obtain icd counts for all zcta for a given day and in the next x days
            t = date(day[0], day[1], day[2])
            t90 = t + timedelta(days=90)

            # Select all dates with a positive n value for the given icd code in the 90 day window
            conn.execute(f"""
            CREATE OR REPLACE TABLE index AS
            SELECT
                zcta, 
                date
            FROM read_parquet('{input_files}')
            WHERE 
                icd10 = '{cfg.var}' AND 
                date >= DATE '{t}' AND
                date <= DATE '{t90}'
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

            conn.execute(f"""
            CREATE OR REPLACE TABLE n90 AS
            WITH i90 AS (
                SELECT 
                    zcta, 
                    date,
                    n as n90
                FROM read_parquet('{input_files}')
                WHERE 
                    icd10 = '{cfg.var}' AND 
                    date >= DATE '{t}' AND
                    date <= DATE '{t90}'
            )
            SELECT 
                zcta, 
                date,
                SUM(n90) AS n90
            FROM i90
            GROUP BY zcta, date
            ORDER BY zcta, date
            """)

            # Join the tables and write to output
            conn.execute(f"""
            CREATE OR REPLACE TABLE output AS
            WITH i AS ( 
                SELECT 
                    index.zcta, 
                    index.date, 
                    COALESCE(n.n, 0) AS n
                FROM index
                LEFT JOIN n ON index.zcta = n.zcta AND index.date = n.date
            )
            SELECT 
                i.zcta, 
                i.date, 
                i.n AS n,
                COALESCE(n90.n90,0) AS n90
            FROM i
            LEFT JOIN n90 ON i.zcta = n90.zcta AND i.date = n90.date
            """)
            
            LOGGER.info(f"Writing output to {output_fname}")
            conn.execute(f"""
            COPY output TO '{output_fname}' (FORMAT PARQUET)
            """)
    LOGGER.info("Processing complete.")
    conn.close()
    
if __name__ == "__main__":
    main()