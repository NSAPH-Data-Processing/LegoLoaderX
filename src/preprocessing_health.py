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

    LOGGER.info(f"Processing data for {cfg.var}")
    resolution = f"{cfg.min_spatial_res}_{cfg.min_temporal_res}"
    input_files = f"{cfg.input_dir}/{cfg.lego_dir}/medpar_outcomes/{cfg.vg_name}/{resolution}/{cfg.lego_prefix}_*.parquet"
    output_folder = f"{cfg.output_dir}/{cfg.vg_name}/{cfg.var}"
    os.makedirs(output_folder, exist_ok=True)

    year = cfg.year
    horizons = cfg.horizons
    LOGGER.info(f"Processing year {year}")

    # get days list for a given year with calendar days
    days_list = [(year, month, day) for month in range(1, 13) for day in range(1, calendar.monthrange(year, month)[1] + 1)]
    for day in tqdm(days_list, desc="Processing days"):
        date_str = f"{day[0]}{day[1]:02d}{day[2]:02d}"
        output_fname = f"{output_folder}/{cfg.var}__{date_str}.parquet"

        t = date(day[0], day[1], day[2])

        # Build queries for each horizon, starting with same-day (horizon = 0)
        queries = []

        # Same-day count (horizon = 0)
        queries.append(f"""
            SELECT 
                zcta, 
                0 AS horizon, 
                n
            FROM '{input_files}'
            WHERE
                var = '{cfg.var}' AND 
                date = DATE '{t}'
        """)

        # Future horizons
        for horizon in horizons:
            t_end = t + timedelta(days=horizon)
            queries.append(f"""
                SELECT 
                    zcta, 
                    {horizon} AS horizon, 
                    SUM(n) AS n
                FROM '{input_files}'
                WHERE 
                    var = '{cfg.var}' AND 
                    date >= DATE '{t}' AND 
                    date <= DATE '{t_end}'
                GROUP BY zcta
            """)

        # Combine all queries into one
        full_query = " UNION ALL ".join(queries)

        # Execute and save
        conn.execute(f"""
            CREATE OR REPLACE TABLE output AS
            {full_query}
        """)

        conn.execute(f"""
            COPY (SELECT * FROM output ORDER BY zcta, horizon) 
            TO '{output_fname}'
        """)


    conn.close()
    
if __name__ == "__main__":
    main()