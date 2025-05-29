import duckdb
import hydra
import os
import logging

# Configure logging
LOGGER = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

@hydra.main(config_path="../conf/health", config_name="conf", version_base=None)
def main(cfg):
    """
    Preprocess health data for data loader.
    Current implementation is for the LEGO dataset.
    Only zcta daily data is supported (with hardcoded vars)
    """
    
    # setting up duckdb connection
    conn = duckdb.connect()

    LOGGER.info(f"Processing data for icd code {cfg.var}")
    # setting output filepath
    out_dir = f"{cfg.output_dir}/{cfg.vg_name}/{cfg.var}/"

    years_list = list(range(cfg.min_year, cfg.max_year + 1))
    for year in years_list:
        LOGGER.info(f"Processing year {year}")
        # getting input filepaths
        input_fname = f"{cfg.input_dir}/{cfg.lego_dir}/{cfg.lego_nm}__{year}.parquet"
        output_folder = f"{cfg.out_dir}/{cfg.var}"
        mkdirs = os.makedirs(output_folder, exist_ok=True)

        for day in range(1, 366):
            # Constructing the date string
            date_str = f"{year}-{str(day).zfill(3)}"
            output_fname = f"{out_dir}/{cfg.var}__{date_str}.parquet"

            # Check if the output file already exists
            if os.path.exists(output_fname):
                LOGGER.info(f"Output file {output_fname} already exists. Skipping.")
                continue

            # Query to create index table
            conn.execute(f"""
                CREATE TABLE index AS 
                SELECT zcta, DATE '{date_str}' AS date
                FROM read_parquet('{cfg.input_dir}/{cfg.uniqid_dir}/{cfg.uniqid_nm}/{cfg.spatial_res}_yearly/{cfg.uniqid_nm}__{cfg.spatial_res}_yearly__{year}.parquet')
                WHERE continental_us = TRUE
                ORDER BY zcta
            """)

            # Query to copy data into output file
            conn.execute(f"""
                COPY (
                    SELECT i.zcta, d.{cfg.var}
                    FROM index i
                    JOIN read_parquet('{input_fname}') d 
                    ON (i.zcta = d.zcta AND i.date = d.date)
                ) TO '{output_fname}' (FORMAT PARQUET);
            """)
            LOGGER.info(f"Data for {date_str} written to {output_fname}")


        os.makedirs(out_dir, exist_ok=True)
        # 
    

    
if __name__ == "__main__":
    main()