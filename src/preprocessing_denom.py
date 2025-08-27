import os
import hydra
import logging
import duckdb
import pyarrow.parquet as pq


# Configure logging
LOGGER = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

@hydra.main(config_path="../conf/health", config_name="config", version_base=None)
def main(cfg):
    """
    Preprocess denominator data for health datasets.
    """
    
    year = str(cfg.year)
    denom_path = f"{cfg.input_dir}/lego/medicare/mbsf_medpar_denom/zcta_yearly/counts_{year}.parquet"

    LOGGER.info(f"Reading denominator data from {denom_path}")
    
    # make duckdb query counting rows grouping by zcta where age_dob in between 65 and 110
    # denom_df = duckdb.sql(
    #     f"""
    #     SELECT zcta, COUNT(*) as count
    #     FROM read_parquet('{denom_path}')
    #     WHERE age_dob BETWEEN 65 AND 110
    #     GROUP BY zcta
    #     """
    # Alternative implementation using DuckDB SQL is available in version control if needed.
    denom_df = pq.read_table(denom_path, columns=['zcta', 'n_bene']).to_pandas()
    
    # save table
    tgt_file = f"{cfg.output_dir}/denom/denom__{year}.parquet"
    os.makedirs(f"{cfg.output_dir}/denom/", exist_ok=True)
    LOGGER.info(f"Saving processed denominator data to {tgt_file}")
    denom_df.to_parquet(tgt_file)

    LOGGER.info(f"Saved processed denominator data to {tgt_file}")


if __name__ == "__main__":
    main()