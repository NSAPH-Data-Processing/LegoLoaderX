import duckdb
import hydra
import pandas as pd


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg):
    # getting input filepath
    input_fname = f"{cfg.input_path}/{cfg.file_prefix}__{cfg.spatial_res}_{cfg.temporal_res}__{cfg.year}.parquet"

    output_fname = f"{cfg.output_path}/{cfg.var_group}__{cfg.var}__{cfg.year}"
    if cfg.temporal_res == "monthly":
        output_fname += str(cfg.month)
    if cfg.temporal_res == "daily":
        output_fname += str(cfg.day)
    output_fname += ".parquet"


    query = f"""
    COPY (
        SELECT spatial_res, var
        FROM read_parquet('{input_fname}') 
        WHERE year = {cfg.year}
          AND month = {cfg.month}
          AND day = {cfg.day}
    ) TO '{output_fname}' (FORMAT 'parquet');
    """

    return None

if __name__ == "__main__":
    main()