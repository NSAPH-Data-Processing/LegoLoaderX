import duckdb
import hydra


@hydra.main(config_path="../conf", config_name="conf", version_base=None)
def main(cfg):
    month,day = None, None
    timestr = str(cfg.timestr)
    year = timestr[:4]
    if cfg.temporal_res == "monthly" or cfg.temporal_res == "daily":
        month = timestr[4:6]
    if cfg.temporal_res == "daily":
        day = timestr[6:8]

    # getting input filepath
    cfg_vg = cfg.var_group
    input_fname = f"{cfg_vg.input_path}/{cfg.spatial_res}_{cfg.temporal_res}/{cfg_vg.file_prefix}__{cfg.spatial_res}_{cfg.temporal_res}__{year}.parquet"

    # setting output filepath
    output_fname = f"{cfg.output_dir}/{cfg.vg_name}__{cfg.var}__{year}"
    if month:
        output_fname += str(month).zfill(2)
    if day:
        output_fname += str(day).zfill(2)
    output_fname += ".parquet"

    # setting duckdb query
    query_base = f"""
    COPY (
        SELECT {cfg.spatial_res}, {cfg.var}
        FROM read_parquet('{input_fname}') \n"""
    if day:
        query_date = f"WHERE date = '{year}-{month}-{day}'\n"
    elif month:
        query_date = f"WHERE year = '{year}' AND month = '{month}'\n"
    elif year:
        query_date = f"WHERE year = '{year}'\n"

    query = query_base + query_date + f""") TO '{output_fname}' (FORMAT 'parquet');"""

    duckdb.execute(query)

if __name__ == "__main__":
    main()