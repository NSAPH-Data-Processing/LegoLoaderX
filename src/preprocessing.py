import duckdb
import hydra


@hydra.main(config_path="../conf", config_name="conf", version_base=None)
def main(cfg):
    # parsing timestring
    month,day = None, None
    timestr = str(cfg.timestr)
    year = timestr[:4]
    if cfg.temporal_res == "monthly" or cfg.temporal_res == "daily":
        month = timestr[4:6]
    if cfg.temporal_res == "daily":
        day = timestr[6:8]

    # getting input filepath
    cfg_vg = cfg.var_group
    input_fname = f"{cfg.input_dir}/{cfg_vg.lego_dir}/{cfg_vg.lego_nm}/{cfg.spatial_res}_{cfg.temporal_res}/{cfg_vg.lego_nm}__{cfg.spatial_res}_{cfg.temporal_res}__{year}.parquet"

    # setting output filepath
    output_fname = f"{cfg.output_dir}/{cfg.vg_name}__{cfg.var}__{year}"
    if month:
        output_fname += str(month).zfill(2)
    if day:
        output_fname += str(day).zfill(2)
    output_fname += ".parquet"

    # getting unique id list
    uniq_path = f"{cfg.input_dir}/{cfg.uniqid_dir}/{cfg.uniqid_nm}/{cfg.spatial_res}_yearly/{cfg.uniqid_nm}__{cfg.spatial_res}_yearly__{year}.parquet"

    # Create index table
    duckdb.execute(f"""
        CREATE TABLE index AS SELECT * FROM read_parquet('{uniq_path}') ORDER BY {cfg.spatial_res}
    """)

    # Construct filtering condition
    if day:
        filter_condition = f"WHERE date = '{year}-{month}-{day}'"
    elif month:
        filter_condition = f"WHERE year = '{year}' AND month = '{month}'"
    elif year:
        filter_condition = f"WHERE year = '{year}'"
    else:
        filter_condition = ""

    # Optimized query: Apply filtering *before* joining
    query = f"""
        COPY (
            SELECT i.{cfg.spatial_res}, d.{cfg.var}
            FROM index AS i
            LEFT JOIN (
                SELECT {cfg.spatial_res}, {cfg.var} FROM read_parquet('{input_fname}')
                {filter_condition}  -- Filtering before join
            ) AS d
            ON i.{cfg.spatial_res} = d.{cfg.spatial_res}
        ) TO '{output_fname}' (FORMAT 'parquet');
    """

    duckdb.execute(query)
if __name__ == "__main__":
    main()