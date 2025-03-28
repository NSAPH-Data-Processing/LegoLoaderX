import duckdb
import hydra
import os
import pandas as pd


@hydra.main(config_path="../conf", config_name="conf", version_base=None)
def main(cfg):
    # parsing timestring
    month,day = None, None
    timestr = str(cfg.timestr)
    year = timestr[:4]
    if cfg.temporal_res == "yearly":
        time_col = "year"
        time_query = f"""SELECT {cfg.spatial_res}, '{year}' AS year"""
        match_query = f"""ON (i.{cfg.spatial_res} = d.{cfg.spatial_res} AND i.year = d.year)"""
    elif cfg.temporal_res == "monthly":
        month = timestr[4:6]
        time_col = "year, month"
        time_query = f"""SELECT {cfg.spatial_res}, '{year}', '{month}' AS year, month"""
        match_query = f"""ON (i.{cfg.spatial_res} = d.{cfg.spatial_res} AND 
                              i.year = d.year AND
                              i.month = d.month)"""
    elif cfg.temporal_res == "daily":
        month = timestr[4:6]
        day = timestr[6:8]
        time_col = "date"
        time_query = f"""SELECT {cfg.spatial_res}, DATE '{year}-{month}-{day}' AS date"""
        match_query = f"""ON (i.{cfg.spatial_res} = d.{cfg.spatial_res} AND i.date = d.date)"""

    # getting input filepath
    cfg_vg = cfg.var_group
    input_fname = f"{cfg.input_dir}/{cfg_vg.lego_dir}/{cfg_vg.lego_nm}/{cfg.spatial_res}_{cfg.temporal_res}/{cfg_vg.lego_nm}__{cfg.spatial_res}_{cfg.temporal_res}__{year}.parquet"

    # setting output filepath
    out_dir = f"{cfg.output_dir}/{cfg.vg_name}/{cfg.var}/"
    os.makedirs(out_dir, exist_ok=True)

    output_fname = f"{out_dir}/{cfg.var}__{year}"
    if month:
        output_fname += str(month).zfill(2)
    if day:
        output_fname += str(day).zfill(2)
    output_fname += ".parquet"

    # getting unique id list
    uniq_path = f"{cfg.input_dir}/{cfg.uniqid_dir}/{cfg.uniqid_nm}/{cfg.spatial_res}_yearly/{cfg.uniqid_nm}__{cfg.spatial_res}_yearly__{year}.parquet"

    duckdb.execute(f"""
        CREATE TABLE index AS 
        {time_query}
        FROM read_parquet('{uniq_path}') 
        WHERE continental_us = TRUE
        ORDER BY {cfg.spatial_res}
    """)

    # Optimized query: Apply filtering *before* joining
    query = f"""
        COPY (
            SELECT i.{cfg.spatial_res}, d.{cfg.var}
            FROM index AS i
            LEFT JOIN (
                SELECT {cfg.spatial_res}, {time_col}, {cfg.var} FROM read_parquet('{input_fname}')
            ) AS d
            {match_query}
        ) TO '{output_fname}' (FORMAT 'parquet');
    """

    duckdb.execute(query)

    
if __name__ == "__main__":
    main()