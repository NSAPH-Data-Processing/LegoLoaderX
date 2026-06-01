import calendar
import os

import duckdb
import hydra
import numpy as np
import pandas as pd


@hydra.main(config_path="../conf", config_name="conf", version_base=None)
def main(cfg):
    target = cfg.get("target", "var_year")
    fmt = cfg.get("format", "daily_parquet")

    if target == "idx2zcta":
        _write_idx2zcta(cfg)
        return

    if fmt == "daily_parquet":
        _build_daily_parquet(cfg)
    elif fmt == "yearly_mmap_dense":
        _build_yearly_mmap_dense(cfg)
    else:
        raise ValueError(f"unsupported format {fmt!r} for covariates")


def _source_col(cfg):
    """Source column name in the lego view for the requested var.

    Defaults to the var itself. A var_group may declare a ``lego_col`` template
    when the lego column name differs from the canonical var name — e.g.
    climate_types stores per-zone fractions as ``pct_af`` while the canonical
    var (and downstream embedding key) is ``Af``. The template may reference
    ``{var}`` and ``{var_lower}``.
    """
    tmpl = cfg.var_group.get("lego_col", None)
    if tmpl:
        return tmpl.format(var=cfg.var, var_lower=cfg.var.lower())
    return cfg.var


def _build_daily_parquet(cfg):
    # parsing timestring
    month, day = None, None
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

    cfg_vg = cfg.var_group
    input_fname = f"{cfg.input_dir}/{cfg_vg.lego_dir}/{cfg_vg.lego_nm}__{year}.parquet"

    out_dir = f"{cfg.output_dir}/{cfg.vg_name}/{cfg.var}/{cfg.format}/"
    os.makedirs(out_dir, exist_ok=True)

    output_fname = f"{out_dir}/{cfg.var}__{year}"
    if month:
        output_fname += str(month).zfill(2)
    if day:
        output_fname += str(day).zfill(2)
    output_fname += ".parquet"

    uniq_path = f"{cfg.input_dir}/{cfg.uniqid_dir}/{cfg.uniqid_nm}/{cfg.spatial_res}_yearly/{cfg.uniqid_nm}__{cfg.spatial_res}_yearly__{year}.parquet"

    duckdb.execute(f"""
        CREATE TABLE index AS
        {time_query}
        FROM read_parquet('{uniq_path}')
        WHERE continental_us = TRUE
        ORDER BY {cfg.spatial_res}
    """)

    src_col = _source_col(cfg)
    query = f"""
        COPY (
            SELECT i.{cfg.spatial_res}, d.{cfg.var}
            FROM index AS i
            LEFT JOIN (
                SELECT {cfg.spatial_res}, {time_col}, {src_col} AS {cfg.var} FROM read_parquet('{input_fname}')
            ) AS d
            {match_query}
        ) TO '{output_fname}' (FORMAT 'parquet');
    """

    duckdb.execute(query)


def _build_yearly_mmap_dense(cfg):
    """Build the per-(var, year) dense .npy in idx2zcta (row index -> zcta) order.

    Reads the same lego-raw year file as ``_build_daily_parquet`` (a sibling
    format, not a transform of the daily_parquet output) and scatters the
    values into a (n_periods, n_zctas) array. The ZCTA axis is the
    LAST axis so a contiguous time-window slice (the loader's access pattern)
    is a fast row-slice rather than a strided column-slice. ZCTAs absent from
    a year stay NaN (same semantics as the daily LEFT JOIN). Yearly vars have
    no time axis and stay 1-D (n_zctas,).
    """
    year = int(str(cfg.timestr)[:4])
    cfg_vg = cfg.var_group
    input_fname = f"{cfg.input_dir}/{cfg_vg.lego_dir}/{cfg_vg.lego_nm}__{year}.parquet"

    out_dir = f"{cfg.output_dir}/{cfg.vg_name}/{cfg.var}/{cfg.format}/"
    os.makedirs(out_dir, exist_ok=True)
    output_fname = f"{out_dir}/{cfg.var}__{year}.npy"

    idx2zcta = _ensure_idx2zcta(cfg)
    n = len(idx2zcta)
    z2i = {z: i for i, z in enumerate(idx2zcta)}
    src_col = _source_col(cfg)

    if cfg.temporal_res == "yearly":
        df = duckdb.execute(f"""
            SELECT {cfg.spatial_res} AS zcta, {src_col} AS val
            FROM read_parquet('{input_fname}')
            WHERE year = {year}
        """).df()
        arr = np.full((n,), np.nan, dtype=np.float32)
        rows = df["zcta"].map(z2i).to_numpy()
        keep = ~pd.isna(rows)
        arr[rows[keep].astype(np.int64)] = df.loc[keep, "val"].to_numpy().astype(np.float32, copy=False)
    elif cfg.temporal_res == "monthly":
        df = duckdb.execute(f"""
            SELECT {cfg.spatial_res} AS zcta, month, {src_col} AS val
            FROM read_parquet('{input_fname}')
            WHERE year = {year}
        """).df()
        # (n_months, n_zctas): time axis first so a month-window is a row-slice.
        arr = np.full((12, n), np.nan, dtype=np.float32)
        rows = df["zcta"].map(z2i).to_numpy()
        keep = ~pd.isna(rows)
        cols = df.loc[keep, "month"].to_numpy().astype(np.int64) - 1
        arr[cols, rows[keep].astype(np.int64)] = df.loc[keep, "val"].to_numpy().astype(np.float32, copy=False)
    elif cfg.temporal_res == "daily":
        n_days = 366 if calendar.isleap(year) else 365
        df = duckdb.execute(f"""
            SELECT {cfg.spatial_res} AS zcta, date, {src_col} AS val
            FROM read_parquet('{input_fname}')
            WHERE date >= DATE '{year}-01-01' AND date <= DATE '{year}-12-31'
        """).df()
        # (n_days, n_zctas): time axis first so a day-window is a row-slice.
        arr = np.full((n_days, n), np.nan, dtype=np.float32)
        rows = df["zcta"].map(z2i).to_numpy()
        keep = ~pd.isna(rows)
        doys = (pd.to_datetime(df.loc[keep, "date"]) - pd.Timestamp(year=year, month=1, day=1)).dt.days.to_numpy()
        arr[doys, rows[keep].astype(np.int64)] = df.loc[keep, "val"].to_numpy().astype(np.float32, copy=False)
    else:
        raise ValueError(f"unsupported temporal_res {cfg.temporal_res!r}")

    np.save(output_fname, arr)


def _ensure_idx2zcta(cfg):
    """Return the idx2zcta list; write idx2zcta.txt at output_dir if absent."""
    idx2zcta_path = f"{cfg.output_dir}/idx2zcta.txt"
    if os.path.exists(idx2zcta_path):
        with open(idx2zcta_path) as f:
            return [line.strip() for line in f if line.strip()]
    return _write_idx2zcta(cfg)


def _write_idx2zcta(cfg):
    """Derive the idx2zcta list from the lego unique-id parquets and write it once."""
    uniq_glob = f"{cfg.input_dir}/{cfg.uniqid_dir}/{cfg.uniqid_nm}/{cfg.spatial_res}_yearly/{cfg.uniqid_nm}__{cfg.spatial_res}_yearly__*.parquet"
    idx2zcta = duckdb.execute(f"""
        SELECT DISTINCT {cfg.spatial_res}
        FROM read_parquet('{uniq_glob}')
        WHERE continental_us = TRUE
        ORDER BY {cfg.spatial_res}
    """).df()[cfg.spatial_res].tolist()
    idx2zcta_path = f"{cfg.output_dir}/idx2zcta.txt"
    os.makedirs(os.path.dirname(idx2zcta_path), exist_ok=True)
    with open(idx2zcta_path, "w") as f:
        f.write("\n".join(idx2zcta) + "\n")
    return idx2zcta


if __name__ == "__main__":
    main()
