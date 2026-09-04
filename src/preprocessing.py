import calendar
import os

import duckdb
import hydra
import numpy as np
import pandas as pd


@hydra.main(config_path="../conf", config_name="conf", version_base=None)
def main(cfg):
    """Build the per-(var, year) dense covariate mmap in idx2zcta row order.

    One ``.npy`` per (var, year) with the **time axis first** so a contiguous
    time-window is a fast row-slice: daily -> ``(n_days, n_zctas)``, monthly ->
    ``(12, n_zctas)``, yearly -> ``(n_zctas,)``. Values are scattered into the
    array via ``idx2zcta`` (row index -> zcta); ZCTAs absent from the year stay
    ``NaN`` (same semantics as the old daily LEFT JOIN).
    """
    year = int(str(cfg.year)[:4])
    cfg_vg = cfg.var_group

    year_sep = cfg_vg.get("lego_year_sep", "__")
    input_fname = f"{cfg.input_dir}/{cfg_vg.lego_dir}/{cfg_vg.lego_nm}{year_sep}{year}.parquet"

    out_dir = f"{cfg.output_dir}/{cfg.vg_name}/{cfg.var}"
    os.makedirs(out_dir, exist_ok=True)
    output_fname = f"{out_dir}/{cfg.var}__{year}.npy"

    # canonical row order: row index -> zcta (built in step 8)
    idx2zcta = pd.read_parquet(f"{cfg.output_dir}/idx2zcta.parquet")[cfg.spatial_res].tolist()
    n = len(idx2zcta)
    z2i = {z: i for i, z in enumerate(idx2zcta)}

    if cfg.temporal_res == "yearly":
        df = duckdb.execute(f"""
            SELECT {cfg.spatial_res} AS zcta, {cfg.var} AS val
            FROM read_parquet('{input_fname}')
            WHERE year = {year}
        """).df()
        mapped = df["zcta"].map(z2i).fillna(-1).astype(np.int64).to_numpy()  # -1 = zcta not in idx2zcta
        keep = mapped >= 0
        rows = mapped[keep]
        vals = df["val"].to_numpy(dtype=np.float32)[keep]
        arr = np.full((n,), np.nan, dtype=np.float32)
        arr[rows] = vals
    elif cfg.temporal_res == "monthly":
        df = duckdb.execute(f"""
            SELECT {cfg.spatial_res} AS zcta, month, {cfg.var} AS val
            FROM read_parquet('{input_fname}')
            WHERE year = {year}
        """).df()
        mapped = df["zcta"].map(z2i).fillna(-1).astype(np.int64).to_numpy()  # -1 = zcta not in idx2zcta
        keep = mapped >= 0
        rows = mapped[keep]
        months = (df["month"].to_numpy().astype(np.int64) - 1)[keep]
        vals = df["val"].to_numpy(dtype=np.float32)[keep]
        # (n_months, n_zctas): time axis first so a month-window is a row-slice.
        arr = np.full((12, n), np.nan, dtype=np.float32)
        arr[months, rows] = vals
    elif cfg.temporal_res == "daily":
        n_days = 366 if calendar.isleap(year) else 365
        # date column name is configurable per var_group (some datasets call it `day`)
        date_col = cfg_vg.get("date_col", "date")
        df = duckdb.execute(f"""
            SELECT {cfg.spatial_res} AS zcta, {date_col} AS date, {cfg.var} AS val
            FROM read_parquet('{input_fname}')
            WHERE {date_col} >= DATE '{year}-01-01' AND {date_col} <= DATE '{year}-12-31'
        """).df()
        mapped = df["zcta"].map(z2i).fillna(-1).astype(np.int64).to_numpy()  # -1 = zcta not in idx2zcta
        keep = mapped >= 0
        rows = mapped[keep]
        doys = ((pd.to_datetime(df["date"]) - pd.Timestamp(year=year, month=1, day=1)).dt.days.to_numpy())[keep]
        vals = df["val"].to_numpy(dtype=np.float32)[keep]
        # (n_days, n_zctas): time axis first so a day-window is a row-slice.
        arr = np.full((n_days, n), np.nan, dtype=np.float32)
        arr[doys, rows] = vals
    else:
        raise ValueError(f"unsupported temporal_res {cfg.temporal_res!r}")

    np.save(output_fname, arr)


if __name__ == "__main__":
    main()
