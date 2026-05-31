import calendar
import logging
import os

import duckdb
import hydra
import numpy as np
import pandas as pd
import scipy.sparse
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
    Only zcta daily data is supported (with hardcoded vars).
    """
    target = cfg.get("target", "var_year")
    fmt = cfg.get("format", "daily_parquet")

    if target == "idx2zcta":
        _write_idx2zcta(cfg)
        return

    if fmt == "daily_parquet":
        _build_daily_parquet(cfg)
    elif fmt in ("yearly_mmap_dense", "yearly_mmap_sparse"):
        _build_yearly_mmap(cfg, sparse=(fmt == "yearly_mmap_sparse"))
    else:
        raise ValueError(f"unsupported format {fmt!r}")


def _build_daily_parquet(cfg):
    LOGGER.info(f"Processing data for {cfg.var}")
    resolution = f"{cfg.min_spatial_res}_{cfg.min_temporal_res}"
    input_files = f"{cfg.input_dir}/{cfg.lego_dir}/medpar_outcomes/{cfg.vg_name}/{resolution}/{cfg.lego_prefix}_*.parquet"
    output_folder = f"{cfg.output_dir}/{cfg.vg_name}/{cfg.var}/{cfg.format}"
    os.makedirs(output_folder, exist_ok=True)

    year = int(cfg.year)
    LOGGER.info(f"Processing year {year}")

    # One query for the whole year, then split into per-day files in memory.
    # Output is identical to a per-day query (one file per calendar day with
    # columns (zcta, n) sorted by zcta; empty file on days with no counts) but
    # avoids re-scanning the source once per day.
    df = duckdb.execute(f"""
        SELECT zcta, date, n
        FROM '{input_files}'
        WHERE var = '{cfg.var}' AND date >= DATE '{year}-01-01' AND date <= DATE '{year}-12-31'
    """).df()
    df["ds"] = pd.to_datetime(df["date"]).dt.strftime("%Y%m%d")
    by_day = dict(tuple(df.groupby("ds")))
    empty = df.iloc[0:0][["zcta", "n"]]

    days_list = [(year, month, day) for month in range(1, 13) for day in range(1, calendar.monthrange(year, month)[1] + 1)]
    for (yy, mm, dd) in tqdm(days_list, desc="Processing days"):
        date_str = f"{yy}{mm:02d}{dd:02d}"
        sub = by_day.get(date_str)
        out = sub[["zcta", "n"]].sort_values("zcta") if sub is not None else empty
        out.to_parquet(f"{output_folder}/{cfg.var}__{date_str}.parquet", index=False)


def _build_yearly_mmap(cfg, sparse):
    """Build (n_days_in_year, n_zctas) int16 count matrix for one (var, year).
    Time axis first so a contiguous day-window is a fast row-slice (dense) / CSR
    row-slice (sparse) rather than a strided column-slice. Same-day counts only."""
    LOGGER.info(f"Building {cfg.format} for {cfg.var}/{cfg.year}")
    resolution = f"{cfg.min_spatial_res}_{cfg.min_temporal_res}"
    input_files = f"{cfg.input_dir}/{cfg.lego_dir}/medpar_outcomes/{cfg.vg_name}/{resolution}/{cfg.lego_prefix}_*.parquet"

    output_folder = f"{cfg.output_dir}/{cfg.vg_name}/{cfg.var}/{cfg.format}"
    os.makedirs(output_folder, exist_ok=True)
    ext = "npz" if sparse else "npy"
    output_fname = f"{output_folder}/{cfg.var}__{cfg.year}.{ext}"

    idx2zcta = _ensure_idx2zcta(cfg)
    n_zctas = len(idx2zcta)
    year = int(cfg.year)
    n_days = 366 if calendar.isleap(year) else 365

    df = duckdb.execute(f"""
        SELECT zcta, date, n
        FROM '{input_files}'
        WHERE
            var = '{cfg.var}' AND
            date >= DATE '{year}-01-01' AND
            date <= DATE '{year}-12-31'
    """).df()

    z2i = {z: i for i, z in enumerate(idx2zcta)}
    rows = df["zcta"].map(z2i).to_numpy()
    keep = ~pd.isna(rows)
    rows = rows[keep].astype(np.int64)
    doys = (pd.to_datetime(df.loc[keep, "date"]) - pd.Timestamp(year=year, month=1, day=1)).dt.days.to_numpy()
    vals = df.loc[keep, "n"].to_numpy()
    if vals.size and vals.max() > 32_000:
        raise OverflowError(f"{cfg.var}/{year}: max count {vals.max()} > 32 000; int16 would overflow")

    arr = np.zeros((n_days, n_zctas), dtype=np.int16)
    arr[doys, rows] = vals.astype(np.int16, copy=False)

    if sparse:
        # CSR with day as the major axis -> a day-window is a contiguous CSR row-slice.
        scipy.sparse.save_npz(output_fname, scipy.sparse.csr_matrix(arr))
    else:
        np.save(output_fname, arr)
    LOGGER.info(f"Saved {output_fname}")


def _ensure_idx2zcta(cfg):
    idx2zcta_path = f"{cfg.output_dir}/idx2zcta.txt"
    if os.path.exists(idx2zcta_path):
        with open(idx2zcta_path) as f:
            return [line.strip() for line in f if line.strip()]
    return _write_idx2zcta(cfg)


def _write_idx2zcta(cfg):
    """Derive the idx2zcta list from the medpar denom files and write it once."""
    denom_glob = f"{cfg.input_dir}/{cfg.lego_dir}/mbsf_medpar_denom/{cfg.min_spatial_res}_yearly/counts_*.parquet"
    idx2zcta = duckdb.execute(f"""
        SELECT DISTINCT zcta
        FROM read_parquet('{denom_glob}')
        ORDER BY zcta
    """).df()["zcta"].tolist()
    idx2zcta_path = f"{cfg.output_dir}/idx2zcta.txt"
    os.makedirs(os.path.dirname(idx2zcta_path), exist_ok=True)
    with open(idx2zcta_path, "w") as f:
        f.write("\n".join(idx2zcta) + "\n")
    LOGGER.info(f"Wrote {idx2zcta_path} ({len(idx2zcta)} zctas)")
    return idx2zcta


if __name__ == "__main__":
    main()
