import calendar
import logging
import os

import duckdb
import hydra
import numpy as np
import pandas as pd


LOGGER = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


@hydra.main(config_path="../conf/health", config_name="config", version_base=None)
def main(cfg):
    """Build the per-(var, year) dense outcome mmap in idx2zcta row order.

    One ``.npy`` per (var, year): ``(n_days, n_zctas)`` int16 of same-day
    counts, time axis first so a contiguous day-window is a fast row-slice.

    Source data are wide yearly parquet files with one column per health
    outcome. Absent or zero zcta-days remain ``0`` in the output array.

    Any forecast windowing is a read-time concern (no precomputed horizons).
    """
    year = int(cfg.year)
    resolution = f"{cfg.min_spatial_res}_{cfg.min_temporal_res}"

    input_file = (
        f"{cfg.input_dir}/{cfg.lego_dir}/medpar_outcomes/"
        f"{cfg.vg_name}/{resolution}/"
        f"{cfg.lego_prefix}_{year}.parquet"
    )

    out_dir = f"{cfg.output_dir}/{cfg.vg_name}/{cfg.var}"
    os.makedirs(out_dir, exist_ok=True)
    output_fname = f"{out_dir}/{cfg.var}__{year}.npy"

    idx2zcta = pd.read_parquet(
        f"{cfg.output_dir}/idx2zcta.parquet"
    )["zcta"].tolist()

    n_zctas = len(idx2zcta)
    z2i = {z: i for i, z in enumerate(idx2zcta)}

    n_days = 366 if calendar.isleap(year) else 365

    LOGGER.info(f"Processing {cfg.var} {year}")

    df = duckdb.execute(f"""
        SELECT
            zcta,
            date,
            "{cfg.var}" AS n
        FROM '{input_file}'
        WHERE date >= DATE '{year}-01-01'
          AND date <= DATE '{year}-12-31'
          AND "{cfg.var}" > 0
    """).df()

    mapped = (
        df["zcta"]
        .map(z2i)
        .fillna(-1)
        .astype(np.int64)
        .to_numpy()
    )

    keep = mapped >= 0
    rows = mapped[keep]

    doys = ( #day of year indices
        (
            pd.to_datetime(df["date"])
            - pd.Timestamp(year=year, month=1, day=1)
        )
        .dt.days
        .to_numpy()
    )[keep]

    vals = df["n"].to_numpy()

    if vals.size and vals.max() > 32_000:
        raise OverflowError(
            f"{cfg.var}/{year}: max count {vals.max()} > 32 000; "
            "int16 would overflow"
        )

    vals = vals.astype(np.int16, copy=False)[keep]

    arr = np.zeros(
        (n_days, n_zctas),
        dtype=np.int16,
    )

    arr[doys, rows] = vals

    np.save(output_fname, arr)
    LOGGER.info(f"Saved {output_fname}")


if __name__ == "__main__":
    main()