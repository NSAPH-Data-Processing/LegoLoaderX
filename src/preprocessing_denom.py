import os
import logging

import duckdb
import hydra
import numpy as np
import pandas as pd
import pyarrow.parquet as pq


# Configure logging
LOGGER = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


@hydra.main(config_path="../conf/health", config_name="config", version_base=None)
def main(cfg):
    """Preprocess denominator data for health datasets."""
    fmt = cfg.get("format", "daily_parquet")
    year = str(cfg.year)
    denom_path = f"{cfg.input_dir}/{cfg.lego_dir}/mbsf_medpar_denom/{cfg.min_spatial_res}_yearly/counts_{year}.parquet"

    LOGGER.info(f"Reading denominator data from {denom_path}")
    denom_df = pq.read_table(denom_path, columns=['zcta', 'n_bene']).to_pandas()

    out_root = f"{cfg.output_dir}/denom/{fmt}"
    os.makedirs(out_root, exist_ok=True)

    if fmt == "daily_parquet":
        tgt_file = f"{out_root}/denom__{year}.parquet"
        LOGGER.info(f"Saving processed denominator data to {tgt_file}")
        denom_df.to_parquet(tgt_file)
    else:
        # yearly_mmap_dense: int32 vector in idx2zcta (row index -> zcta) order, sentinel -1 for missing.
        idx2zcta = _ensure_idx2zcta(cfg)
        z2i = {z: i for i, z in enumerate(idx2zcta)}
        arr = np.full((len(idx2zcta),), -1, dtype=np.int32)
        rows = denom_df["zcta"].map(z2i).to_numpy()
        keep = ~pd.isna(rows)
        arr[rows[keep].astype(np.int64)] = denom_df.loc[keep, "n_bene"].to_numpy().astype(np.int32, copy=False)
        # For yearly_mmap_sparse the denom shares the dense root, so write to
        # /yearly_mmap_dense/ regardless (HealthDataset reads it from there).
        out_root_dense = f"{cfg.output_dir}/denom/yearly_mmap_dense"
        os.makedirs(out_root_dense, exist_ok=True)
        tgt_file = f"{out_root_dense}/denom__{year}.npy"
        np.save(tgt_file, arr)
        LOGGER.info(f"Saved processed denominator data to {tgt_file}")


def _ensure_idx2zcta(cfg):
    idx2zcta_path = f"{cfg.output_dir}/idx2zcta.txt"
    if not os.path.exists(idx2zcta_path):
        # Build it on demand using the same source as preprocessing_health.py.
        denom_glob = f"{cfg.input_dir}/{cfg.lego_dir}/mbsf_medpar_denom/{cfg.min_spatial_res}_yearly/counts_*.parquet"
        idx2zcta = duckdb.execute(f"""
            SELECT DISTINCT zcta
            FROM read_parquet('{denom_glob}')
            ORDER BY zcta
        """).df()["zcta"].tolist()
        os.makedirs(os.path.dirname(idx2zcta_path), exist_ok=True)
        with open(idx2zcta_path, "w") as f:
            f.write("\n".join(idx2zcta) + "\n")
        LOGGER.info(f"Wrote {idx2zcta_path} ({len(idx2zcta)} zctas)")
        return idx2zcta
    with open(idx2zcta_path) as f:
        return [line.strip() for line in f if line.strip()]


if __name__ == "__main__":
    main()
