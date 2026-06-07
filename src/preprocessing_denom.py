import logging
import os

import hydra
import numpy as np
import pandas as pd
import pyarrow.parquet as pq


LOGGER = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


@hydra.main(config_path="../conf/health", config_name="config", version_base=None)
def main(cfg):
    """Build the per-year denominator mmap in idx2zcta row order.

    ``float32`` ``(n_zctas,)`` of beneficiary counts; ``NaN`` for ZCTAs absent
    from the source (``0.0`` means a genuine zero count). float32 represents
    integer counts exactly well past any beneficiary total, so it keeps "missing"
    consistent with the covariate/outcome mmaps without a magic sentinel.
    """
    year = str(cfg.year)
    denom_path = f"{cfg.input_dir}/{cfg.lego_dir}/mbsf_medpar_denom/{cfg.min_spatial_res}_yearly/counts_{year}.parquet"

    LOGGER.info(f"Reading denominator data from {denom_path}")
    denom_df = pq.read_table(denom_path, columns=['zcta', 'n_bene']).to_pandas()

    idx2zcta = pd.read_parquet(f"{cfg.output_dir}/idx2zcta.parquet")["zcta"].tolist()
    z2i = {z: i for i, z in enumerate(idx2zcta)}

    mapped = denom_df["zcta"].map(z2i).fillna(-1).astype(np.int64).to_numpy()  # -1 = zcta not in idx2zcta
    keep = mapped >= 0
    rows = mapped[keep]
    vals = denom_df["n_bene"].to_numpy(dtype=np.float32)[keep]

    arr = np.full((len(idx2zcta),), np.nan, dtype=np.float32)
    arr[rows] = vals

    out_dir = f"{cfg.output_dir}/denom"
    os.makedirs(out_dir, exist_ok=True)
    tgt_file = f"{out_dir}/denom__{year}.npy"
    np.save(tgt_file, arr)
    LOGGER.info(f"Saved {tgt_file}")


if __name__ == "__main__":
    main()
