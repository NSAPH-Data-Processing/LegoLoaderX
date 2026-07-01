import calendar
import logging
import os
import random
from datetime import date

import geopandas as gpd
import hydra
import numpy as np
import pandas as pd
from src.synthetic_causal import expected_rate_grid, offset_vector
from src.synthetic_denom import get_zcta_data_with_geo_pop
from src.synthetic_manifest import counts_path, write_manifest

# Configure logging
LOGGER = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


def generate_synthetic_data(zcta_data, date_list, var_name, rate_grid, offset):
    """Poisson-sample synthetic counts from a PRECOMPUTED floored rate grid, in one vectorized draw.

    ``rate_grid``: ``(n_days, n_zctas)`` floored Poisson RATE aligned to ``zcta_data`` row order, as
    returned by :func:`src.synthetic_causal.expected_rate_grid` -- which already includes the
    background + seasonal + geography + causal (beta*exposure + sum_k gamma_k*confounder_k) terms AND
    the optional spatial/temporal coupling. ``offset``: ``(n_zctas,)`` per-ZCTA Poisson offset
    (``offset_vector``). We sample the WHOLE grid in a single ``np.random.poisson`` call (verified
    byte-identical to the old per-day loop, which drew the same stream in the same C-order) and then
    melt to the long sparse ``(zcta, var, date, n)`` format, dropping zeros.

    This deliberately no longer rebuilds the rate day-by-day: that duplicated ``expected_rate_grid``
    and risked silently drifting from it / from the ground-truth ERC. There is now ONE definition of
    the DGP rate. ``date_list`` may be shorter than ``rate_grid`` (debug mode): only its first
    ``len(date_list)`` days are sampled -- the temporal coupling is causal, so those leading days are
    already correct.
    """
    n_days = len(date_list)
    if n_days > rate_grid.shape[0]:
        raise ValueError(f"date_list has {n_days} days but rate_grid only has {rate_grid.shape[0]}")
    rate_grid = rate_grid[:n_days]
    LOGGER.info(
        f"Generating synthetic data for {var_name}: {n_days} dates and {rate_grid.shape[1]} ZCTAs"
    )

    # Expected count per (day, zcta) = rate * offset; one vectorized Poisson draw over the whole grid.
    counts = np.random.poisson(rate_grid * offset[None, :])

    zctas = zcta_data["zcta"].to_numpy()
    all_synthetic_data = []
    for day_of_year, target_date in enumerate(date_list):
        day_counts = counts[day_of_year]
        nz = day_counts > 0                              # keep only non-zero records (sparse format)
        date_obj = date(target_date[0], target_date[1], target_date[2])
        all_synthetic_data.append(pd.DataFrame({
            "zcta": zctas[nz],
            "var": var_name,
            "date": date_obj,
            "n": day_counts[nz],
        }))

    # Log sparsity level
    concat_df = pd.concat(all_synthetic_data, ignore_index=True)
    total_possible = rate_grid.shape[1] * n_days
    sparsity = 100 * (1 - len(concat_df) / total_possible)
    LOGGER.info(f"  > Generated {len(concat_df):,} records implying sparsity of {sparsity:.2f}%")

    return concat_df


@hydra.main(config_path="../conf/synthetic", config_name="config", version_base=None)
def main(cfg):
    """
    Generating synthetic health data for data loader.
    This creates synthetic data that mimics the structure of the LEGO health sparse counts dataset.
    Generates all diseases configured in the disease_params section of the config file.
    """
    LOGGER.info(f"Processing synthetic data for year {cfg.year}")

    # setup random seed
    LOGGER.info(f"Using random seed: {cfg.synthetic.random_seed}")
    random.seed(cfg.synthetic.random_seed)
    np.random.seed(cfg.synthetic.random_seed)

    # Get ZCTA data with geographic coordinates and population information
    LOGGER.info("Loading ZCTA data with geographic and population information")
    zcta_data = get_zcta_data_with_geo_pop(
        unique_fpath=cfg.synthetic.zcta_unique_path,
        shapefile_fpath=cfg.synthetic.zcta_shapefile_path,
        population_fpath=cfg.synthetic.population_path,
        year=cfg.year,
        mainland_only=cfg.synthetic.mainland_only,
    )

    LOGGER.info(f"Found {len(zcta_data)} ZCTAs for year {cfg.year} with complete data")

    # The full (n_days, n_zctas) floored Poisson RATE: background + seasonal + geography + causal
    # terms (beta*exposure + sum_k gamma_k*confounder_k) + optional spatial/temporal coupling. This
    # is the SAME function the ground-truth ERC uses, so the generator can no longer drift from it.
    rate_grid = expected_rate_grid(cfg, zcta_data)
    offset = offset_vector(cfg, zcta_data)

    # get days list for a given year with calendar days
    days_list = [
        (cfg.year, month, day)
        for month in range(1, 13)
        for day in range(1, calendar.monthrange(cfg.year, month)[1] + 1)
    ]

    # Debug option: limit to first few days for testing
    if cfg.debug:
        days_list = days_list[: cfg.debug_days]
        LOGGER.info(f"Debug mode: processing only first {len(days_list)} days")

    # Generate synthetic data for ALL diseases
    LOGGER.info("Generating synthetic data for all diseases...")

    # Poisson-sample counts from the precomputed rate grid (one vectorized draw)
    disease_df = generate_synthetic_data(
        zcta_data, days_list, cfg.synthetic.var_name, rate_grid, offset,
    )

    # Save synthetic data as input files for the real health processing script.
    # Path matches the snakemake config: data/input/{counts_lego_path}/sparse_counts_{var}_{year}.parquet
    synthetic_input_file = counts_path(cfg.synthetic.var_name, cfg.year)
    LOGGER.info(f"Saving synthetic input data to {synthetic_input_file}")
    os.makedirs(os.path.dirname(synthetic_input_file), exist_ok=True)
    disease_df.to_parquet(synthetic_input_file, index=False)

    # Persist the exact DGP parameters as a manifest beside the data, so the ground truth for this
    # dataset is read back from here (not from a config that may later change). See synthetic_manifest.
    write_manifest(cfg)

    LOGGER.info(f"Synthetic data generation completed for year {cfg.year}")


if __name__ == "__main__":
    main()
