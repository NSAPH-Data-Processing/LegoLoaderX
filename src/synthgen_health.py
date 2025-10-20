import calendar
import logging
import os
import random
from datetime import date

import geopandas as gpd
import hydra
import numpy as np
import pandas as pd
from src.synthgen_denom import get_zcta_data_with_geo_pop

# Configure logging
LOGGER = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

def generate_synthetic_data(zcta_data, date_list, var_name, disease_params):
    """
    Generate synthetic health data for ALL dates and ZCTAs at once using vectorized operations
    Much faster than generating one date at a time
    """
    LOGGER.info(
        f"Generating synthetic data for {var_name}: {len(date_list)} dates and {len(zcta_data)} ZCTAs"
    )

    # Geographic effects (arbitrary variation functions)
    lat_normalized = (zcta_data["latitude"] - 35) / 15
    lon_normalized = (zcta_data["longitude"] + 95) / 30
    lat_effect = disease_params.latitude_effect * np.sin(lat_normalized * np.pi)
    lon_effect = disease_params.longitude_effect * np.cos(lon_normalized * np.pi)

    # Create all data at once
    all_synthetic_data = []

    for day_of_year, target_date in enumerate(date_list):
        # Calculate seasonal effect for this date
        seasonal_effect = disease_params.seasonal_amplitude * np.sin(
            2 * np.pi * day_of_year / 365.25
        )

        # Calculate lambda parameters for all ZCTAs at once
        lambda_params = np.maximum(
            0.01,
            disease_params.base_rate + seasonal_effect + lat_effect + lon_effect,
        )

        # Generate all counts at once using vectorized Poisson
        offset = disease_params.population_normalizer * zcta_data["population"]
        counts = np.random.poisson(lambda_params * offset)

        # Create records for this date - use date object for compatibility with health script
        date_obj = date(target_date[0], target_date[1], target_date[2])
        date_data = {
            "zcta": zcta_data["zcta"].values,
            "var": var_name,
            "date": date_obj,
            "n": counts,
        }
        df = pd.DataFrame(date_data)

        # Remove zeros
        df = df[df["n"] > 0]

        all_synthetic_data.append(df)


    # Log sparsity level
    concat_df = pd.concat(all_synthetic_data, ignore_index=True)
    total_possible = len(zcta_data) * len(date_list)
    total_records = sum(len(df) for df in all_synthetic_data)
    sparsity = 100 * (1 - total_records / total_possible)
    LOGGER.info(f"  > Generated {len(concat_df):,} records implying sparsity of {sparsity:.2f}%")

    # Concatenate all data
    return concat_df


@hydra.main(config_path="../conf/synthgen", config_name="config", version_base=None)
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

    # Get disease-specific parameters
    disease_params = cfg.synthetic.poisson_params

    # Generate synthetic data for this disease
    disease_df = generate_synthetic_data(
        zcta_data, days_list, cfg.synthetic.var_name, disease_params
    )

    # Save synthetic data as input files for the real health processing script
    # Path matches the snakemake config: data/input/{counts_lego_path}/sparse_counts_{var}_{year}.parquet
    synthetic_input_file = f"data/input/lego/medicare_synthetic/medpar_outcomes/ccw/zcta_daily/sparse_counts_{cfg.synthetic.var_name}_{cfg.year}.parquet"
    LOGGER.info(f"Saving synthetic input data to {synthetic_input_file}")
    os.makedirs(os.path.dirname(synthetic_input_file), exist_ok=True)
    disease_df.to_parquet(synthetic_input_file, index=False)

    LOGGER.info(f"Synthetic data generation completed for year {cfg.year}")


if __name__ == "__main__":
    main()
