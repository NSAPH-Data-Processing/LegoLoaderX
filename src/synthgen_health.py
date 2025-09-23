import calendar
import logging
import os
import random
from datetime import date

import geopandas as gpd
import hydra
import numpy as np
import pandas as pd

# Configure logging
LOGGER = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


def get_zcta_data_with_geo_pop(
    unique_fpath, shapefile_fpath, population_fpath, year, mainland_only=True
):
    """
    Extract ZCTA IDs with geographic coordinates and population data
    Returns a comprehensive dataset for mainland US ZCTAs
    """
    LOGGER.info(f"Loading ZCTA data for year {year}")

    # Read unique ID file for the given year
    unique_file = f"{unique_fpath}/us_uniqueid__census__zcta_yearly__{year}.parquet"
    df_unique = pd.read_parquet(unique_file)

    # Filter for mainland US if requested
    if mainland_only and "continental_us" in df_unique.columns:
        df_unique = df_unique[df_unique.continental_us]
        LOGGER.info(f"Filtered to {len(df_unique)} mainland US ZCTAs")

    # Read shapefile for geographic data
    shapefile_dir = f"{shapefile_fpath}/us_shapefile__census__zcta_yearly__{year}"
    shapefile_path = f"{shapefile_dir}/us_shapefile__census__zcta_yearly__{year}.shp"

    if os.path.exists(shapefile_path):
        gdf = gpd.read_file(shapefile_path)

        # Calculate centroids for lat/lon (project to appropriate CRS first)
        gdf_projected = gdf.to_crs("EPSG:3857")  # Web Mercator for accurate centroids
        centroids = gdf_projected.geometry.centroid.to_crs(
            gdf.crs
        )  # Back to original CRS

        gdf["longitude"] = centroids.x
        gdf["latitude"] = centroids.y

        # Keep only necessary columns
        geo_data = gdf[["zcta", "longitude", "latitude"]].copy()

        LOGGER.info(f"Loaded geographic data for {len(geo_data)} ZCTAs")
    else:
        raise FileNotFoundError(
            f"Shapefile not found: {shapefile_path}. "
            f"Cannot proceed without geographic data for year {year}."
        )

    # Read population data
    df_pop_full = pd.read_parquet(population_fpath).reset_index()

    # Map year to nearest census year (population data only available for 2000, 2010, 2020)
    available_years = sorted(df_pop_full["year"].unique())

    # Find the closest census year
    if year <= 2005:
        census_year = 2000
    elif year <= 2015:
        census_year = 2010
    else:
        census_year = 2020

    # Use population data from the mapped census year
    df_pop = df_pop_full[df_pop_full["year"] == census_year]

    if len(df_pop) == 0:
        raise ValueError(
            f"No population data available for census year {census_year} "
            f"(mapped from year {year}). Available years: {available_years}"
        )

    if census_year != year:
        LOGGER.info(
            f"Using population data from census year {census_year} for year {year}"
        )

    # Merge all data together
    zcta_data = df_unique.merge(geo_data, on="zcta", how="left")
    zcta_data = zcta_data.merge(df_pop[["zcta", "population"]], on="zcta", how="left")

    # Check for missing population data - this indicates a data consistency problem
    zctas_with_nan = zcta_data[zcta_data["population"].isna()]["zcta"].tolist()
    if len(zctas_with_nan) > 0:
        if len(zctas_with_nan) == len(zcta_data):
            raise ValueError(
                f"All ZCTAs are missing population data for year {census_year}. "
                f"Check population data file: {population_fpath}"
            )
        LOGGER.warning(
            f"Can't generate data for zcta's {len(zctas_with_nan)} - missing population data for year {census_year}"
        )

    # Filter nans
    zcta_data = zcta_data[~zcta_data["population"].isna()]

    # Check for missing coordinates - this indicates a real problem
    missing_lon = zcta_data["longitude"].isna()
    missing_lat = zcta_data["latitude"].isna()
    if missing_lon.any() or missing_lat.any():
        missing_count = missing_lon.sum()
        num_seen = len(zcta_data)
        # raise an error if missing exceeds 5% of total
        if missing_count / num_seen > 0.1:
            raise ValueError(
                f"More than 10% of ZCTAs ({missing_count} out of {num_seen}) are missing coordinates! "
                f"Check shapefile data for year {year}."
            )

        LOGGER.warning(f"Found {missing_count} ZCTAs with missing coordinates (<10%). Removing them.")
        zcta_data = zcta_data[~missing_lon & ~missing_lat]


    LOGGER.info(f"Final dataset: {len(zcta_data)} ZCTAs with geo and population data")

    return zcta_data


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
