import pandas as pd
import geopandas as gpd
import numpy as np
import random
import os
import hydra
import logging

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


@hydra.main(config_path="../conf/synthgen", config_name="config", version_base=None)
def main(cfg):
    """
    Generate synthetic denominator data for preprocessing_denom.py.
    This creates denominator data (zcta, n_bene) from population data.
    """
    LOGGER.info(f"Processing synthetic denominator data for year {cfg.year}")

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

    # Generate synthetic denominator data
    LOGGER.info("Generating synthetic denominator data...")
    denom_df = zcta_data[['zcta', 'population']].copy()
    denom_df = denom_df.rename(columns={'population': 'n_bene'})
    
    # Save denominator data for preprocessing_denom.py
    # Based on the new snakemake config, the path is defined by denom_lego_path
    denom_input_file = f"data/input/lego/medicare_synthetic/mbsf_medpar_denom/zcta_yearly/counts_{cfg.year}.parquet"
    LOGGER.info(f"Saving synthetic denominator data to {denom_input_file}")
    os.makedirs(os.path.dirname(denom_input_file), exist_ok=True)
    denom_df.to_parquet(denom_input_file, index=False)

    LOGGER.info(f"Synthetic denominator data generation completed for year {cfg.year}")


if __name__ == "__main__":
    main()