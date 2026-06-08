import duckdb
import hydra
import os


@hydra.main(config_path="../conf", config_name="conf", version_base=None)
def main(cfg):
    """Generate idx2zcta.parquet: the canonical row-index -> zcta order.

    A single-column parquet whose row order defines the index used by the
    covariate mmaps. It is derived once from the union of the lego unique-id
    parquets across *all* years, so the row axis is stable year to year (a zcta
    missing in one year keeps its fixed row).
    """
    # union of the unique-id parquets across all years (glob on year)
    uniq_glob = (
        f"{cfg.input_dir}/{cfg.uniqid_dir}/{cfg.uniqid_nm}/"
        f"{cfg.spatial_res}_yearly/{cfg.uniqid_nm}__{cfg.spatial_res}_yearly__*.parquet"
    )

    os.makedirs(cfg.output_dir, exist_ok=True)
    output_fname = f"{cfg.output_dir}/idx2zcta.parquet"

    duckdb.execute(f"""
        COPY (
            SELECT DISTINCT {cfg.spatial_res}
            FROM read_parquet('{uniq_glob}')
            WHERE continental_us = TRUE
            ORDER BY {cfg.spatial_res}
        ) TO '{output_fname}' (FORMAT 'parquet');
    """)


if __name__ == "__main__":
    main()
