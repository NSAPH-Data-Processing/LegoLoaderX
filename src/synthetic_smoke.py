"""Smoke run: build the DGP rate grid and print the distribution of lambda -- NO sampling, NO writes.

Reuses the exact generator rate (``expected_rate_grid``), so the distribution shown here is the true
per-cell Poisson rate the outcome WOULD be drawn from. Use it to sanity-check that lambda stays low,
how many cells hit ``rate_floor``, and what count sparsity to expect -- BEFORE launching a full
generation run. Because it calls the same function the generator and the ground-truth ERC use, what
you see is exactly the rate those pipelines would use (including any spatial/temporal coupling).

Run:
    PYTHONPATH=. python -m src.synthetic_smoke year=2010 synthetic.var_name=diabetes
    PYTHONPATH=. python -m src.synthetic_smoke year=2010 debug=true debug_days=30    # fewer days
    PYTHONPATH=. python -m src.synthetic_smoke year=2010 synthetic.spacetime.rho=0.5 synthetic.spacetime.phi=0.93
    PYTHONPATH=. python -m src.synthetic_smoke year=2010 synthetic.poisson_params.rate_floor=0.02
"""

import logging

import hydra

from src.synthetic_causal import (
    describe_rate_grid,
    expected_rate_grid,
    offset_vector,
    plot_rate_distribution,
)
from src.synthetic_denom import get_zcta_data_with_geo_pop

LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")


@hydra.main(config_path="../conf/synthetic", config_name="config", version_base=None)
def main(cfg):
    LOGGER.info(f"Smoke run: building rate grid for {cfg.synthetic.var_name} {cfg.year} (no sampling, no writes)")

    # same ZCTA grid + offset the generator uses (row order defines W alignment)
    zcta_data = get_zcta_data_with_geo_pop(
        unique_fpath=cfg.synthetic.zcta_unique_path,
        shapefile_fpath=cfg.synthetic.zcta_shapefile_path,
        population_fpath=cfg.synthetic.population_path,
        year=cfg.year,
        mainland_only=cfg.synthetic.mainland_only,
    )

    rate_grid = expected_rate_grid(cfg, zcta_data)   # the SAME floored rate the generator draws from
    offset = offset_vector(cfg, zcta_data)

    if cfg.get("debug", False):
        n = int(cfg.get("debug_days", 3))
        rate_grid = rate_grid[:n]
        LOGGER.info(f"debug: limiting the lambda summary to the first {rate_grid.shape[0]} days")

    floor = float(cfg.synthetic.poisson_params.get("rate_floor", 0.01))
    label = f"lambda[{cfg.synthetic.var_name} {cfg.year}]"
    describe_rate_grid(rate_grid, offset=offset, floor=floor, label=label)

    # save a histogram PNG (diagnostics.plot_path overrides; default lands in outputs/, gitignored)
    diag = cfg.synthetic.get("diagnostics", {}) or {}
    plot_path = diag.get("plot_path", None) or f"outputs/lambda_hist_{cfg.synthetic.var_name}_{cfg.year}.png"
    plot_rate_distribution(rate_grid, plot_path, floor=floor, offset=offset, label=label)


if __name__ == "__main__":
    main()
