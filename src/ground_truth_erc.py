"""Compute and SAVE the ground-truth exposure-response curve (ERC) for the current DGP.

WHAT THE GROUND TRUTH IS  (matches how the *model's* ERC is computed)
---------------------------------------------------------------------
We trace the curve by g-computation, with the SAME two-loop collapse the model uses:

  outer loop over exposure levels a in a grid A:
    inner, per node z:  force the exposure to `a` for EVERY node and timestep (confounders left
                        untouched), then SUM the expected outcome over the forecast window T_fc
                            F_true(z, a) = sum_{t in forecast}  E[Y_{z,t} | do(exposure = a)]
                                         = sum_{t in forecast}  lambda_{z,t}(a) * offset_z
    aggregate over the V nodes:
                            F_true(a)    = (1/V) * sum_z F_true(z, a)        # average over nodes

Each a gives one point (a, F_true(a)); sweeping A traces the curve. This is exactly the model's
estimand, so the model's ERC and this ground truth are directly comparable.

WHY IT CANNOT DRIFT FROM THE GENERATOR
--------------------------------------
The per-(day, node) rate `lambda_{z,t}(a)` comes from `src.synthetic_causal.expected_rate_grid`,
which is the same rate the data generator draws from (verified: regeneration is byte-identical and
the OLS check recovers the exact beta/gammas). `exposure_override=a` is the do(exposure=a)
intervention; confounders are still read from disk unchanged. Change the DGP and the ground truth
follows automatically — nothing about the specific terms is hard-coded here.

FORECAST WINDOW
---------------
`forecast` selects the days T_fc summed per node. To compare against a particular model run, set it
to that run's forecast horizon via conf:  synthetic.erc.forecast_start / forecast_len. Default sums
over the whole year (per-node annual expected count, averaged over nodes).

Run:
    PYTHONPATH=. python -m src.ground_truth_erc year=2010 synthetic.var_name=diabetes
Outputs: outputs/erc_ground_truth_<var>_<year>.csv  and  outputs/erc_ground_truth_<var>_<year>.meta.json
"""

import json
import logging
import os

import hydra
import numpy as np
from omegaconf import OmegaConf
import pandas as pd

from src.synthetic_causal import expected_rate_grid, offset_vector, marginal_outcome
from src.synthetic_denom import get_zcta_data_with_geo_pop

LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")


@hydra.main(config_path="../conf/synthetic", config_name="config", version_base=None)
def main(cfg):
    year = int(cfg.year)
    var = cfg.synthetic.var_name
    n_days = 366 if (year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)) else 365

    # same zcta grid + offset the generator uses
    zcta_data = get_zcta_data_with_geo_pop(
        unique_fpath=cfg.synthetic.zcta_unique_path,
        shapefile_fpath=cfg.synthetic.zcta_shapefile_path,
        population_fpath=cfg.synthetic.population_path,
        year=year, mainland_only=cfg.synthetic.mainland_only,
    )
    offset = offset_vector(cfg, zcta_data)
    n_z = len(offset)

    # forecast window T_fc (days summed per node) + how to aggregate over nodes
    erc_cfg = cfg.synthetic.get("erc", {}) or {}
    fc_start = int(erc_cfg.get("forecast_start", 0))
    fc_len = int(erc_cfg.get("forecast_len", n_days - fc_start))
    forecast = slice(fc_start, fc_start + fc_len)
    node_agg = str(erc_cfg.get("node_aggregation", "mean"))

    # exposure grid: span the realistic exposure range (1st-99th pct), unless pinned in conf
    grid_cfg = erc_cfg.get("grid", None)
    if grid_cfg is not None:
        xs = np.linspace(float(grid_cfg.x_min), float(grid_cfg.x_max), int(grid_cfg.n_points))
    else:
        exp_root = cfg.synthetic.get("exposure_covars_root", "data/covars")
        exp_var = cfg.synthetic.exposure_var
        pm = np.load(f"{exp_root}/{cfg.synthetic.exposure_var_group}/{exp_var}/{exp_var}__{year}.npy")
        x_lo, x_hi = np.nanpercentile(pm, [1, 99])
        xs = np.linspace(float(x_lo), float(x_hi), 25)

    LOGGER.info(f"Ground-truth ERC: {len(xs)} exposure levels in [{xs[0]:.2f}, {xs[-1]:.2f}]; "
                f"forecast days [{fc_start}, {fc_start + fc_len}); node aggregation = {node_agg}")

    # outer loop over exposure a; inner collapse = sum over forecast days per node, then avg nodes
    mu = np.array([
        marginal_outcome(
            expected_rate_grid(cfg, zcta_data, exposure_override=a),   # lambda_{z,t}(a), same fn as the generator
            offset, forecast=forecast, node_aggregation=node_agg,
        )
        for a in xs
    ])

    os.makedirs("outputs", exist_ok=True)
    csv_path = f"outputs/erc_ground_truth_{var}_{year}.csv"
    pd.DataFrame({"exposure": xs, "erc_true": mu}).to_csv(csv_path, index=False)

    meta = {
        "var": var, "year": year,
        "estimand": "F(a) = node-aggregate over z of [ sum_{t in forecast} E[Y_{z,t}|do(exposure=a)] ]",
        "node_aggregation": node_agg,
        "forecast_window_days": [fc_start, fc_start + fc_len],
        "n_nodes": int(n_z),
        "scale": ("per-node windowed expected count, averaged over nodes" if node_agg == "mean"
                  else "total windowed expected count summed over nodes"),
        "exposure_variable": f"{cfg.synthetic.exposure_var_group}/{cfg.synthetic.exposure_var}",
        "beta_true": float(cfg.synthetic.get("beta", 0.0)),
        "confounders": OmegaConf.to_container(cfg.synthetic.get("confounders", []) or [], resolve=True),
        "rate_floor": 0.01,
        "exposure_range": [float(xs[0]), float(xs[-1])],
        "n_exposure_points": int(len(xs)),
        "source": "src/synthetic_causal.expected_rate_grid (same rate the generator draws from)",
    }
    meta_path = f"outputs/erc_ground_truth_{var}_{year}.meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nsaved ground-truth ERC -> {csv_path}")
    print(f"saved metadata        -> {meta_path}")
    print(f"  exposure x in [{xs[0]:.2f}, {xs[-1]:.2f}] ({len(xs)} pts), forecast {fc_len} day(s), node-{node_agg}")
    print(f"  F_true(x) in [{mu.min():.2f}, {mu.max():.2f}]  ({meta['scale']})\n")


if __name__ == "__main__":
    main()
