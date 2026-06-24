"""Demo exposure-response curve (ERC) by g-computation — a stand-in "model" for the assessment.

This fits a simple linear model to the generated counts and traces its ERC, to exercise the
ground-truth + assessment pipeline. Your real model (e.g. ClimHealth) replaces the fitted model
here; everything else — the ground truth and the scoring — is identical.

The ERC is computed with the SAME collapse as src/ground_truth_erc.py and the real model:
  for each exposure level a:  set exposure=a everywhere, predict the per-(day, node) rate, then
    per node  SUM the expected count over the forecast window,  and AVERAGE over nodes.

Three curves are produced and saved:
  TRUE      -- from the DGP (expected_rate_grid); identical to ground_truth_erc.py. The target.
  ADJUSTED  -- a model that adjusts for the confounders; should land on TRUE.
  NAIVE     -- a model using the exposure only; comes out biased (the confounding made visible).

Run (after a FULL-year generation of the same year/var):
    PYTHONPATH=. python -m src.erc_curve year=2010 synthetic.var_name=diabetes
Outputs: outputs/erc_<var>_<year>.png  and  outputs/erc_<var>_<year>.csv
"""

import logging
import os

import hydra
import numpy as np
import pandas as pd

from src.synthetic_causal import (
    _load_covar_aligned, expected_rate_grid, offset_vector, marginal_outcome,
)
from src.synthetic_denom import get_zcta_data_with_geo_pop

LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")


def _ols(X, y):
    """Ordinary least squares: best-fit coefficients b for y ~= X @ b (X includes an intercept)."""
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    return b


@hydra.main(config_path="../conf/synthetic", config_name="config", version_base=None)
def main(cfg):
    year, var = int(cfg.year), cfg.synthetic.var_name
    p = cfg.synthetic.poisson_params
    cov_root = cfg.synthetic.get("exposure_covars_root", "data/covars")
    confounders = cfg.synthetic.get("confounders", None) or []
    n_days = 366 if (year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)) else 365

    # forecast window + node aggregation: MUST match ground_truth_erc.py for comparability
    erc_cfg = cfg.synthetic.get("erc", {}) or {}
    fc_start = int(erc_cfg.get("forecast_start", 0))
    fc_len = int(erc_cfg.get("forecast_len", n_days - fc_start))
    forecast = slice(fc_start, fc_start + fc_len)
    node_agg = str(erc_cfg.get("node_aggregation", "mean"))

    # --- grid + generated counts + the exact rate components (full (n_days, n_z) grids) ---
    zcta_data = get_zcta_data_with_geo_pop(
        unique_fpath=cfg.synthetic.zcta_unique_path,
        shapefile_fpath=cfg.synthetic.zcta_shapefile_path,
        population_fpath=cfg.synthetic.population_path,
        year=year, mainland_only=cfg.synthetic.mainland_only,
    )
    zctas = zcta_data["zcta"].astype(str).to_numpy()
    n_z = len(zctas)
    offset = offset_vector(cfg, zcta_data)

    z2row = {z: i for i, z in enumerate(zctas)}
    sparse = f"data/input/lego/medicare_synthetic/medpar_outcomes/ccw/zcta_daily/sparse_counts_{var}_{year}.parquet"
    df = pd.read_parquet(sparse)
    rows = df["zcta"].astype(str).map(z2row).to_numpy()
    keep = ~pd.isna(rows)
    Y = np.zeros((n_days, n_z), dtype=np.float64)
    Y[(pd.to_datetime(df["date"]).dt.dayofyear.to_numpy() - 1)[keep], rows[keep].astype(np.int64)] = df["n"].to_numpy()[keep]
    if pd.to_datetime(df["date"]).dt.dayofyear.nunique() < n_days:
        LOGGER.warning("Fewer days than a full year present -> looks like a debug run; regenerate the FULL year.")

    seasonal = p.seasonal_amplitude * np.sin(2 * np.pi * np.arange(n_days) / 365.25)         # (n_days,)
    lat_eff = p.latitude_effect * np.sin(((zcta_data["latitude"].to_numpy() - 35) / 15) * np.pi)  # (n_z,)
    lon_eff = p.longitude_effect * np.cos(((zcta_data["longitude"].to_numpy() + 95) / 30) * np.pi)
    pm25 = _load_covar_aligned(cov_root, cfg.synthetic.exposure_var_group, cfg.synthetic.exposure_var,
                               year, zctas, n_days, temporal_res="daily", standardize=False)  # (n_days, n_z)
    std_confs = [
        _load_covar_aligned(cov_root, c.var_group, c.var, year, zctas, n_days,
                            temporal_res=c.temporal_res, standardize=True)
        for c in confounders
    ]

    # --- FIT the demo models on the per-capita rate r = Y/offset, on a random subsample of cells ---
    valid = np.broadcast_to((offset > 0)[None, :], (n_days, n_z)).ravel()
    idx = np.flatnonzero(valid)
    rng = np.random.default_rng(0)
    if idx.size > 1_000_000:
        idx = rng.choice(idx, 1_000_000, replace=False)

    def sub(grid):
        return np.broadcast_to(grid, (n_days, n_z)).ravel()[idx]

    r = (Y.ravel()[idx]) / sub(offset[None, :])
    X_adj = np.column_stack([np.ones(idx.size), sub(seasonal[:, None]), sub(lat_eff[None, :]),
                             sub(lon_eff[None, :]), sub(pm25), *[sub(sc) for sc in std_confs]])
    b_adj = _ols(X_adj, r)
    b0, b_seas, b_lat, b_lon, beta_hat = b_adj[:5]
    b_conf = b_adj[5:]
    b_naive = _ols(np.column_stack([np.ones(idx.size), sub(pm25)]), r)
    const_naive, beta_naive = b_naive[0], b_naive[1]

    # fitted ADJUSTED nuisance as a FULL grid (everything except the PM25 term) -> add beta_hat*a later
    nuis_hat = (b0 + b_seas * seasonal[:, None] + b_lat * lat_eff[None, :] + b_lon * lon_eff[None, :]).astype(np.float64)
    for bk, sc in zip(b_conf, std_confs):
        nuis_hat = nuis_hat + bk * sc

    # --- exposure grid: honor a config-pinned grid if present (MUST match ground_truth_erc.py),
    #     else the same 1st-99th percentile range with 25 points the ground truth uses ---
    grid_cfg = erc_cfg.get("grid", None)
    if grid_cfg is not None:
        xs = np.linspace(float(grid_cfg.x_min), float(grid_cfg.x_max), int(grid_cfg.n_points))
    else:
        pm_raw = np.load(f"{cov_root}/{cfg.synthetic.exposure_var_group}/{cfg.synthetic.exposure_var}/{cfg.synthetic.exposure_var}__{year}.npy")
        x_lo, x_hi = np.nanpercentile(pm_raw, [1, 99])
        xs = np.linspace(float(x_lo), float(x_hi), 25)

    # --- g-computation with the shared collapse (sum over forecast days per node, then avg nodes) ---
    erc_true = np.array([marginal_outcome(expected_rate_grid(cfg, zcta_data, exposure_override=a),
                                          offset, forecast=forecast, node_aggregation=node_agg) for a in xs])
    erc_adj = np.array([marginal_outcome(np.maximum(0.01, nuis_hat + beta_hat * a),
                                         offset, forecast=forecast, node_aggregation=node_agg) for a in xs])
    # NAIVE rate is constant across cells -> collapse analytically (avoids building a constant grid)
    agg_off = offset.mean() if node_agg == "mean" else offset.sum()
    erc_naive = np.array([fc_len * max(0.01, const_naive + beta_naive * a) * agg_off for a in xs])

    print("\n" + "=" * 64)
    print(f"ERC by g-computation  |  {var} {year}  |  forecast {fc_len}d, node-{node_agg}")
    print("=" * 64)
    print(f"PM25 slope (rate scale):  true beta={cfg.synthetic.get('beta',0.0):+.4f}  "
          f"adjusted={beta_hat:+.4f}  naive={beta_naive:+.4f}")
    print(f"{'PM2.5 (x)':>10}{'TRUE':>14}{'ADJUSTED':>14}{'NAIVE':>14}")
    for x, t, a, nv in list(zip(xs, erc_true, erc_adj, erc_naive))[::6]:
        print(f"{x:>10.2f}{t:>14.2f}{a:>14.2f}{nv:>14.2f}")
    print("=" * 64)

    os.makedirs("outputs", exist_ok=True)
    csv_path = f"outputs/erc_{var}_{year}.csv"
    pd.DataFrame({"exposure": xs, "erc_true": erc_true, "erc_adjusted": erc_adj, "erc_naive": erc_naive}
                 ).to_csv(csv_path, index=False)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.figure(figsize=(7, 5))
    plt.plot(xs, erc_true, "k-", lw=3, label="TRUE")
    plt.plot(xs, erc_adj, "o--", color="tab:green", label=f"ADJUSTED (β̂={beta_hat:+.3f})")
    plt.plot(xs, erc_naive, "s--", color="tab:red", label=f"NAIVE (β̂={beta_naive:+.3f})")
    plt.xlabel("PM2.5 exposure  x  (µg/m³)")
    plt.ylabel(f"ERC  μ(x)  [{'per-node' if node_agg=='mean' else 'total'} windowed count]")
    plt.title(f"Exposure-response curve — {var} {year}")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    png_path = f"outputs/erc_{var}_{year}.png"
    plt.savefig(png_path, dpi=130)
    print(f"saved: {png_path}\nsaved: {csv_path}\n")


if __name__ == "__main__":
    main()
