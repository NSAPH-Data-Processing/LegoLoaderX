"""Ground truth for the ROLLING-FORECAST g-computation estimand (ERC levels *and* delta-shifts).

This is the window-aware sibling of ``src/ground_truth_erc.py``. That script sums each calendar
day exactly once over one forecast slice; the estimand the model-side evaluation actually targets
sweeps a forecast start day over the whole year and sums a whole window off each start:

    G_o(delta) = (1/V) sum_{z in V} sum_{t0} sum_{tau} sum_{k}
                     [ f( X^(t0), A^(t0)[ m* -> delta * A_m*^(t0) ] ) ]_{tau,k,z,o}
    G_o(1) = factual.

Three things make that different from a plain sum over days, and all three are handled here.

1. MULTIPLICITY. Prediction ``(t0, tau, k)`` targets calendar day ``s = t0 + tau + k`` (see
   ``synthetic_causal.forecast_window_weights`` for where that alignment comes from), so an
   interior day is summed ``window * n_forecast`` times, with a ramp at both ends of the year.

2. INTERVENED vs. CARRIED days. ``A^(t0)`` is only the sample's own ``window`` input days, so the
   intervention touches days ``t0..t0+window-1``. A prediction with ``tau + k >= window`` targets a
   day whose exposure stayed factual. With this DGP's SAME-DAY dose-response those predictions
   carry no causal contrast at all, so they enter ``G(delta)`` at their factual value and dilute
   the curve. ``intervention_scope`` picks the reading:

     * ``window`` (default) — the literal estimand above; only own-window days are scaled.
     * ``global``            — the whole exposure field is scaled before forecasting. Use this if
                               the evaluation rescales the covariate store rather than the batch.

   The two differ by a known factor: with ``window=3, n_forecast=5`` the window-local contrast is
   ``6/15 = 40%`` of the global one. Comparing a model against the wrong one is a 2.5x error.

3. UNITS OF delta. ``legoloaderx.XDataset`` serves every covariate standardized, so multiplying
   the tensor the model receives implements ``X -> mean + delta*(X - mean)``, not ``X -> delta*X``.
   ``shift_space`` selects which one the ground truth uses; ``standardized`` reads the mean back
   from the store's ``summary_statistics.json`` so it matches the loader exactly.

Everything is evaluated from ``synthetic_causal.expected_rate_grid`` — the same deterministic rate
the generator draws from — so this is exact, not Monte Carlo, including the ``rate_floor`` clamp.
The model is direct multi-horizon (one forward pass emits every lead day; no prediction is fed back
as an input), so there is no rollout to simulate and no error to compound.

Run:
    PYTHONPATH=. python -m src.ground_truth_gcomp year=2013 synthetic.var_name=diabetes
    PYTHONPATH=. python -m src.ground_truth_gcomp year=2013 synthetic.var_name=diabetes \
        synthetic.gcomp.mode=level synthetic.gcomp.intervention_scope=global

Outputs: <output_dir>/gcomp_<mode>_<var>_<year>.csv  and  .meta.json
"""

import json
import logging
import os

import hydra
import numpy as np
from omegaconf import OmegaConf
import pandas as pd

from src.synthetic_causal import (
    expected_rate_grid,
    forecast_window_weights,
    offset_vector,
    windowed_outcome,
)
from src.synthetic_denom import get_zcta_data_with_geo_pop
from src.synthetic_manifest import apply_dgp_from_manifest, load_manifest

LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

DEFAULTS = {
    "mode": "shift",                  # 'shift' (G(delta), G(1)=factual) | 'level' (do(X=a) ERC)
    "intervention_scope": "window",   # 'window' (only A^(t0)) | 'global' (whole exposure field)
    "shift_space": "raw",             # 'raw' (X -> delta*X) | 'standardized' (X -> m + delta*(X-m))
    "window": 3,                      # T_pred — dataset.window on the model side
    "n_forecast": 5,                  # T_fc  — model.max_forecast_steps
    "t0_start": 0,
    "t0_end": None,                   # exclusive; None + drop_incomplete => every window fits the year
    "drop_incomplete": True,
    "node_aggregation": "mean",
    "output_dir": "outputs",
    "n_points": 25,
    "delta_min": 0.5,
    "delta_max": 1.5,
}


def _exposure_mean(cfg, year):
    """Mean of the exposure the loader normalizes by — ``summary_statistics.json`` if present
    (that is what ``XDataset`` uses), else computed from this year's file."""
    root = cfg.synthetic.get("exposure_covars_root", "data/covars")
    vg, var = cfg.synthetic.exposure_var_group, cfg.synthetic.exposure_var
    stats_path = f"{root}/summary_statistics.json"
    if os.path.exists(stats_path):
        with open(stats_path) as f:
            entry = (json.load(f).get(vg) or {}).get(var)
        if entry is not None and entry.get("mean") is not None:
            LOGGER.info(f"shift_space=standardized: centering on {stats_path} mean={entry['mean']:.4f}")
            return float(entry["mean"])
    arr = np.load(f"{root}/{vg}/{var}/{var}__{year}.npy")
    m = float(np.nanmean(arr))
    LOGGER.warning(f"{vg}/{var} absent from summary_statistics.json; centering on this year's mean {m:.4f}")
    return m


@hydra.main(config_path="../conf/synthetic", config_name="config", version_base=None)
def main(cfg):
    year = int(cfg.year)
    var = cfg.synthetic.var_name
    n_days = 366 if (year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)) else 365

    # The answer key must describe the data on disk, not whatever the live config says today.
    g0 = cfg.synthetic.get("gcomp", {}) or {}
    if g0.get("use_manifest", True):
        manifest = load_manifest(var, year, path=g0.get("manifest_path", None))
        if manifest is not None:
            cfg = apply_dgp_from_manifest(cfg, manifest)
            LOGGER.info(f"DGP from manifest generated_at={manifest.get('generated_at_utc')} "
                        f"git={manifest.get('git_commit')}")
        else:
            LOGGER.warning("No manifest beside the data; using the LIVE config as ground truth.")

    g = dict(DEFAULTS)
    g.update(OmegaConf.to_container(cfg.synthetic.get("gcomp", {}) or {}, resolve=True))
    mode, scope, space = g["mode"], g["intervention_scope"], g["shift_space"]
    if mode not in ("shift", "level"):
        raise ValueError(f"gcomp.mode must be 'shift' or 'level', got {mode!r}")
    if scope not in ("window", "global"):
        raise ValueError(f"gcomp.intervention_scope must be 'window' or 'global', got {scope!r}")

    zcta_data = get_zcta_data_with_geo_pop(
        unique_fpath=cfg.synthetic.zcta_unique_path,
        shapefile_fpath=cfg.synthetic.zcta_shapefile_path,
        population_fpath=cfg.synthetic.population_path,
        year=year, mainland_only=cfg.synthetic.mainland_only,
    )
    offset = offset_vector(cfg, zcta_data)
    p_norm = float(cfg.synthetic.poisson_params.population_normalizer)

    w_all, w_int = forecast_window_weights(
        n_days, window=int(g["window"]), n_forecast=int(g["n_forecast"]),
        t0_start=int(g["t0_start"]), t0_end=g["t0_end"], drop_incomplete=bool(g["drop_incomplete"]),
    )
    n_pred = float(w_all.sum())
    frac_int = float(w_int.sum() / n_pred) if n_pred else 0.0
    LOGGER.info(
        f"estimand: window={g['window']} n_forecast={g['n_forecast']} -> {int(n_pred)} summed "
        f"(t0,tau,k) predictions per node; {100 * frac_int:.1f}% of them target a day whose own "
        f"exposure is intervened (scope={scope})"
    )

    # factual rate grid: the carried (non-intervened) days read from this, and it is also G(1).
    rate_factual = expected_rate_grid(cfg, zcta_data)

    if mode == "shift":
        xs = np.linspace(float(g["delta_min"]), float(g["delta_max"]), int(g["n_points"]))
        center = _exposure_mean(cfg, year) if space == "standardized" else 0.0
        if space not in ("raw", "standardized"):
            raise ValueError(f"gcomp.shift_space must be 'raw' or 'standardized', got {space!r}")
        grids = (expected_rate_grid(cfg, zcta_data, exposure_scale=float(d), scale_center=center) for d in xs)
        x_name, x_label = "delta", "multiplicative shift delta (1 = factual)"
    else:
        grid_cfg = g.get("grid", None)
        if grid_cfg is not None:
            xs = np.linspace(float(grid_cfg["x_min"]), float(grid_cfg["x_max"]), int(grid_cfg["n_points"]))
        else:
            root = cfg.synthetic.get("exposure_covars_root", "data/covars")
            ev = cfg.synthetic.exposure_var
            arr = np.load(f"{root}/{cfg.synthetic.exposure_var_group}/{ev}/{ev}__{year}.npy")
            lo, hi = np.nanpercentile(arr, [1, 99])
            xs = np.linspace(float(lo), float(hi), int(g["n_points"]))
        center = 0.0
        grids = (expected_rate_grid(cfg, zcta_data, exposure_override=float(a)) for a in xs)
        x_name, x_label = "exposure", "do(exposure = a)"

    carried = None if scope == "global" else rate_factual
    rows = []
    for x, grid in zip(xs, grids):
        g_count = windowed_outcome(grid, carried, offset, w_all, w_int, g["node_aggregation"])
        g_rate = windowed_outcome(grid, carried, np.full_like(offset, p_norm),
                                  w_all, w_int, g["node_aggregation"])
        rows.append((float(x), g_count, g_rate))

    df = pd.DataFrame(rows, columns=[x_name, "g_true_count", "g_true_rate"])
    # G(1) / the factual reference, so contrasts can be read straight off the CSV.
    factual = windowed_outcome(rate_factual, None, offset, w_all, w_all, g["node_aggregation"])
    df["contrast_vs_factual"] = df["g_true_count"] - factual

    out_dir = str(g["output_dir"])
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, f"gcomp_{mode}_{var}_{year}.csv")
    df.to_csv(csv_path, index=False)

    meta = {
        "var": var, "year": year, "mode": mode,
        "estimand": ("G(x) = agg_z offset_z * sum_s [ w_int[s]*lambda_int[s,z] + "
                     "(w_all[s]-w_int[s])*lambda_factual[s,z] ]"),
        "x_axis": x_label,
        "intervention_scope": scope,
        "shift_space": space if mode == "shift" else None,
        "shift_center": float(center),
        "window_T_pred": int(g["window"]),
        "n_forecast_T_fc": int(g["n_forecast"]),
        "t0_range": [int(g["t0_start"]), int(g["t0_end"]) if g["t0_end"] is not None
                     else n_days - (int(g["window"]) + int(g["n_forecast"]) - 2)],
        "predictions_per_node": int(n_pred),
        "frac_predictions_intervened": frac_int,
        "node_aggregation": g["node_aggregation"],
        "n_nodes": int(len(offset)),
        "factual_G1_count": factual,
        "scale_count": "expected windowed COUNT per node, averaged over nodes",
        "scale_rate": (f"same sum with offset replaced by population_normalizer={p_norm}; compare "
                       "against the model's per-person ZINB mean (1-pi)*mu summed over the window"),
        "exposure_variable": f"{cfg.synthetic.exposure_var_group}/{cfg.synthetic.exposure_var}",
        "beta_true": float(cfg.synthetic.get("beta", 0.0)),
        "exposure_shape": str(cfg.synthetic.get("exposure_shape", "linear")),
        "confounders": OmegaConf.to_container(cfg.synthetic.get("confounders", []) or [], resolve=True),
        "spacetime": OmegaConf.to_container(cfg.synthetic.get("spacetime", {}) or {}, resolve=True),
        "rate_floor": float(cfg.synthetic.poisson_params.get("rate_floor", 0.01)),
        "source": "src/synthetic_causal.expected_rate_grid (same rate the generator draws from)",
    }
    meta_path = os.path.join(out_dir, f"gcomp_{mode}_{var}_{year}.meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nsaved windowed ground truth -> {csv_path}")
    print(f"saved metadata              -> {meta_path}")
    print(f"  {x_label}, {len(xs)} pts in [{xs[0]:.3f}, {xs[-1]:.3f}]")
    print(f"  scope={scope}  {int(n_pred)} predictions/node, {100 * frac_int:.1f}% intervened")
    print(f"  G(factual) = {factual:.4f} counts/node;  G range "
          f"[{df.g_true_count.min():.4f}, {df.g_true_count.max():.4f}]\n")


if __name__ == "__main__":
    main()
