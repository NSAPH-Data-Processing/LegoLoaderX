"""DGP-agnostic assessment: how close is a model's ERC to the ground-truth ERC?

This script knows NOTHING about the data-generating process. It takes two plain curves —
  * a ground-truth ERC  (from src/ground_truth_erc.py), and
  * a model's ERC        (whatever your model produced),
each a CSV of (exposure, value) — aligns them on a common exposure grid by linear interpolation,
and reports accuracy metrics. Because it only consumes two curves, it works for ANY DGP and ANY
model that can emit an exposure-response curve. Swap in a different DGP or a different model and
nothing here changes.

Column names are auto-detected (exposure in {exposure,x,pm25,dose,level}; value in
{erc,mu,value,y,pred,prediction,rate,count}) or set explicitly via flags.

Examples:
    PYTHONPATH=. python -m src.assess_erc \
        --truth outputs/erc_ground_truth_diabetes_2012.csv \
        --model outputs/erc_diabetes_2012.csv --model-col erc_adjusted

    # compare the NAIVE (confounded) estimate to show the framework catches a bad ERC:
    PYTHONPATH=. python -m src.assess_erc \
        --truth outputs/erc_ground_truth_diabetes_2012.csv \
        --model outputs/erc_diabetes_2012.csv --model-col erc_naive
"""

import argparse
import json
import os

import numpy as np
import pandas as pd

_EXPOSURE_NAMES = ["exposure", "x", "pm25", "dose", "level", "conc"]
# Model-prediction column candidates. Deliberately does NOT include 'erc_true' (that is the
# ground-truth column name) so auto-detection can't silently score truth-vs-truth.
_VALUE_NAMES = ["prediction", "pred", "erc_adjusted", "mu", "value", "y", "rate", "count", "erc"]


def _pick_column(df, preferred, candidates, role):
    """Choose a column: explicit `preferred` if given, else the first candidate present."""
    if preferred is not None:
        if preferred not in df.columns:
            raise ValueError(f"column '{preferred}' not in {list(df.columns)}")
        return preferred
    for c in candidates:
        if c in df.columns:
            return c
    # last resort: first column for exposure, second for value
    fallback = df.columns[0 if role == "exposure" else min(1, len(df.columns) - 1)]
    return fallback


def _load_curve(path, x_col, v_col):
    """Load a (exposure, value) curve from CSV, sorted by exposure."""
    df = pd.read_csv(path)
    xc = _pick_column(df, x_col, _EXPOSURE_NAMES, "exposure")
    vc = _pick_column(df, v_col, _VALUE_NAMES, "value")
    out = df[[xc, vc]].rename(columns={xc: "x", vc: "v"}).dropna().sort_values("x")
    return out["x"].to_numpy(float), out["v"].to_numpy(float), (xc, vc)


def main():
    ap = argparse.ArgumentParser(description="Score a model ERC against a ground-truth ERC (DGP-agnostic).")
    ap.add_argument("--truth", required=True, help="ground-truth ERC CSV (exposure, erc_true)")
    ap.add_argument("--model", required=True, help="model ERC CSV (exposure, prediction)")
    ap.add_argument("--truth-col", default="erc_true", help="value column in the truth CSV")
    ap.add_argument("--model-col", default=None, help="value column in the model CSV (auto-detect if omitted)")
    ap.add_argument("--x-col", default=None, help="exposure column name in both CSVs (auto-detect if omitted)")
    ap.add_argument("--n-grid", type=int, default=200, help="points on the shared comparison grid")
    ap.add_argument("--out", default=None, help="metrics JSON path (default: outputs/erc_assessment_<model>.json)")
    ap.add_argument("--plot", default=None, help="comparison PNG path (default alongside --out)")
    args = ap.parse_args()

    # --- load both curves ---
    xt, vt, (txc, tvc) = _load_curve(args.truth, args.x_col, args.truth_col)
    xm, vm, (mxc, mvc) = _load_curve(args.model, args.x_col, args.model_col)

    # guard against accidentally scoring the ground truth against itself
    if args.model == args.truth and mvc == tvc:
        raise ValueError(f"model column '{mvc}' == truth column on the same file -> would score "
                         f"truth-vs-truth. Pass --model-col explicitly (e.g. erc_adjusted).")
    if mvc == "erc_true":
        print(f"WARNING: model column resolved to 'erc_true' (the ground-truth column name). "
              f"Pass --model-col to point at your model's prediction column.")

    # --- align on a common exposure grid (the overlap of the two ranges), interpolate both ---
    lo, hi = max(xt.min(), xm.min()), min(xt.max(), xm.max())
    if not (hi > lo):
        raise ValueError(f"no overlapping exposure range: truth [{xt.min()},{xt.max()}] vs model [{xm.min()},{xm.max()}]")
    grid = np.linspace(lo, hi, args.n_grid)
    t = np.interp(grid, xt, vt)     # ground truth on the grid
    m = np.interp(grid, xm, vm)     # model on the grid
    diff = m - t

    # --- accuracy metrics (all on the shared grid) ---
    rng = t.max() - t.min()
    ss_res = float(np.sum(diff ** 2))
    ss_tot = float(np.sum((t - t.mean()) ** 2))
    # slopes via a straight-line fit to each curve (handy when the ERC is ~linear)
    slope_t = float(np.polyfit(grid, t, 1)[0])
    slope_m = float(np.polyfit(grid, m, 1)[0])
    metrics = {
        "rmse": float(np.sqrt(np.mean(diff ** 2))),
        "mae": float(np.mean(np.abs(diff))),
        "max_abs_error": float(np.max(np.abs(diff))),
        "mean_bias": float(np.mean(diff)),                          # signed: + = model over-predicts
        "rmse_relative_to_range": float(np.sqrt(np.mean(diff ** 2)) / rng) if rng else None,
        "integrated_abs_error": float(np.trapz(np.abs(diff), grid)),
        "r2": float(1 - ss_res / ss_tot) if ss_tot else None,       # 1.0 = perfect; can go negative
        "slope_true": slope_t,
        "slope_model": slope_m,
        "slope_ratio": float(slope_m / slope_t) if slope_t else None,
        "exposure_overlap": [float(lo), float(hi)],
        "n_grid": int(args.n_grid),
        "truth_file": args.truth, "truth_col": tvc,
        "model_file": args.model, "model_col": mvc,
    }
    # a coarse verdict purely from relative RMSE (tune to taste)
    rel = metrics["rmse_relative_to_range"]
    metrics["verdict"] = ("excellent" if rel is not None and rel < 0.02 else
                          "good" if rel is not None and rel < 0.05 else
                          "fair" if rel is not None and rel < 0.15 else "poor")

    # --- report ---
    print("\n" + "=" * 60)
    print(f"ERC assessment   model[{mvc}]  vs  truth[{tvc}]")
    print("=" * 60)
    print(f"  exposure overlap     : [{lo:.2f}, {hi:.2f}]")
    print(f"  RMSE                 : {metrics['rmse']:.4f}")
    print(f"  MAE                  : {metrics['mae']:.4f}")
    print(f"  max abs error        : {metrics['max_abs_error']:.4f}")
    print(f"  mean bias (signed)   : {metrics['mean_bias']:+.4f}")
    print(f"  RMSE / truth-range   : {rel:.4f}" if rel is not None else "  RMSE / truth-range   : n/a")
    print(f"  R^2 (vs truth)       : {metrics['r2']:.4f}" if metrics['r2'] is not None else "  R^2 : n/a")
    print(f"  slope  true / model  : {slope_t:+.4f} / {slope_m:+.4f}  (ratio {metrics['slope_ratio']:.3f})")
    print(f"  VERDICT              : {metrics['verdict'].upper()}")
    print("=" * 60)

    # --- save metrics + comparison plot ---
    os.makedirs("outputs", exist_ok=True)
    base = os.path.splitext(os.path.basename(args.model))[0]
    out_json = args.out or f"outputs/erc_assessment_{base}_{mvc}.json"
    with open(out_json, "w") as f:
        json.dump(metrics, f, indent=2)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.figure(figsize=(7, 5))
    plt.plot(grid, t, "k-", lw=3, label=f"ground truth [{tvc}]")
    plt.plot(grid, m, "o--", color="tab:green", label=f"model [{mvc}]")
    plt.fill_between(grid, t, m, color="tab:red", alpha=0.15, label="error")
    plt.xlabel("exposure")
    plt.ylabel("ERC  μ(x)")
    plt.title(f"ERC assessment — RMSE {metrics['rmse']:.3f} ({metrics['verdict']})")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    out_png = args.plot or f"outputs/erc_assessment_{base}_{mvc}.png"
    plt.savefig(out_png, dpi=130)
    print(f"saved metrics -> {out_json}")
    print(f"saved plot    -> {out_png}\n")


if __name__ == "__main__":
    main()
