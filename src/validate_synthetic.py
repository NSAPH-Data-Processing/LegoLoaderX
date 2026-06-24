"""Double-check the closed-form DGP of the semi-synthetic outcome.

WHAT THIS SCRIPT IS FOR
-----------------------
We *generated* the synthetic outcome Y from a known formula (in synthetic_health.py +
synthetic_causal.py). The coefficients in that formula -- the treatment effect ``beta`` and the
confounder effects ``gamma_k`` -- are our "ground truth". This script does the reverse: it
pretends it does NOT know those numbers and tries to *estimate them back* from the generated
data. If the estimates come out equal to what we put in, then the data really does follow the
formula we think it does, and the ground truth is trustworthy.

WHY A SIMPLE LINEAR REGRESSION CAN RECOVER THEM
-----------------------------------------------
Each outcome count is drawn as   Y_{i,t} ~ Poisson( lambda_{i,t} * offset_i )   where
``offset_i = population_normalizer * population_i`` and the *rate* is built additively:

    lambda_{i,t} = base + seasonal_t + lat_i + lon_i
                   + beta * PM25_{i,t} + sum_k gamma_k * std(C^k_{i,t})

A Poisson count has mean equal to its parameter, so  E[Y_{i,t}] = lambda_{i,t} * offset_i.
Divide both sides by the (known) offset:

    E[ Y_{i,t} / offset_i ] = lambda_{i,t}    = a plain linear sum of the terms above.

So if we form the response  r = Y/offset  and regress it on those exact terms, ordinary least
squares (OLS) must return the coefficients used to build lambda:
    intercept ~ base_rate,  the geo/season terms ~ 1.0,  PM25 ~ beta,  each confounder ~ gamma_k.
The single Poisson draw per cell is just mean-zero noise around the line, which OLS averages out.

THE THREE CHECKS PRINTED
------------------------
1. ADJUSTED fit  -- regress on ALL terms -> should recover every coefficient (incl. beta).
2. NAIVE fit     -- regress on PM25 ONLY -> beta comes out BIASED, because PM25 is correlated
                    with the confounders we left out. This is exactly the confounding bias that
                    a correct ERC / g-computation pipeline must remove (adjusting removes it).
3. FORWARD check -- independently, sum up the predicted counts and compare to the observed total.

Run (AFTER generating a FULL year of the same year/var -- not a 3-day debug run):
    PYTHONPATH=. python -m src.validate_synthetic year=2010 synthetic.var_name=diabetes
"""

import calendar
import logging

import hydra
import numpy as np
import pandas as pd

# _load_covar_aligned: read a covariate .npy from data/covars and return it as a
# (n_days, n_zctas) grid lined up with our zcta order (same helper the generator used, so the
# numbers here are identical to the ones that built the outcome).
from src.synthetic_causal import _load_covar_aligned
# get_zcta_data_with_geo_pop: the exact zcta list + lat/lon + population the generator started
# from, so our reconstructed grid matches cell-for-cell.
from src.synthetic_denom import get_zcta_data_with_geo_pop

LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")


def _ols(X, y):
    """Ordinary least squares: find the coefficients b that best fit  y ~= X @ b.

    X is the "design matrix" with one column per regressor (and one column of 1s for the
    intercept) and one row per observation; y is the response. np.linalg.lstsq returns the
    least-squares coefficients (the rest of its return tuple -- residuals, rank, etc. -- we drop).
    """
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return beta


@hydra.main(config_path="../conf/synthetic", config_name="config", version_base=None)
def main(cfg):
    # ---- read the ground truth straight from the config that generated the data ----
    # These are the numbers we expect the regression to recover.
    year = int(cfg.year)
    var = cfg.synthetic.var_name
    p = cfg.synthetic.poisson_params           # base_rate, seasonal_amplitude, lat/lon effects, etc.
    beta_true = float(cfg.synthetic.get("beta", 0.0))           # the treatment effect (PM25)
    cov_root = cfg.synthetic.get("exposure_covars_root", "data/covars")
    confounders = cfg.synthetic.get("confounders", None) or []  # list of {var_group,var,gamma,temporal_res}
    n_days = 366 if calendar.isleap(year) else 365

    # ---- STEP 1: rebuild the exact zcta grid the generator used ----
    # Same source -> same 32,657 mainland zctas, in the same order, with lat/lon/population.
    zcta_data = get_zcta_data_with_geo_pop(
        unique_fpath=cfg.synthetic.zcta_unique_path,
        shapefile_fpath=cfg.synthetic.zcta_shapefile_path,
        population_fpath=cfg.synthetic.population_path,
        year=year,
        mainland_only=cfg.synthetic.mainland_only,
    )
    zctas = zcta_data["zcta"].astype(str).to_numpy()
    n_z = len(zctas)
    z2row = {z: i for i, z in enumerate(zctas)}   # zcta string -> its row index in our grid

    # ---- STEP 2: load the generated counts into a dense (n_days, n_zctas) grid ----
    # The saved parquet is SPARSE: it only stores cells where the count was > 0. Every other
    # (zcta, day) really had a count of 0, so we start from an all-zeros grid and scatter the
    # nonzero counts into their (day, zcta) positions.
    sparse = f"data/input/lego/medicare_synthetic/medpar_outcomes/ccw/zcta_daily/sparse_counts_{var}_{year}.parquet"
    df = pd.read_parquet(sparse)
    rows = df["zcta"].astype(str).map(z2row).to_numpy()   # which grid row each record belongs to
    keep = ~pd.isna(rows)                                  # drop any zcta not in our grid
    rows = rows[keep].astype(np.int64)
    doys = (pd.to_datetime(df["date"]).dt.dayofyear.to_numpy() - 1)[keep]  # day-of-year 0..364
    Y = np.zeros((n_days, n_z), dtype=np.float64)
    Y[doys, rows] = df["n"].to_numpy()[keep]              # place each nonzero count
    n_obs_days = int(pd.to_datetime(df["date"]).dt.dayofyear.nunique())
    LOGGER.info(f"Loaded {sparse}: {len(df)} nonzero rows spanning {n_obs_days} day(s)")
    if n_obs_days < n_days:
        # A debug generation only writes the first 3 days, which is far too little to fit on.
        LOGGER.warning(f"Only {n_obs_days}/{n_days} days present -> looks like a debug run; "
                       f"regenerate the FULL year for a meaningful check.")

    # ---- STEP 3: rebuild each term of the rate, EXACTLY as the generator computed it ----
    # offset_i = population_normalizer * population_i  -> converts a per-capita rate into a count.
    pop = zcta_data["population"].to_numpy(dtype=np.float64)
    offset = p.population_normalizer * pop                                    # shape (n_z,)
    # Geographic effects: a fixed value per zcta (constant across days). Same sin/cos as the generator.
    lat_eff = p.latitude_effect * np.sin(((zcta_data["latitude"].to_numpy() - 35) / 15) * np.pi)
    lon_eff = p.longitude_effect * np.cos(((zcta_data["longitude"].to_numpy() + 95) / 30) * np.pi)
    # Seasonal effect: a fixed value per day-of-year (constant across zctas).
    doy = np.arange(n_days)
    seasonal = p.seasonal_amplitude * np.sin(2 * np.pi * doy / 365.25)        # shape (n_days,)
    # The treatment: real PM2.5, raw scale (NOT standardized), same array the model will train on.
    pm25 = _load_covar_aligned(cov_root, cfg.synthetic.exposure_var_group, cfg.synthetic.exposure_var,
                               year, zctas, n_days, temporal_res="daily", standardize=False)

    # ---- STEP 4: assemble the list of regressors (the columns of the design matrix) ----
    # Each entry is (name, a (n_days, n_z) grid of that term's value, the coefficient we EXPECT).
    # np.broadcast_to cheaply expands a per-day (or per-zcta) vector to the full grid without
    # copying: e.g. `seasonal` (one number per day) becomes the same number across every zcta.
    cols = [
        ("const",    np.ones((n_days, n_z)),                              p.base_rate),  # intercept
        ("seasonal", np.broadcast_to(seasonal[:, None], (n_days, n_z)),   1.0),  # per-day, all zctas
        ("lat_eff",  np.broadcast_to(lat_eff[None, :], (n_days, n_z)),    1.0),  # per-zcta, all days
        ("lon_eff",  np.broadcast_to(lon_eff[None, :], (n_days, n_z)),    1.0),
        ("PM25",     pm25,                                                beta_true),  # the treatment
    ]
    # Confounders: standardized exactly as in generation (so the recovered coef should equal gamma).
    for c in confounders:
        std_c = _load_covar_aligned(cov_root, c.var_group, c.var, year, zctas, n_days,
                                    temporal_res=c.temporal_res, standardize=True)
        cols.append((f"{c.var_group}/{c.var}", std_c, float(c.gamma)))

    # ---- STEP 5: form the response r = Y/offset and pick the cells to fit on ----
    # r = counts / offset = the per-capita rate, whose expectation is the linear sum above.
    valid_z = offset > 0   # a zcta with 0/NaN population would divide by zero -> exclude it
    r = np.where(valid_z[None, :], Y / np.where(valid_z, offset, 1.0)[None, :], np.nan)
    # We have n_days * n_z ~= 11.9 million cells. That's more than we need to estimate ~8
    # coefficients precisely, so we randomly subsample 2 million valid cells to keep it fast/light.
    # `ravel()` flattens the 2-D grid into one long 1-D vector; `flatnonzero` gives the positions
    # of the valid cells; `rng.choice(..., seed 0)` makes the subsample reproducible.
    flat_valid = np.broadcast_to(valid_z[None, :], (n_days, n_z)).ravel()
    idx = np.flatnonzero(flat_valid)
    rng = np.random.default_rng(0)
    if idx.size > 2_000_000:
        idx = rng.choice(idx, 2_000_000, replace=False)

    y = r.ravel()[idx]                                              # response for the chosen cells
    Xfull = np.column_stack([arr.ravel()[idx] for _, arr, _ in cols])  # one column per regressor

    # ---- STEP 6: the two fits ----
    # ADJUSTED: regress on every term -> should recover beta and every gamma (confounding removed).
    coef_adj = _ols(Xfull, y)
    # NAIVE: regress on just an intercept and PM25, ignoring the confounders -> beta comes out
    # biased, because PM25 is correlated with those omitted confounders. This is the whole point:
    # it shows confounding exists, and that adjusting (the ADJUSTED fit) is what fixes it.
    pm_j = [n for n, _, _ in cols].index("PM25")   # column index of PM25 inside Xfull
    Xnaive = np.column_stack([np.ones(idx.size), Xfull[:, pm_j]])
    coef_naive = _ols(Xnaive, y)

    # ---- STEP 7: print the comparison table (recovered vs ground truth) ----
    print("\n" + "=" * 72)
    print(f"Closed-form check  |  {var} {year}  |  cells used: {idx.size:,}")
    print("=" * 72)
    print(f"{'term':<28}{'recovered':>12}{'ground truth':>14}{'abs err':>10}")
    print("-" * 72)
    for (name, _, truth), est in zip(cols, coef_adj):
        # Flag a term only if it's off by more than 2% (or 0.01 absolute) -- a loose tolerance,
        # since estimates carry a little sampling noise.
        flag = "" if abs(est - truth) <= max(0.02 * abs(truth), 0.01) else "  <-- check"
        print(f"{name:<28}{est:>12.4f}{truth:>14.4f}{abs(est - truth):>10.4f}{flag}")
    print("-" * 72)
    print(f"ADJUSTED  beta(PM25) = {coef_adj[pm_j]:+.4f}   (truth {beta_true:+.4f})")
    print(f"NAIVE     beta(PM25) = {coef_naive[1]:+.4f}   <- biased by confounding "
          f"(off by {coef_naive[1] - beta_true:+.4f})")
    print("=" * 72)

    # ---- STEP 8: forward check -- an independent sanity test that needs no regression ----
    # Recompute the full rate for every cell, apply the same max(0.01, .) floor the generator
    # used, multiply by offset to get expected counts, and sum. The observed total should match
    # the expected total to within Poisson noise (ratio ~ 1.0).
    full_rate = p.base_rate + seasonal[:, None] + lat_eff[None, :] + lon_eff[None, :] + beta_true * pm25
    for c in confounders:
        std_c = _load_covar_aligned(cov_root, c.var_group, c.var, year, zctas, n_days,
                                    temporal_res=c.temporal_res, standardize=True)
        full_rate = full_rate + float(c.gamma) * std_c
    expected_total = float((np.maximum(0.01, full_rate) * offset[None, :])[:n_obs_days].sum())
    observed_total = float(Y[:n_obs_days].sum())
    print(f"forward check (first {n_obs_days} day(s)):  observed total = {observed_total:,.0f}   "
          f"expected E[sum] = {expected_total:,.0f}   ratio = {observed_total / expected_total:.4f}")
    # The floor slightly breaks exact linearity; report how often it bites. ~0% => closed form is tight.
    clamp_frac = float((full_rate < 0.01).mean())
    print(f"fraction of cells hitting the 0.01 rate clamp: {clamp_frac:.4%} "
          f"(should be ~0 for the closed form to hold exactly)\n")


if __name__ == "__main__":
    main()
