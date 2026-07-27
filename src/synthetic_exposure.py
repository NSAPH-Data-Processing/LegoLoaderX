"""Generate synthetic environmental EXPOSURES with a KNOWN dependence on the confounders.

WHY
---
The semi-synthetic outcome already has a known dose-response (``beta``) and known confounder
coefficients (``gamma_k``). But the EXPOSURE itself was, until now, the *real* PM2.5 / gridmet read
from disk -- so the confounder->treatment arm of the confounding (the "propensity"/dose-assignment
mechanism) was whatever nature produced: uncontrolled and unknown. For a clean ERC / g-computation
validation we want that arm to be KNOWN and STRONG, so that a naive exposure-outcome fit is provably
biased and only a correct adjustment for the confounders recovers ``beta``.

This module GENERATES each environmental exposure as a treatment-assignment model

    T_e[t, i] ~ N( mu_e[t, i], sigma_e^2 ),   mu_e = f_e(X) + spatial structure + temporal structure

where ``X`` are the (real) confounders. ``f_e`` is a simple linear combination of standardized
confounders; the spatial + temporal structure reuses the SAR + AR(1) operators from
``src/synthetic_spacetime.py`` (the same machinery the outcome rate uses), so nearby ZCTAs and
adjacent days get correlated exposures -- realistic for an environmental field.

VARIANCE-FRACTION PARAMETERIZATION (so "signal strength" is a direct dial)
-------------------------------------------------------------------------
Rather than tuning raw weights, each exposure mixes three UNIT-VARIANCE, mutually-independent parts:

    z_drivers = zscore( sum_k w_{e,k} * standardized(confounder_k) )     # f_e(X): the confounder signal
    z_struct  = zscore( spacetime_couple( white_noise ; rho_e, phi_e ) ) # spatial + temporal structure
    z_noise   = zscore( iid N(0, 1) )                                     # independent measurement noise

    x_std     = sqrt(f_drivers)*z_drivers + sqrt(f_struct)*z_struct + sqrt(1 - f_drivers - f_struct)*z_noise
    exposure  = target_mean + target_sd * x_std          # PM2.5 is clipped >= 0 (a concentration)

Because ``z_struct`` and ``z_noise`` are independent of the confounders, an OLS of the exposure on the
confounders recovers ``R^2 ~ f_drivers`` BY CONSTRUCTION. Keep ``f_noise = 1 - f_drivers - f_struct``
strictly positive so the exposure still varies CONDITIONAL on the confounders (positivity/overlap) --
otherwise ``do(exposure = a)`` would extrapolate off-support and no estimator could recover the curve.

WHY THE GROUND TRUTH IS UNAFFECTED
----------------------------------
The ground-truth ERC (``src/ground_truth_erc.py``) is a ``do(exposure = a)`` intervention: it OVERRIDES
the exposure term and reads every confounder from disk. It does not care HOW the exposure was assigned,
only that the outcome rate is ``beta*shape(exposure) + sum_k gamma_k*...``. So making the exposure
synthetic changes only what is on disk, never the closed-form estimand. (The 4 gridmet exposures enter
the outcome STANDARDIZED, so their absolute scale is irrelevant; only PM2.5 -- read unstandardized as
``beta*PM25`` -- needs a realistic scale, hence ``target_mean``/``target_sd``.)

Output: dense covar ``.npy`` per (var, year) at ``<covars_root>/<var_group>/<var>/<var>__<year>.npy``,
float32, shape ``(n_days, n_zctas)``, in ``<covars_root>/idx2zcta.parquet`` column order (NaN for ZCTAs
absent from the generator's mainland grid) -- i.e. the exact format the dataloader + the outcome DGP
read. A small ``<var>__<year>.meta.json`` records the generation parameters and the achieved R^2.

Run:
    PYTHONPATH=. python -m src.synthetic_exposure year=2011
"""

import calendar
import json
import logging
import os

import hydra
import numpy as np
import pandas as pd

from src.synthetic_causal import _load_covar_aligned
from src.synthetic_denom import get_zcta_data_with_geo_pop
from src.synthetic_spacetime import apply_spacetime_coupling, get_W

LOGGER = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


def _zscore(a):
    """Standardize to mean 0, sd 1 over the finite cells (nan-safe). A constant field maps to 0."""
    a = np.asarray(a, dtype=np.float64)
    m = np.nanmean(a)
    s = np.nanstd(a)
    return (a - m) / s if s > 0 else a - m


def _driver_matrix(cfg, zcta_data, year, n_days, cov_root, drivers):
    """Standardized confounder arrays, one per driver, each ``(n_days, n_zctas)`` on ``zcta_data`` order.

    Reuses ``_load_covar_aligned(standardize=True)`` -- the SAME reader/standardization the outcome DGP
    uses for its confounder terms -- so ``f_e(X)`` is built from exactly the values the outcome sees.
    """
    mats = {}
    for d in drivers:
        mats[d["var"]] = _load_covar_aligned(
            cov_root, d["var_group"], d["var"], year,
            zcta_data["zcta"], n_days, temporal_res=d.get("temporal_res", "yearly"),
            standardize=True,
        )
    return mats


def _r2_on_drivers(exposure, driver_mats, sample=400_000, seed=0):
    """Multiple R^2 of ``exposure ~ [all drivers]`` via OLS on a random subsample of cells.

    This is the honest "fraction of exposure variance explained by the confounders" -- the strength of
    the confounder->treatment relationship. By construction it should land near ``f_drivers``.
    """
    y = np.asarray(exposure, dtype=np.float64).ravel()
    X = np.column_stack([m.ravel() for m in driver_mats.values()])
    finite = np.isfinite(y) & np.isfinite(X).all(axis=1)
    y, X = y[finite], X[finite]
    if y.size == 0:
        return float("nan")
    if y.size > sample:                                   # subsample for speed; deterministic
        idx = np.random.default_rng(seed).choice(y.size, size=sample, replace=False)
        y, X = y[idx], X[idx]
    X = np.column_stack([np.ones(len(y)), X])             # intercept
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    ss_res = float(resid @ resid)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def build_one_exposure(cfg, zcta_data, year, spec, driver_mats, base_seed, e_idx):
    """Build one synthetic exposure as ``(n_days, n_zctas)`` on ``zcta_data`` order.

    ``spec`` is one entry of ``synthetic.exposure_generation.exposures``. ``driver_mats`` is the shared
    per-year dict of standardized confounders (built once, reused across exposures).
    """
    n_days = 366 if calendar.isleap(int(year)) else 365
    n_z = len(zcta_data)
    eg = cfg.synthetic.exposure_generation
    f_drivers = float(spec.get("f_drivers", 0.6))
    f_struct = float(spec.get("f_struct", 0.2))
    f_noise = 1.0 - f_drivers - f_struct
    if f_noise < 1e-9:
        raise ValueError(
            f"exposure {spec['name']}: f_drivers + f_struct = {f_drivers + f_struct} leaves no "
            f"independent noise; keep it < 1 so the exposure varies conditional on the confounders."
        )

    # independent RNG per (exposure, year) so the 5 treatments are not mutually collinear
    rng = np.random.default_rng(int(base_seed) + 1000 * int(e_idx) + int(year))

    # --- f_e(X): weighted sum of standardized confounders, then re-standardized to unit variance ---
    weights = dict(spec.get("weights", {}) or {})
    signal = np.zeros((n_days, n_z), dtype=np.float64)
    for name, w in weights.items():
        if name not in driver_mats:
            raise ValueError(f"exposure {spec['name']} weight on {name!r} which is not in drivers")
        signal = signal + float(w) * driver_mats[name]
    z_drivers = _zscore(signal)

    # --- spatial + temporal structure: couple white noise, then re-standardize to unit variance ---
    if f_struct > 0:
        W = get_W(zcta_data, method=str(eg.get("method", "knn")), k=int(eg.get("k", 8)),
                  length_scale_km=eg.get("length_scale_km", None)) if float(spec.get("rho", 0.0)) else None
        wn = rng.standard_normal((n_days, n_z))
        coupled = apply_spacetime_coupling(
            wn, W, rho=float(spec.get("rho", 0.0)), phi=float(spec.get("phi", 0.0)), normalize=True
        )
        z_struct = _zscore(coupled)
    else:
        z_struct = np.zeros((n_days, n_z), dtype=np.float64)

    # --- independent measurement noise ---
    z_noise = _zscore(rng.standard_normal((n_days, n_z)))

    x_std = (np.sqrt(f_drivers) * z_drivers
             + np.sqrt(f_struct) * z_struct
             + np.sqrt(f_noise) * z_noise)
    exposure = float(spec.get("target_mean", 0.0)) + float(spec.get("target_sd", 1.0)) * x_std
    if bool(spec.get("nonnegative", False)):
        exposure = np.clip(exposure, 0.0, None)
    return exposure.astype(np.float32)


def _save_as_covar(exposure, zcta_data, covars_root, var_group, var, year):
    """Map ``(n_days, n_mainland)`` in ``zcta_data`` order -> covars ``idx2zcta`` column order and save.

    NaN for ZCTAs absent from the generator's mainland grid -- the same missing convention as every
    other covar file, so ``_load_covar_aligned`` round-trips it back to the exact generated values.
    """
    covars_zcta = pd.read_parquet(f"{covars_root}/idx2zcta.parquet")["zcta"].astype(str).tolist()
    z2col = {z: i for i, z in enumerate(covars_zcta)}
    gen_zctas = zcta_data["zcta"].astype(str).to_numpy()
    cols = np.array([z2col.get(z, -1) for z in gen_zctas], dtype=np.int64)
    valid = cols >= 0
    n_days = exposure.shape[0]
    out = np.full((n_days, len(covars_zcta)), np.nan, dtype=np.float32)
    out[:, cols[valid]] = exposure[:, np.where(valid)[0]]

    out_dir = f"{covars_root}/{var_group}/{var}"
    os.makedirs(out_dir, exist_ok=True)
    fpath = f"{out_dir}/{var}__{year}.npy"
    np.save(fpath, out)
    return fpath, int((~valid).sum())


@hydra.main(config_path="../conf/synthetic", config_name="config", version_base=None)
def main(cfg):
    """Generate all configured synthetic exposures for ``cfg.year`` and write them to the covar store."""
    eg = cfg.synthetic.get("exposure_generation", None)
    if not eg or not eg.get("enabled", False):
        LOGGER.info("synthetic.exposure_generation.enabled is false -> nothing to do.")
        return

    year = int(cfg.year)
    n_days = 366 if calendar.isleap(year) else 365
    covars_root = cfg.synthetic.get("exposure_covars_root", "data/covars")
    base_seed = int(eg.get("base_seed", 0))
    drivers = list(eg.get("drivers", []) or [])
    exposures = list(eg.get("exposures", []) or [])
    LOGGER.info(f"Generating {len(exposures)} synthetic exposures for {year} "
                f"(drivers={len(drivers)}, covars_root={covars_root})")

    zcta_data = get_zcta_data_with_geo_pop(
        unique_fpath=cfg.synthetic.zcta_unique_path,
        shapefile_fpath=cfg.synthetic.zcta_shapefile_path,
        population_fpath=cfg.synthetic.population_path,
        year=year, mainland_only=cfg.synthetic.mainland_only,
    )
    LOGGER.info(f"Generator grid: {len(zcta_data)} mainland ZCTAs")

    driver_mats = _driver_matrix(cfg, zcta_data, year, n_days, covars_root, drivers)

    from omegaconf import OmegaConf
    for e_idx, spec in enumerate(exposures):
        spec = OmegaConf.to_container(spec, resolve=True) if not isinstance(spec, dict) else spec
        name = spec["name"]
        exposure = build_one_exposure(cfg, zcta_data, year, spec, driver_mats, base_seed, e_idx)
        r2 = _r2_on_drivers(exposure, driver_mats)
        fpath, n_missing = _save_as_covar(
            exposure, zcta_data, covars_root,
            spec.get("var_group", "synth_exposure"), name, year,
        )
        finite = np.isfinite(exposure)
        LOGGER.info(
            f"  {name}: R^2(exposure~confounders)={r2:.3f} (target f_drivers={spec.get('f_drivers')}) | "
            f"mean={float(np.nanmean(exposure)):.3f} sd={float(np.nanstd(exposure)):.3f} "
            f"min={float(np.nanmin(exposure)):.3f} | {n_missing} mainland ZCTAs not in covars grid "
            f"-> {fpath}"
        )
        meta = {
            "var": name, "year": year, "var_group": spec.get("var_group", "synth_exposure"),
            "model": "T ~ N(mu, sigma^2); mu = f(confounders) + spatial(SAR) + temporal(AR1)",
            "f_drivers": spec.get("f_drivers"), "f_struct": spec.get("f_struct"),
            "rho": spec.get("rho"), "phi": spec.get("phi"),
            "target_mean": spec.get("target_mean"), "target_sd": spec.get("target_sd"),
            "nonnegative": spec.get("nonnegative", False),
            "weights": spec.get("weights", {}),
            "drivers": [d["var"] for d in drivers],
            "seed": base_seed + 1000 * e_idx + year,
            "achieved_r2_on_confounders": r2,
        }
        with open(f"{os.path.dirname(fpath)}/{name}__{year}.meta.json", "w") as f:
            json.dump(meta, f, indent=2)

    LOGGER.info(f"Synthetic exposure generation complete for {year}")


if __name__ == "__main__":
    main()
