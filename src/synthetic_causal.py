"""Known causal rate terms for the semi-synthetic outcome.

Keeps the exposure/confounder machinery out of ``synthetic_health.py`` so that file reads
almost like the original generator. The outcome's Poisson rate gains an additive ``extra``
term built here:

    extra = beta * exposure(raw)  +  sum_k gamma_k * standardized(confounder_k)

All values are read from the dense covariate store (``data/covars``) so the model later trains
on exactly the numbers that generated the outcome. ``beta`` and the ``gamma_k`` are the ground
truth used to validate the ERC / g-computation pipeline. With ``beta=0`` and no confounders
``build_extra_rate`` returns ``None`` and the outcome reverts to the original exposure-free DGP.
"""

import logging

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)


def _load_covar_aligned(root, var_group, var, year, zctas, n_days, temporal_res, standardize=False):
    """Load a covariate from the dense store as ``(n_days, n_zctas)`` aligned to ``zctas`` order.

    Yearly vars are broadcast across days. ZCTAs absent from the store and NaNs contribute 0.
    With ``standardize`` the array is centered/scaled (nan-aware, over the whole file) so a
    coefficient on it is per-standard-deviation — handy when confounders live on very different
    scales (income vs temperature vs NO2).
    """
    arr = np.load(f"{root}/{var_group}/{var}/{var}__{year}.npy")
    if standardize:
        m, s = np.nanmean(arr), np.nanstd(arr)
        arr = (arr - m) / s if s else arr - m
    cov_zcta = pd.read_parquet(f"{root}/idx2zcta.parquet")["zcta"].astype(str).tolist()
    z2col = {z: i for i, z in enumerate(cov_zcta)}
    cols = np.array([z2col.get(str(z), -1) for z in zctas], dtype=np.int64)
    valid = cols >= 0
    out = np.zeros((n_days, len(cols)), dtype=np.float32)
    if temporal_res == "daily":
        if arr.shape[0] != n_days:
            raise ValueError(f"{var}: daily array has {arr.shape[0]} days, expected {n_days}")
        out[:, valid] = np.nan_to_num(arr[:, cols[valid]], nan=0.0)
    elif temporal_res == "yearly":
        vec = np.zeros(len(cols), dtype=np.float32)
        vec[valid] = np.nan_to_num(arr[cols[valid]], nan=0.0)
        out[:] = vec  # same value every day
    else:
        raise ValueError(f"temporal_res {temporal_res!r} not supported (daily/yearly)")
    n_missing = int((~valid).sum())
    if n_missing:
        LOGGER.warning(f"{var}: {n_missing}/{len(cols)} ZCTAs absent from store; set to 0 there")
    return out


def build_extra_rate(cfg, zcta_data):
    """Build the additive causal rate term for one year, aligned to ``zcta_data`` row order.

    Reads ``cfg.synthetic`` (``beta``, ``exposure_*``, ``confounders``) and ``cfg.year``.
    Returns a ``(n_days, n_zctas)`` float32 array, or ``None`` when there are no causal terms
    (``beta=0`` and no confounders) — in which case the outcome keeps its original behaviour.
    """
    import calendar

    n_days = 366 if calendar.isleap(int(cfg.year)) else 365
    zctas = zcta_data["zcta"]
    cov_root = cfg.synthetic.get("exposure_covars_root", "data/covars")
    extra = np.zeros((n_days, len(zctas)), dtype=np.float32)

    beta = float(cfg.synthetic.get("beta", 0.0))
    if beta:
        exp_vg, exp_var = cfg.synthetic.exposure_var_group, cfg.synthetic.exposure_var
        LOGGER.info(f"Exposure term: {beta} * {exp_vg}/{exp_var} (raw)")
        extra += beta * _load_covar_aligned(
            cov_root, exp_vg, exp_var, cfg.year, zctas, n_days,
            temporal_res="daily", standardize=False,
        )

    confounders = cfg.synthetic.get("confounders", None) or []
    for c in confounders:
        LOGGER.info(f"Confounder term: {c.gamma} * standardized({c.var_group}/{c.var}) [{c.temporal_res}]")
        extra += float(c.gamma) * _load_covar_aligned(
            cov_root, c.var_group, c.var, cfg.year, zctas, n_days,
            temporal_res=c.temporal_res, standardize=True,
        )

    return extra if (beta or confounders) else None
