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


# Module-level caches so repeated reads within one process (e.g. the ground-truth ERC sweeps the
# exposure and re-evaluates the rate many times) don't reload the same .npy / parquet every call.
_RAW_COVAR_CACHE = {}   # (root, var_group, var, year, standardize) -> loaded (+standardized) array
_IDX2ZCTA_CACHE = {}    # root -> list of zcta strings in store-column order


def _load_raw_covar(root, var_group, var, year, standardize):
    """np.load (and optionally standardize) one covariate file, memoized for the process."""
    key = (root, var_group, var, int(year), bool(standardize))
    arr = _RAW_COVAR_CACHE.get(key)
    if arr is None:
        arr = np.load(f"{root}/{var_group}/{var}/{var}__{year}.npy")
        if standardize:
            m, s = np.nanmean(arr), np.nanstd(arr)
            arr = (arr - m) / s if s else arr - m
        _RAW_COVAR_CACHE[key] = arr
    return arr


def _load_covar_aligned(root, var_group, var, year, zctas, n_days, temporal_res, standardize=False):
    """Load a covariate from the dense store as ``(n_days, n_zctas)`` aligned to ``zctas`` order.

    Yearly vars are broadcast across days. ZCTAs absent from the store and NaNs contribute 0.
    With ``standardize`` the array is centered/scaled (nan-aware, over the whole file) so a
    coefficient on it is per-standard-deviation — handy when confounders live on very different
    scales (income vs temperature vs NO2).
    """
    arr = _load_raw_covar(root, var_group, var, year, standardize)
    cov_zcta = _IDX2ZCTA_CACHE.get(root)
    if cov_zcta is None:
        cov_zcta = pd.read_parquet(f"{root}/idx2zcta.parquet")["zcta"].astype(str).tolist()
        _IDX2ZCTA_CACHE[root] = cov_zcta
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


def build_extra_rate(cfg, zcta_data, exposure_override=None):
    """Build the additive causal rate term for one year, aligned to ``zcta_data`` row order.
    Reads ``cfg.synthetic`` (``beta``, ``exposure_*``, ``confounders``) and ``cfg.year``.
    Returns a ``(n_days, n_zctas)`` float32 array, or ``None`` when there are no causal terms
    (``beta=0`` and no confounders) — in which case the outcome keeps its original behaviour.

    ``exposure_override``: if given a scalar ``x``, the exposure term uses ``beta * x`` in EVERY
    cell instead of the real PM2.5 loaded from disk — i.e. the ``do(exposure = x)`` intervention.
    Confounders are always read from disk unchanged. This is what lets the ground-truth ERC be
    computed straight from the DGP (see ``expected_rate_grid``).
    """
    import calendar

    n_days = 366 if calendar.isleap(int(cfg.year)) else 365
    zctas = zcta_data["zcta"]
    cov_root = cfg.synthetic.get("exposure_covars_root", "data/covars")
    extra = np.zeros((n_days, len(zctas)), dtype=np.float32)

    beta = float(cfg.synthetic.get("beta", 0.0))
    if beta:
        if exposure_override is not None:
            extra += beta * float(exposure_override)      # do(exposure = x): same x in every cell
        else:
            exp_vg, exp_var = cfg.synthetic.exposure_var_group, cfg.synthetic.exposure_var
            LOGGER.info(f"Exposure term: {beta} * {exp_vg}/{exp_var} (raw)")
            extra += beta * _load_covar_aligned(
                cov_root, exp_vg, exp_var, cfg.year, zctas, n_days,
                temporal_res="daily", standardize=False,
            )

    confounders = cfg.synthetic.get("confounders", None) or []
    for c in confounders:
        if exposure_override is None:
            LOGGER.info(f"Confounder term: {c.gamma} * standardized({c.var_group}/{c.var}) [{c.temporal_res}]")
        extra += float(c.gamma) * _load_covar_aligned(
            cov_root, c.var_group, c.var, cfg.year, zctas, n_days,
            temporal_res=c.temporal_res, standardize=True,
        )

    return extra if (beta or confounders) else None


def offset_vector(cfg, zcta_data):
    """Per-zcta Poisson offset ``population_normalizer * population`` (shape ``(n_zctas,)``).

    The expected outcome COUNT in a cell is ``rate * offset``; dividing a count by the offset
    recovers the per-capita rate. Same definition the generator uses.
    """
    p = cfg.synthetic.poisson_params
    return p.population_normalizer * zcta_data["population"].to_numpy(dtype=np.float64)


def expected_rate_grid(cfg, zcta_data, exposure_override=None):
    """The DGP's per-cell Poisson RATE ``lambda`` as a ``(n_days, n_zctas)`` array.

        lambda = max(0.01, base + seasonal + lat + lon + beta*exposure + sum_k gamma_k*std(C_k))

    This is the SINGLE definition of the data-generating mean used for *evaluation*: the
    ground-truth ERC (``ground_truth_erc.py``) is just this averaged over cells while sweeping the
    exposure. It mirrors the rate built in ``synthetic_health.generate_synthetic_data`` — if you
    change the DGP's functional form, change it here too and every downstream check follows.

    ``exposure_override=x`` fixes the exposure to ``x`` in every cell (the ``do(exposure=x)``
    intervention used to trace the ERC); otherwise the real PM2.5 from the store is used.
    """
    import calendar

    p = cfg.synthetic.poisson_params
    year = int(cfg.year)
    n_days = 366 if calendar.isleap(year) else 365

    # base + seasonal + geography (exactly as in generate_synthetic_data)
    lat_eff = p.latitude_effect * np.sin(((zcta_data["latitude"].to_numpy() - 35) / 15) * np.pi)
    lon_eff = p.longitude_effect * np.cos(((zcta_data["longitude"].to_numpy() + 95) / 30) * np.pi)
    seasonal = p.seasonal_amplitude * np.sin(2 * np.pi * np.arange(n_days) / 365.25)
    rate = (p.base_rate + seasonal[:, None] + lat_eff[None, :] + lon_eff[None, :]).astype(np.float64)

    # + beta*exposure + sum_k gamma_k*std(confounder_k)  (with optional do(exposure=x))
    extra = build_extra_rate(cfg, zcta_data, exposure_override=exposure_override)
    if extra is not None:
        rate = rate + extra

    return np.maximum(0.01, rate)


def marginal_outcome(rate_grid, offset, forecast=slice(None), node_aggregation="mean"):
    """Collapse a ``(n_days, n_zctas)`` expected-RATE grid into ONE exposure-response point.

    This is the estimand the model's ERC also targets (see the g-computation spec): for each node
    SUM the expected count over the forecast-day window, then AGGREGATE over nodes::

        per_node F(z) = sum_{t in forecast}  rate[t, z] * offset[z]      # windowed count per node
        F            = mean_z F(z)            (node_aggregation='mean')   # average over the V nodes
                       sum_z  F(z)            (node_aggregation='sum')

    ``forecast`` is a slice (or index array) selecting the forecast days T_fc to sum over; the
    default sums over every day. Summing over time is the right collapse when the outcome is a
    windowed count Y_z = sum_t Y_{z,t}, since E[Y_z] = sum_t E[Y_{z,t}].
    """
    expected_counts = rate_grid * offset[None, :]           # E[Y_{t,z}] per (day, node)
    per_node = expected_counts[forecast, :].sum(axis=0)     # sum over the forecast window -> (n_zctas,)
    return float(per_node.mean() if node_aggregation == "mean" else per_node.sum())
