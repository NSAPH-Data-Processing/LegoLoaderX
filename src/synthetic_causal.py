"""Known causal rate terms for the semi-synthetic outcome.

Keeps the exposure/confounder machinery out of ``synthetic_health.py`` so that file reads
almost like the original generator. The outcome's Poisson rate gains an additive ``extra``
term built here:

    extra = beta * shape(exposure)
            + sum_k gamma_k * shape_k(standardized(confounder_k))
            + sum_j delta_j * prod(interacting terms_j)      # only when interactions enabled

All values are read from the dense covariate store (``data/covars``) so the model later trains
on exactly the numbers that generated the outcome. ``beta``, the ``gamma_k`` and the interaction
``delta_j`` are the ground truth used to validate the ERC / g-computation pipeline. With
``beta=0``, no confounders and no interactions, ``build_extra_rate`` returns ``None`` and the
outcome reverts to the original exposure-free DGP.

NON-LINEAR SHAPES (config-driven, evaluated here so the ground truth follows automatically)
-------------------------------------------------------------------------------------------
Each term may be passed through a fixed-form ``shape`` (see ``_apply_shape``):
  * exposure ``synthetic.exposure_shape`` -- e.g. ``sqrt`` gives a concave PM2.5 dose-response,
    matching the sub-linear exposure-response curves reported in the air-pollution literature.
  * confounder ``shape`` (+ optional ``center``) -- e.g. ``u_shape`` makes temperature U-shaped
    (risk rises at both cold and hot extremes, minimum near ``center`` in standardized units).
The default shape is ``linear`` everywhere, which reproduces the original additive-linear DGP
byte-for-byte.

INTERACTIONS (``synthetic.interactions: true``)
-----------------------------------------------
When enabled, each entry of ``synthetic.interaction_terms`` adds ``delta * prod(refs)`` where a
ref is ``"exposure"`` (the shaped exposure) or a confounder's ``var`` (its standardized value).
This injects effect modification -- confounder x confounder (e.g. income x age) and
confounder x treatment (e.g. PM2.5 x %over-65) -- into the known DGP.
"""

import logging

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)


# Module-level caches so repeated reads within one process (e.g. the ground-truth ERC sweeps the
# exposure and re-evaluates the rate many times) don't reload the same .npy / parquet every call.
_RAW_COVAR_CACHE = {}   # (root, var_group, var, year, standardize) -> loaded (+standardized) array
_IDX2ZCTA_CACHE = {}    # root -> list of zcta strings in store-column order


def _apply_shape(x, shape=None, center=0.0):
    """Pass a value through a fixed-form non-linearity. Works on scalars and arrays alike.

    ``x`` is the raw exposure (for the exposure term) or the standardized confounder (for a
    confounder term). Supported ``shape`` values:

      * ``linear`` / ``identity`` (default): ``x``  -- the original additive-linear behaviour.
      * ``sqrt``:   ``sqrt(max(x, 0))``     -- concave, saturating (sub-linear dose-response).
      * ``log1p``:  ``log(1 + max(x, 0))``  -- concave, even more saturating.
      * ``quadratic`` / ``u_shape``: ``(x - center)**2`` -- U-shaped about ``center`` (use a
        POSITIVE coefficient so risk rises on both sides, e.g. cold AND hot temperature).
      * ``abs`` / ``v_shape``: ``|x - center|`` -- V-shaped about ``center``.

    ``center`` shifts the minimum of the U/V shapes. For a confounder it is in *standardized*
    units (0 = the variable's mean); for the raw exposure it is in the exposure's own units.
    """
    shape = (shape or "linear").lower()
    if shape in ("linear", "identity", "none"):
        return x
    if shape == "sqrt":
        return np.sqrt(np.clip(x, 0.0, None))
    if shape == "log1p":
        return np.log1p(np.clip(x, 0.0, None))
    if shape in ("quadratic", "square", "u_shape", "u-shape", "ushape"):
        return (x - center) ** 2
    if shape in ("abs", "v_shape", "v-shape"):
        return np.abs(x - center)
    raise ValueError(
        f"unknown shape {shape!r}; supported: linear, sqrt, log1p, quadratic/u_shape, abs"
    )


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


def _shaped_exposure(cfg, zctas, n_days, cov_root, exposure_override):
    """Return ``shape(exposure)`` — a scalar under ``do(exposure=x)``, else a ``(n_days, n_zctas)``
    array of the real PM2.5 from disk. The shape (``synthetic.exposure_shape``) is applied here so
    both the generator and the ground-truth ERC see the identical (possibly non-linear) form."""
    shape = cfg.synthetic.get("exposure_shape", "linear")
    if exposure_override is not None:
        return _apply_shape(float(exposure_override), shape)          # do(exposure=x): same x everywhere
    exp_vg, exp_var = cfg.synthetic.exposure_var_group, cfg.synthetic.exposure_var
    raw = _load_covar_aligned(cov_root, exp_vg, exp_var, cfg.year, zctas, n_days,
                              temporal_res="daily", standardize=False)
    return _apply_shape(raw, shape)


def _confounder_value(cfg, c, zctas, n_days, cov_root, shape):
    """Standardized confounder ``c`` aligned to ``zctas``, optionally passed through ``shape``.

    ``shape`` is decoupled from the entry's main-effect shape so interaction terms can reuse the
    plain standardized value while the main effect uses, say, a U-shape.
    """
    std = _load_covar_aligned(cov_root, c.var_group, c.var, cfg.year, zctas, n_days,
                              temporal_res=c.temporal_res, standardize=True)
    return _apply_shape(std, shape, float(c.get("center", 0.0)))


def build_extra_rate(cfg, zcta_data, exposure_override=None):
    """Build the additive causal rate term for one year, aligned to ``zcta_data`` row order.
    Reads ``cfg.synthetic`` (``beta``, ``exposure_*``, ``exposure_shape``, ``confounders``,
    ``interactions``/``interaction_terms``) and ``cfg.year``. Returns a ``(n_days, n_zctas)``
    float32 array, or ``None`` when there are no causal terms (``beta=0``, no confounders, no
    interactions) — in which case the outcome keeps its original behaviour.

    ``exposure_override``: if given a scalar ``x``, the exposure term uses ``beta * shape(x)`` in
    EVERY cell instead of the real PM2.5 loaded from disk — i.e. the ``do(exposure = x)``
    intervention. Confounders are always read from disk unchanged. This is what lets the
    ground-truth ERC be computed straight from the DGP (see ``expected_rate_grid``).
    """
    import calendar

    n_days = 366 if calendar.isleap(int(cfg.year)) else 365
    zctas = zcta_data["zcta"]
    cov_root = cfg.synthetic.get("exposure_covars_root", "data/covars")
    quiet = exposure_override is not None        # ERC sweeps call this many times; log only the real build
    extra = np.zeros((n_days, len(zctas)), dtype=np.float32)

    # --- exposure term: beta * shape(exposure) ---
    beta = float(cfg.synthetic.get("beta", 0.0))
    shaped_exposure = None
    if beta:
        shaped_exposure = _shaped_exposure(cfg, zctas, n_days, cov_root, exposure_override)
        if not quiet:
            exp_vg, exp_var = cfg.synthetic.exposure_var_group, cfg.synthetic.exposure_var
            LOGGER.info(f"Exposure term: {beta} * {cfg.synthetic.get('exposure_shape', 'linear')}({exp_vg}/{exp_var})")
        extra += beta * shaped_exposure

    # --- confounder terms: sum_k gamma_k * shape_k(standardized(C_k)) ---
    confounders = cfg.synthetic.get("confounders", None) or []
    for c in confounders:
        shape = c.get("shape", "linear")
        if not quiet:
            LOGGER.info(f"Confounder term: {c.gamma} * {shape}(standardized({c.var_group}/{c.var})) [{c.temporal_res}]")
        extra += float(c.gamma) * _confounder_value(cfg, c, zctas, n_days, cov_root, shape)

    # --- interaction terms (effect modification): sum_j delta_j * prod(refs_j) ---
    n_interactions = 0
    if cfg.synthetic.get("interactions", False):
        conf_by_var = {c.var: c for c in confounders}
        for term in (cfg.synthetic.get("interaction_terms", None) or []):
            refs = list(term.get("vars", []))
            prod = float(term.gamma)
            for name in refs:
                if name == "exposure":
                    if shaped_exposure is None:   # interaction needs the exposure but beta==0
                        shaped_exposure = _shaped_exposure(cfg, zctas, n_days, cov_root, exposure_override)
                    prod = prod * shaped_exposure
                else:
                    c = conf_by_var.get(name)
                    if c is None:
                        raise ValueError(
                            f"interaction term references {name!r}, which is neither 'exposure' nor a "
                            f"confounder var; known confounders: {sorted(conf_by_var)}"
                        )
                    prod = prod * _confounder_value(cfg, c, zctas, n_days, cov_root, shape="linear")
            if not quiet:
                LOGGER.info(f"Interaction term: {term.gamma} * prod({refs})")
            extra += prod
            n_interactions += 1

    return extra if (beta or confounders or n_interactions) else None


def offset_vector(cfg, zcta_data):
    """Per-zcta Poisson offset ``population_normalizer * population`` (shape ``(n_zctas,)``).

    The expected outcome COUNT in a cell is ``rate * offset``; dividing a count by the offset
    recovers the per-capita rate. Same definition the generator uses.
    """
    p = cfg.synthetic.poisson_params
    return p.population_normalizer * zcta_data["population"].to_numpy(dtype=np.float64)


def expected_rate_grid(cfg, zcta_data, exposure_override=None):
    """The DGP's per-cell Poisson RATE ``lambda`` as a ``(n_days, n_zctas)`` array.

        lambda = max(rate_floor, base + seasonal + lat + lon
                           + beta*shape(exposure) + sum_k gamma_k*shape_k(std(C_k)) + interactions)

    ``rate_floor`` (``cfg.synthetic.poisson_params.rate_floor``, default 0.01) is the lower clamp
    that keeps the per-capita rate positive; it is applied ONCE, after the spacetime coupling.

    This is the SINGLE definition of the data-generating mean used for *evaluation*: the
    ground-truth ERC (``ground_truth_erc.py``) is just this averaged over cells while sweeping the
    exposure. It mirrors the rate built in ``synthetic_health.generate_synthetic_data`` — if you
    change the DGP's functional form, change it here too and every downstream check follows.

    ``exposure_override=x`` fixes the exposure to ``x`` in every cell (the ``do(exposure=x)``
    intervention used to trace the ERC); otherwise the real PM2.5 from the store is used.

    SPATIAL + TEMPORAL COUPLING (``cfg.synthetic.spacetime``)
    --------------------------------------------------------
    After the additive rate is built, an optional spatial (SAR) + temporal (AR(1)/EWMA) coupling is
    applied to the PRE-floor rate (see ``src/synthetic_spacetime.py``): each ZCTA's rate is blended
    with its neighbours' rates and its own recent past, with a geometrically decaying weight. The
    coupling is linear and deterministic, so this function still returns the EXACT ground-truth rate
    under ``do(exposure=x)`` -- no Monte Carlo. With ``normalize=True`` (default) the operators are
    mean-preserving, so the marginal exposure slope is unchanged and the rate stays as low as the
    uncoupled DGP; with ``normalize=False`` the slope is amplified by ``1/((1-rho)(1-phi))``. The
    block is a no-op (byte-identical output) when absent or when ``rho == phi == 0``.
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

    # spatial (neighbours) + temporal (own past) coupling of the PRE-floor rate; no-op when rho=phi=0.
    st = cfg.synthetic.get("spacetime", None) or {}
    rho = float(st.get("rho", 0.0) or 0.0)
    phi = float(st.get("phi", 0.0) or 0.0)
    if rho or phi:
        from src.synthetic_spacetime import apply_spacetime_coupling, get_W
        W = get_W(zcta_data, method=st.get("method", "knn"), k=int(st.get("k", 8)),
                  length_scale_km=st.get("length_scale_km", None)) if rho else None
        rate = apply_spacetime_coupling(rate, W, rho=rho, phi=phi,
                                        normalize=bool(st.get("normalize", True)))

    floor = float(p.get("rate_floor", 0.01))   # configurable lower clamp (default 0.01)
    return np.maximum(floor, rate)


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


def _ascii_hist(x, bins=20, width=48):
    """A compact ASCII histogram of ``x`` (clipped at p99.5 so a few outliers don't flatten it)."""
    x = np.asarray(x, dtype=np.float64).ravel()
    x = x[np.isfinite(x)]
    if x.size == 0:
        return "  histogram: (no finite values)"
    lo = float(x.min())
    hi = float(np.percentile(x, 99.5))
    if not (hi > lo):
        hi = lo + 1.0
    counts, edges = np.histogram(np.clip(x, lo, hi), bins=bins, range=(lo, hi))
    top = int(counts.max()) or 1
    out = ["  histogram (clipped at p99.5):"]
    for c, e0, e1 in zip(counts, edges[:-1], edges[1:]):
        out.append(f"    [{e0:10.4g},{e1:10.4g})  {'#' * int(round(width * c / top))} {int(c)}")
    return "\n".join(out)


def describe_rate_grid(rate_grid, offset=None, floor=None, label="lambda", logger=None):
    """Log (and return) the distribution of the per-cell Poisson RATE ``lambda`` — a diagnostic only.

    Does NOT touch the data or the DGP; it just summarizes an already-built ``(n_days, n_zctas)``
    rate grid so you can see whether ``lambda`` stays low before a full run. Reports:

      * percentiles of ``lambda`` over all ``(day, zcta)`` cells (min/p1/p5/p25/median/p75/p95/p99/max),
        plus mean and std, and a compact ASCII histogram;
      * ``frac_at_floor`` — the fraction of cells pinned at ``floor`` (when given); a large value means
        the rate is being clamped a lot, i.e. the DGP wants to go lower than ``rate_floor`` allows;
      * when ``offset`` is given: the distribution of the expected count ``lambda*offset`` and the
        implied sparsity ``E[frac zero cells] = mean(exp(-lambda*offset))`` — comparable to the
        sparsity the generator logs after sampling.

    Returns a plain dict of the summary (handy for asserting in tests). Wired into the generator via
    ``synthetic.diagnostics.lambda_distribution`` and used by the smoke run ``src/synthetic_smoke.py``.
    """
    log = logger or LOGGER
    lam = np.asarray(rate_grid, dtype=np.float64)
    flat = lam.ravel()
    pcts = [0, 1, 5, 25, 50, 75, 95, 99, 100]
    q = np.percentile(flat, pcts)
    summary = {
        "n_cells": int(flat.size),
        "mean": float(flat.mean()),
        "std": float(flat.std()),
        "percentiles": {p: float(v) for p, v in zip(pcts, q)},
    }
    lines = [
        f"{label}: distribution over {flat.size:,} (day x zcta) cells",
        f"  mean={flat.mean():.4g} std={flat.std():.4g} | min={q[0]:.4g} p1={q[1]:.4g} "
        f"p5={q[2]:.4g} p25={q[3]:.4g} median={q[4]:.4g} p75={q[5]:.4g} p95={q[6]:.4g} "
        f"p99={q[7]:.4g} max={q[8]:.4g}",
    ]
    if floor is not None:
        frac = float(np.mean(flat <= float(floor) + 1e-12))
        summary["floor"] = float(floor)
        summary["frac_at_floor"] = frac
        lines.append(f"  at floor (lambda <= {float(floor):g}): {100 * frac:.2f}% of cells")
    if offset is not None:
        ec = (lam * np.asarray(offset, dtype=np.float64)[None, :]).ravel()
        eq = np.percentile(ec, [50, 95, 99, 100])
        exp_zero = float(np.mean(np.exp(-ec)))
        summary["expected_count"] = {"median": float(eq[0]), "p95": float(eq[1]),
                                     "p99": float(eq[2]), "max": float(eq[3])}
        summary["implied_sparsity_pct"] = 100.0 * exp_zero
        lines.append(f"  expected count lambda*offset: median={eq[0]:.3g} p95={eq[1]:.3g} "
                     f"p99={eq[2]:.3g} max={eq[3]:.3g}")
        lines.append(f"  implied sparsity E[frac zero] = mean(exp(-lambda*offset)): "
                     f"{100 * exp_zero:.2f}%")
    lines.append(_ascii_hist(flat))
    log.info("\n".join(lines))
    return summary


def plot_rate_distribution(rate_grid, path, floor=None, offset=None, label="lambda", bins=60):
    """Save a histogram PNG of the per-cell rate ``lambda`` (and, if ``offset`` is given, the expected
    counts) to ``path``. Diagnostic only — writes an image, nothing else. Returns the absolute path,
    or ``None`` if matplotlib is unavailable. Histograms are clipped at p99.5 so a few outliers don't
    flatten the bulk; the mean, median and (if given) the ``floor`` are drawn as reference lines.
    """
    import os
    try:
        import matplotlib
        matplotlib.use("Agg")           # headless: no display needed on a compute node
        import matplotlib.pyplot as plt
    except Exception as e:              # pragma: no cover - plotting is optional
        LOGGER.warning(f"lambda plot skipped: matplotlib unavailable ({e})")
        return None

    lam = np.asarray(rate_grid, dtype=np.float64).ravel()
    lam = lam[np.isfinite(lam)]
    ncols = 2 if offset is not None else 1
    fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 4), squeeze=False)

    ax = axes[0][0]
    hi = float(np.percentile(lam, 99.5))
    ax.hist(np.clip(lam, lam.min(), max(hi, lam.min() + 1e-12)), bins=bins, color="#4477aa")
    ax.axvline(float(lam.mean()), color="tab:orange", lw=1.2, label=f"mean={lam.mean():.4g}")
    ax.axvline(float(np.median(lam)), color="k", ls="--", lw=1, label=f"median={np.median(lam):.4g}")
    if floor is not None:
        frac = float(np.mean(lam <= float(floor) + 1e-12))
        ax.axvline(float(floor), color="tab:red", ls=":", lw=1.5,
                   label=f"floor={float(floor):g} ({100 * frac:.1f}% at floor)")
    ax.set_xlabel("lambda (per-cell rate)"); ax.set_ylabel("cells")
    ax.set_title(f"{label}  (n={lam.size:,} cells)"); ax.legend(fontsize=8)

    if offset is not None:
        ec = (np.asarray(rate_grid, dtype=np.float64) * np.asarray(offset, dtype=np.float64)[None, :]).ravel()
        ax2 = axes[0][1]
        hi2 = float(np.percentile(ec, 99.5))
        ax2.hist(np.clip(ec, ec.min(), max(hi2, ec.min() + 1e-12)), bins=bins, color="#66aa55")
        ax2.set_xlabel("expected count  lambda*offset"); ax2.set_ylabel("cells")
        ax2.set_title(f"expected counts (implied sparsity {100 * np.mean(np.exp(-ec)):.1f}%)")

    fig.tight_layout()
    path = os.path.abspath(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    LOGGER.info(f"saved lambda distribution plot -> {path}")
    return path
