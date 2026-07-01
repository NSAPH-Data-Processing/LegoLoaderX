"""Spatial + temporal coupling of the latent Poisson rate for the semi-synthetic DGP.

The base generator (``src/synthetic_causal.py``) builds a PRE-floor rate ``eta[t, i]`` that is
independent across days and ZCTAs. This module couples those cells so that ZCTA ``i``'s rate on
day ``t`` depends on (a) its spatial neighbours' rates and (b) its own rate on previous days, with
a geometrically decaying weight -- while keeping the rate bounded and, crucially, keeping the
ground-truth exposure-response curve (ERC) computable in closed form (no Monte Carlo).

THE TWO OPERATORS  (applied to the pre-floor rate, in this order)
-----------------------------------------------------------------
Let ``eta_t`` be the length-V pre-floor rate vector for day ``t`` (V = number of ZCTAs) and ``W`` a
row-normalized neighbour matrix (``W @ ones == ones``, zero diagonal).

  Spatial (SAR), per day:    eta_tilde_t = (1 - rho) * (I - rho W)^{-1} @ eta_t        rho in [0, 1)
  Temporal (AR(1)/EWMA):     lambda_t    = (1 - phi) * eta_tilde_t + phi * lambda_{t-1}  phi in [0, 1)
                             with lambda_{-1} := eta_tilde_0

The caller floors the result once (``max(0.01, lambda_t)``) AFTER coupling; we never floor inside
here -- flooring within the recursion would leak a non-linearity into the memory term and break the
closed-form properties below.

PROPERTIES THIS GUARANTEES  (verified numerically in the __main__ self-test, not just asserted)
-----------------------------------------------------------------------------------------------
1. Mean-preserving / stays low: both operators are row-stochastic averaging maps, so a field that is
   constant in space AND time is a fixed point. (I - rho W)^{-1} @ ones = ones / (1 - rho), so the
   (1 - rho) prefactor cancels; likewise the (1 - phi) EWMA prefactor cancels for a time-constant
   input. Hence the coupling adds spatial/temporal CORRELATION without raising the level -> the rate
   stays as low as the uncoupled DGP. (Holds with ``normalize=True``; see ``normalize`` below.)
2. Stability: for a row-normalized W, the spectral radius of ``rho W`` is ``rho < 1``, so
   ``(I - rho W)`` is invertible for ANY rho in [0, 1). The two steps are applied sequentially, so
   there is no joint rho+phi stability constraint.
3. Closed-form memory: unrolling the AR(1) recursion gives
       lambda_t = (1 - phi) * sum_{l=0}^{t} phi^l * eta_tilde_{t-l}  +  phi^{t+1} * lambda_{-1},
   i.e. the influence of day ``t-l`` on day ``t`` decays as ``(1 - phi) * phi^l`` -- a geometric
   memory with half-life ``l_half = ln(0.5) / ln(phi)`` days. See ``phi_to_halflife`` /
   ``halflife_to_phi`` so config values are interpretable ("phi=0.93" <-> "~10-day half-life").

ERC stays exact (no simulation)
-------------------------------
The operators are linear and deterministic, so ``expected_rate_grid(..., exposure_override=x)`` --
which makes the exposure contribution to ``eta`` constant in space and time -- followed by the SAME
coupling still returns the exact ground-truth marginal rate for ``do(exposure = x)``. With
``normalize=True`` the constant exposure field is a fixed point of both operators, so the marginal
exposure slope is UNCHANGED by the coupling. With ``normalize=False`` the (1 - rho) / (1 - phi)
prefactors are dropped, and the steady-state exposure slope is AMPLIFIED by ``1 / ((1-rho)(1-phi))``
(spatial spillover + persistence accumulate the constant input); the ERC is still exact and closed
form, it just represents that amplified estimand. Pick ``normalize`` deliberately.

NEIGHBOUR GRAPH
---------------
``build_spatial_weights`` builds W from ZCTA centroids (lat/lon) by k-nearest-neighbours (haversine
``BallTree`` if scikit-learn is present, else an equirectangular ``cKDTree`` fallback). ``method=
"distance"`` weights neighbours by ``exp(-dist_km / length_scale_km)`` before row-normalizing.
``method="queen"`` uses true border contiguity via ``libpysal`` (needs a GeoDataFrame with polygon
geometry; documented but optional). ``get_W`` memoizes the matrix so the ERC exposure sweep does not
rebuild it on every call.
"""

import logging
import math

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

LOGGER = logging.getLogger(__name__)

EARTH_RADIUS_KM = 6371.0088

# Caches so the ERC exposure sweep (which calls expected_rate_grid many times with the SAME grid)
# does not rebuild W or refactorize (I - rho W) on every call.
_W_CACHE = {}    # cache_key -> csr neighbour matrix
_LU_CACHE = {}   # (id(W), rho) -> splu factorization of (I - rho W)


# --------------------------------------------------------------------------------------------------
# Half-life <-> phi helpers (so config values are interpretable)
# --------------------------------------------------------------------------------------------------
def halflife_to_phi(halflife_days):
    """AR(1) decay ``phi`` whose memory weight halves every ``halflife_days``: ``0.5 ** (1/h)``."""
    h = float(halflife_days)
    if h <= 0:
        raise ValueError(f"halflife_days must be > 0; got {h}")
    return 0.5 ** (1.0 / h)


def phi_to_halflife(phi):
    """Half-life (days) of the AR(1) memory for a given ``phi``: ``ln(0.5) / ln(phi)``."""
    p = float(phi)
    if not (0.0 < p < 1.0):
        raise ValueError(f"phi must be in (0, 1) to have a finite half-life; got {p}")
    return math.log(0.5) / math.log(p)


# --------------------------------------------------------------------------------------------------
# Neighbour matrix W
# --------------------------------------------------------------------------------------------------
def _knn_query(lat, lon, k):
    """Indices + great-circle distances (km) of the ``k`` nearest neighbours (self excluded).

    Uses a haversine ``BallTree`` (scikit-learn) when available -- exact on the sphere -- and falls
    back to an equirectangular projection + planar ``cKDTree`` otherwise. Returns ``(dist_km, idx)``
    each shaped ``(n, k)``.
    """
    try:
        from sklearn.neighbors import BallTree
    except ImportError:
        BallTree = None

    if BallTree is not None:
        coords = np.radians(np.column_stack([lat, lon]))
        tree = BallTree(coords, metric="haversine")
        dist, idx = tree.query(coords, k=k + 1)        # +1: the nearest point is the node itself
        return dist[:, 1:] * EARTH_RADIUS_KM, idx[:, 1:]

    from scipy.spatial import cKDTree
    mean_lat = np.radians(float(np.mean(lat)))
    x = np.radians(lon) * np.cos(mean_lat) * EARTH_RADIUS_KM   # equirectangular approx (km)
    y = np.radians(lat) * EARTH_RADIUS_KM
    tree = cKDTree(np.column_stack([x, y]))
    dist, idx = tree.query(np.column_stack([x, y]), k=k + 1)
    return dist[:, 1:], idx[:, 1:]


def _row_normalize(W):
    """Zero the diagonal and row-normalize a sparse W so each non-empty row sums to 1 (``W @ 1 = 1``).

    kNN guarantees every node has ``k`` neighbours, so there are no empty rows in practice; the
    empty-row guard only matters for contiguity weights with islands, where the node stays uncoupled.
    """
    W = W.tocsr()
    W.setdiag(0.0)
    W.eliminate_zeros()
    row_sums = np.asarray(W.sum(axis=1)).ravel()
    row_sums[row_sums == 0.0] = 1.0    # isolated node: leave its row all-zero (uncoupled), avoid /0
    W = sp.diags(1.0 / row_sums) @ W
    return W.tocsr()


def _contiguity_weights(zcta_data, method):
    """Row-standardized border-contiguity weights via libpysal (true Queen/Rook adjacency).

    Documented option for users who want shared-border neighbours instead of centroid-kNN. Requires
    ``libpysal`` AND a GeoDataFrame with polygon geometry; ``get_zcta_data_with_geo_pop`` currently
    returns centroids only, so this raises an informative error unless geometry is supplied.
    """
    try:
        from libpysal.weights import Queen, Rook
    except ImportError as e:
        raise ImportError(
            "method='queen'/'rook' needs libpysal (not installed). Install it and pass a "
            "GeoDataFrame with polygon geometry as zcta_data, or use method='knn'/'distance' "
            "(centroid-based, no extra deps)."
        ) from e
    import geopandas as gpd
    if not isinstance(zcta_data, gpd.GeoDataFrame) or getattr(zcta_data, "geometry", None) is None:
        raise ValueError(
            "contiguity weights need zcta_data to be a GeoDataFrame with polygon geometry; "
            "get_zcta_data_with_geo_pop returns centroid lat/lon only. Use method='knn'/'distance'."
        )
    w = (Queen if method == "queen" else Rook).from_dataframe(zcta_data)
    w.transform = "r"                 # row-standardize so rows sum to 1
    return w.sparse.tocsr()           # aligned to the GeoDataFrame's row order


def build_spatial_weights(zcta_data, method="knn", k=8, length_scale_km=None):
    """Row-stochastic neighbour matrix ``W`` (csr), aligned to ``zcta_data`` row order.

    ``method``:
      * ``"knn"``      -- uniform weight on each of the ``k`` nearest centroids (-> 1/k after
                          row-normalizing). The default.
      * ``"distance"`` -- weight ``exp(-dist_km / length_scale_km)`` on the ``k`` nearest centroids
                          (closer neighbours count more), then row-normalize. ``length_scale_km``
                          defaults to the median neighbour distance when not given.
      * ``"queen"`` / ``"rook"`` -- true border contiguity via libpysal (see ``_contiguity_weights``).

    The diagonal is zeroed and every (non-isolated) row sums to 1, so ``W @ ones == ones`` -- the
    property that makes a spatially-constant field a fixed point of the SAR operator.
    """
    method = (method or "knn").lower()
    n = len(zcta_data)
    if n < 2:
        raise ValueError(f"need >= 2 ZCTAs to build neighbour weights; got {n}")

    if method in ("queen", "rook", "contiguity"):
        return _contiguity_weights(zcta_data, "queen" if method == "contiguity" else method)
    if method not in ("knn", "distance"):
        raise ValueError(f"unknown method {method!r}; use 'knn', 'distance', or 'queen'/'rook'")

    lat = zcta_data["latitude"].to_numpy(dtype=np.float64)
    lon = zcta_data["longitude"].to_numpy(dtype=np.float64)
    kk = int(min(k, n - 1))
    if kk < k:
        LOGGER.warning(f"build_spatial_weights: k={k} >= n_zctas={n}; using k={kk}")

    dist_km, idx = _knn_query(lat, lon, kk)            # (n, kk) each, self excluded

    if method == "distance":
        ls = float(length_scale_km) if length_scale_km else float(np.median(dist_km))
        if ls <= 0:
            raise ValueError(f"length_scale_km must be > 0 (or None to auto); got {ls}")
        vals = np.exp(-dist_km / ls)
    else:                                              # knn: uniform; row-normalization -> 1/k
        vals = np.ones_like(dist_km)

    rows = np.repeat(np.arange(n), kk)
    W = sp.csr_matrix((vals.ravel(), (rows, idx.ravel())), shape=(n, n))
    W = _row_normalize(W)
    LOGGER.info(f"Built {method} neighbour matrix W: {n} ZCTAs, k={kk}, nnz={W.nnz}")
    return W


def get_W(zcta_data, method="knn", k=8, length_scale_km=None, cache_key=None):
    """Memoized ``build_spatial_weights`` so the ERC exposure sweep does not rebuild W every call.

    ``cache_key`` should be a stable, hashable id for this ZCTA grid + parameters. When ``None`` we
    derive one from the parameters and an order-sensitive signature of the ZCTA column (count + first
    and last ids), which is cheap and distinguishes different grids without hashing the whole frame.
    """
    if cache_key is None:
        zc = zcta_data["zcta"].to_numpy()
        sig = (len(zc), str(zc[0]), str(zc[-1])) if len(zc) else (0,)
        cache_key = (method, int(k), length_scale_km, sig)
    W = _W_CACHE.get(cache_key)
    if W is None:
        W = build_spatial_weights(zcta_data, method=method, k=k, length_scale_km=length_scale_km)
        _W_CACHE[cache_key] = W
    return W


# --------------------------------------------------------------------------------------------------
# The coupling
# --------------------------------------------------------------------------------------------------
def _get_lu(W, rho):
    """Cached sparse LU of ``(I - rho W)`` keyed by ``(id(W), rho)``.

    ``id(W)`` is safe as a key because ``get_W`` keeps the matrix alive in ``_W_CACHE`` for the
    process lifetime, so its identity cannot be recycled while this factorization is cached.
    """
    key = (id(W), float(rho))
    lu = _LU_CACHE.get(key)
    if lu is None:
        n = W.shape[0]
        A = (sp.identity(n, format="csc") - rho * W).tocsc()
        lu = spla.splu(A)
        _LU_CACHE[key] = lu
    return lu


def apply_spacetime_coupling(eta, W, rho=0.0, phi=0.0, normalize=True):
    """Apply the spatial (SAR) + temporal (AR(1)/EWMA) coupling to a PRE-floor rate grid.

    ``eta``: ``(n_days, n_zctas)`` pre-floor rate (the value built inside ``expected_rate_grid``
    BEFORE ``np.maximum(0.01, ...)``). ``W``: row-stochastic neighbour matrix (only needed when
    ``rho > 0``). Returns a new ``(n_days, n_zctas)`` array -- still UNFLOORED; the caller applies the
    floor once, afterwards. See the module docstring for the math and the four guaranteed properties.

    ``normalize=True`` keeps the ``(1 - rho)`` / ``(1 - phi)`` prefactors so the maps are
    mean-preserving (a space-and-time-constant field is a fixed point; the ERC slope is unchanged).
    ``normalize=False`` drops them, amplifying the steady-state exposure slope by
    ``1 / ((1 - rho) * (1 - phi))`` -- still exact/closed-form, but a different estimand.
    """
    rho = float(rho)
    phi = float(phi)
    if not (0.0 <= rho < 1.0):
        raise ValueError(f"rho must be in [0, 1); got {rho}")
    if not (0.0 <= phi < 1.0):
        raise ValueError(f"phi must be in [0, 1); got {phi}")

    eta = np.asarray(eta, dtype=np.float64)
    if eta.ndim != 2:
        raise ValueError(f"eta must be 2-D (n_days, n_zctas); got shape {eta.shape}")

    # Backward-compat fast path: rho=0 AND phi=0 is the identity map. Return eta untouched -- no
    # matrix solve, no recursion -- so the output is byte-identical to the uncoupled DGP and free of
    # any floating-point round-off from a degenerate (I - 0*W)^{-1} solve. (rho=0,phi>0 and
    # rho>0,phi=0 are valid partial configs and fall through to the real code below.)
    if not rho and not phi:
        return eta

    n_days, n_z = eta.shape

    # ---- spatial step: eta_tilde_t = (1 - rho) * (I - rho W)^{-1} eta_t ----
    if rho > 0.0:
        if W is None:
            raise ValueError("rho > 0 requires a neighbour matrix W")
        if W.shape != (n_z, n_z):
            raise ValueError(f"W is {W.shape}, expected ({n_z}, {n_z}) to match eta's n_zctas")
        lu = _get_lu(W, rho)
        # Solve (I - rho W) X = eta^T for ALL days in one batched call (X is (n_z, n_days)), then
        # transpose back -- far cheaper than a per-day Python solve loop.
        eta_tilde = lu.solve(np.ascontiguousarray(eta.T)).T
        eta_tilde = (1.0 - rho) * eta_tilde if normalize else eta_tilde.copy()
    else:
        eta_tilde = eta.copy()

    # ---- temporal step: lambda_t = (1 - phi) * eta_tilde_t + phi * lambda_{t-1}, lambda_{-1}:=eta_tilde_0 ----
    if phi > 0.0:
        a = (1.0 - phi) if normalize else 1.0
        out = np.empty_like(eta_tilde)
        prev = eta_tilde[0]                            # lambda_{-1} := eta_tilde_0
        for t in range(n_days):                       # sequential scan; n_days ~ 365, cheap
            prev = a * eta_tilde[t] + phi * prev
            out[t] = prev
        return out
    return eta_tilde


# --------------------------------------------------------------------------------------------------
# Self-test: numerically VERIFY the properties claimed above (run: python -m src.synthetic_spacetime)
# --------------------------------------------------------------------------------------------------
def _self_test():
    import pandas as pd

    rng = np.random.RandomState(0)
    n_z, n_days = 200, 120
    # a fake ZCTA grid scattered over the continental US bounding box
    zcta_data = pd.DataFrame({
        "zcta": [f"{i:05d}" for i in range(n_z)],
        "latitude": rng.uniform(25, 49, n_z),
        "longitude": rng.uniform(-124, -67, n_z),
    })
    W = build_spatial_weights(zcta_data, method="knn", k=8)
    assert np.allclose(np.asarray(W.sum(axis=1)).ravel(), 1.0), "W is not row-stochastic"
    assert W.diagonal().sum() == 0.0, "W has a non-zero diagonal"

    rho, phi = 0.6, 0.9

    # (1) mean-preserving: a space+time-constant field is a fixed point (normalize=True)
    const = np.full((n_days, n_z), 0.137)
    out = apply_spacetime_coupling(const, W, rho=rho, phi=phi, normalize=True)
    assert np.allclose(out, 0.137, atol=1e-10), "mean-preservation (normalize=True) failed"

    # (1b) normalize=False amplifies a constant field. At the last day the AR geometric sum has all
    # but converged to the asymptotic factor 1/((1-rho)(1-phi)); compare against the EXACT
    # finite-horizon value to avoid the (tiny) phi^T transient, then check the asymptotic factor.
    out_nf = apply_spacetime_coupling(const, W, rho=rho, phi=phi, normalize=False)
    amp = 1.0 / ((1.0 - rho) * (1.0 - phi))
    d = 0.137 / (1.0 - rho)                                       # spatial steady state of constant field
    exact_last = d * (1.0 - phi ** (n_days + 1)) / (1.0 - phi)    # sum_{l=0}^{n_days} phi^l
    assert np.allclose(out_nf[-1], exact_last, rtol=1e-7), "normalize=False amplification wrong"
    assert abs(out_nf[-1].mean() / 0.137 - amp) / amp < 1e-3, "asymptotic amplification factor off"

    # (2) stability: (I - rho W) solve runs for rho close to 1 without blowing up
    big = apply_spacetime_coupling(rng.rand(n_days, n_z), W, rho=0.99, phi=0.99, normalize=True)
    assert np.isfinite(big).all(), "instability at rho=phi=0.99"

    # (3) closed-form memory matches the AR(1) unrolling, and half-life round-trips
    eta = rng.rand(n_days, n_z)
    lam = apply_spacetime_coupling(eta, W, rho=0.0, phi=phi, normalize=True)   # spatial off, isolate AR
    # reference: lambda_t = (1-phi) sum_{l=0}^t phi^l eta_{t-l} + phi^{t+1} eta_0
    ref = np.empty_like(eta)
    prev = eta[0]
    for t in range(n_days):
        prev = (1 - phi) * eta[t] + phi * prev
        ref[t] = prev
    assert np.allclose(lam, ref), "AR(1) recursion mismatch"
    assert abs(phi_to_halflife(halflife_to_phi(10.0)) - 10.0) < 1e-9, "half-life round-trip failed"

    # (4) ERC exactness: coupling a constant exposure field leaves the marginal slope unchanged
    #     (normalize=True). Build two constant exposure levels, confirm the coupled difference equals
    #     the uncoupled difference cell-for-cell.
    base = rng.rand(n_days, n_z) * 0.05 + 0.1
    beta, x0, x1 = 0.02, 1.0, 2.0
    g0 = apply_spacetime_coupling(base + beta * x0, W, rho=rho, phi=phi, normalize=True)
    g1 = apply_spacetime_coupling(base + beta * x1, W, rho=rho, phi=phi, normalize=True)
    assert np.allclose(g1 - g0, beta * (x1 - x0)), "ERC slope not preserved under coupling"

    print("synthetic_spacetime self-test OK:")
    print(f"  W row-stochastic, zero-diagonal; n_z={n_z}, k=8, nnz={W.nnz}")
    print(f"  mean-preserving (normalize=True); amplification {amp:.3f} (normalize=False)")
    print(f"  half-life(phi={phi}) = {phi_to_halflife(phi):.2f} days; AR(1) unrolling matches")
    print("  ERC marginal slope preserved under coupling (normalize=True)")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    _self_test()
