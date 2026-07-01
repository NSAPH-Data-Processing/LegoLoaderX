# Spatio-temporal coupling for the semi-synthetic health-count DGP

## 1. Overview

This feature adds optional **spatial neighbour coupling** and **temporal memory** to the latent Poisson rate of the semi-synthetic health-count data-generating process (DGP). Spatial coupling is a simultaneous-autoregressive (SAR) smoothing across neighbouring ZCTAs via a row-stochastic weight matrix `$W$`; temporal coupling is an AR(1)/EWMA smoothing over days. Both are applied to the **pre-floor** additive rate, before the `$\max(0.01, \cdot)$` floor and the Poisson draw. By default (`normalize=true`) the coupling is **mean-preserving**: it keeps the rate bounded and low, and — crucially for validation — leaves the ground-truth exposure–response curve (ERC) slope exactly equal to `$\beta$`, so g-computation ground truth stays closed-form. Setting both `rho=0` and `phi=0` (or omitting the block) reproduces the original uncoupled DGP **byte-for-byte**.

## 2. The data-generating process

Let there be `$V$` ZCTAs indexed by `$i$` and `$D$` days indexed by `$t$`. The pipeline builds a rate grid of shape `$(D, V)$`.

**Step 1 — additive pre-floor rate.** For each day `$t$`, an additive latent rate vector `$\eta_t \in \mathbb{R}^V$` is assembled from base + seasonal + latitude + longitude terms plus the causal terms (exposure, confounders, interactions):

```math
\eta_{t,i} = \text{base} + \text{seasonal}_t + \text{lat}_i + \text{lon}_i + \beta\,\mathrm{shape}(x_{t,i}) + \sum_k \gamma_k\,\mathrm{shape}\!\big(\mathrm{std}(C_{k,t,i})\big) + \text{interaction terms}
```

`$\eta_t$` is a genuine pre-floor value and may be negative. **No floor is applied yet.**

**Step 2 — spatial SAR coupling** (applied per day, only when `$\rho > 0$`). Let `$W$` be a row-stochastic matrix with zero diagonal (`$W\mathbf{1}=\mathbf{1}$`, `$W_{ii}=0$`). With `$\rho \in [0,1)$`:

```math
\tilde{\eta}_t = (1-\rho)\,(I - \rho W)^{-1}\,\eta_t \qquad (\text{normalize=true})
```
```math
\tilde{\eta}_t = (I - \rho W)^{-1}\,\eta_t \qquad (\text{normalize=false})
```

The `$(1-\rho)$` prefactor under `normalize=true` makes the operator mean-preserving. When `$\rho=0$` the spatial step is skipped and `$\tilde{\eta}_t = \eta_t$`.

**Step 3 — temporal AR(1)/EWMA coupling** (applied over days, only when `$\phi > 0$`). Let `$a = 1-\phi$` under `normalize=true`, `$a = 1$` under `normalize=false`, with `$\phi \in [0,1)$`:

```math
\lambda_t = a\,\tilde{\eta}_t + \phi\,\lambda_{t-1}, \qquad \lambda_{-1} := \tilde{\eta}_0
```

This is a causal (one-sided) recursion: `$\lambda_t$` depends only on the present and past. When `$\phi=0$` the temporal step is skipped and `$\lambda_t = \tilde{\eta}_t$`.

**Step 4 — floor.** The coupled rate is floored **once**, after all coupling:

```math
r_{t,i} = \max(0.01,\ \lambda_{t,i})
```

The floor is deliberately applied only at the end. Flooring inside the SAR/AR(1) recursion would break the closed-form (linear) properties, so `apply_spacetime_coupling` returns **unfloored** values and the caller floors afterward.

**Step 5 — Poisson draw.** With the per-ZCTA offset `$o_i = \text{pop\_norm}\cdot\text{pop}_i$`:

```math
y_{t,i} \sim \mathrm{Poisson}\!\big(r_{t,i}\, o_i\big)
```

The full composed rate, matching the manifest description string, is:

```math
r = \max\!\Big(\text{rate\_floor},\ \text{spacetime\_coupling}\big(\ \text{base} + \text{seasonal} + \text{lat} + \text{lon} + \beta\,\mathrm{shape}(\text{exposure}) + \textstyle\sum_k \gamma_k\,\mathrm{shape}(\mathrm{std}(C_k)) + \text{interaction terms}\ \big)\Big)
```

where `spacetime_coupling = AR(1)_phi over days of SAR_rho over neighbours` (a no-op when `rho=phi=0`).

**Definitions.**
- `$W$` — row-stochastic spatial weight matrix, zero diagonal, aligned to `zcta_data` row order. Non-isolated rows sum to 1; isolated rows stay all-zero (uncoupled).
- `$\rho \in [0,1)$` — spatial neighbour weight. Larger `$\rho$` = stronger smoothing toward neighbours.
- `$\phi \in [0,1)$` — temporal memory weight. Larger `$\phi$` = longer memory; half-life `$\ell_{1/2}=\ln 0.5/\ln\phi$` days.
- `normalize` — `true` keeps the `$(1-\rho)$` and `$(1-\phi)$` prefactors (mean-preserving); `false` drops them (amplifies the steady-state exposure slope).

## 3. Properties & guarantees

**Mean-preservation / stays-low (normalize=true).** A field constant in space and time is a fixed point. For a spatially constant `$\eta$`, `$W\eta=\eta$` so `$(I-\rho W)^{-1}\eta = \eta/(1-\rho)$`, and the `$(1-\rho)$` prefactor cancels it: `$\tilde\eta=\eta$`. Likewise the AR(1) steady state of a constant input `$c$` is `$c$` when `$a=1-\phi$`. Hence coupling neither inflates nor collapses the overall rate level, keeping it bounded and low. (Verified numerically: constant `0.137` returns unchanged.)

**Stability near the unit boundary.** For `$\rho\in[0,1)$`, `$(I - \rho W)$` is nonsingular (`$W$` has spectral radius 1 as a row-stochastic matrix, so `$\rho W$` has spectral radius `$\rho<1$`), so the SAR solve is well-posed; the AR(1) recursion with `$\phi\in[0,1)$` is a contraction. At `$\rho=\phi=0.99$` the output is all-finite.

**Closed-form geometric memory.** The AR(1)/EWMA recursion is an exact geometric filter. Its impulse decays as `$\phi^{\Delta t}$`, giving a half-life

```math
\ell_{1/2} = \frac{\ln 0.5}{\ln \phi}\ \text{days}, \qquad \phi = 0.5^{1/\ell_{1/2}}.
```

These conversions (`halflife_to_phi` / `phi_to_halflife`) round-trip exactly.

**ERC stays exact.** Because coupling is linear, an additive exposure shift propagates linearly through it. Under `normalize=true` the `do(exposure)` intervention slope on unfloored cells is preserved exactly:

```math
g_1 - g_0 = \beta\,\Delta x
```

Under `normalize=false`, the same slope is amplified by the closed-form steady-state factor. Asymptotically the amplification is

```math
\frac{1}{(1-\rho)(1-\phi)},
```

and over a finite horizon of `$D$` days the exact temporal factor is `$\tfrac{1-\phi^{D}}{1-\phi}$`, so for `$D=365$`:

```math
g_1 - g_0 = \frac{\beta\,\Delta x}{1-\rho}\cdot\frac{1-\phi^{365}}{1-\phi}.
```

Both estimands are exact and closed-form — `false` simply represents a different (amplified) estimand.

**Floor caveat.** The floor `$\max(0.01,\cdot)$` is a nonlinearity, so the slope guarantees above hold **exactly only on unfloored cells** (`$g_0 > 0.01$`). The ERC slope tests mask with `g0 > 0.01 + 1e-9` before asserting.

## 4. Configuration reference

All keys live under `synthetic.spacetime` in `conf/synthetic/config.yaml`.

| Key | Type | Default | Meaning |
|---|---|---|---|
| `rho` | float in `[0, 1)` | `0.0` | Spatial neighbour weight. `0` disables spatial coupling. |
| `phi` | float in `[0, 1)` | `0.0` | Temporal memory weight. `0` disables temporal coupling. Half-life `= ln(0.5)/ln(phi)` days. |
| `normalize` | bool | `true` | `true` = mean-preserving; `false` = amplify exposure slope by `1/((1-rho)(1-phi))`. |
| `method` | `'knn'` \| `'distance'` \| `'queen'`/`'rook'` | `knn` | Neighbour construction. `knn`/`distance` are centroid-based; `queen`/`rook` use libpysal border contiguity. |
| `k` | int | `8` | Number of nearest neighbours (used by `knn` and `distance`). |
| `length_scale_km` | float \| `null` | `null` | Only used by `method='distance'`; `null` = median neighbour distance. |

**Backward-compatibility note (verbatim from the config):**

```
# BACKWARD COMPAT: rho=0, phi=0 (or omit this whole block) reproduces the original DGP EXACTLY,
# byte-for-byte -- no spatial/temporal coupling. The normalize/method/k/length_scale_km fields are
# then irrelevant and do not affect output.
```

```
# rho=phi=0 (or omitting this block) is a no-op: byte-identical to the uncoupled DGP.
```

## 5. Python API reference (`src/synthetic_spacetime.py`)

Module constant: `EARTH_RADIUS_KM = 6371.0088`. Two process-lifetime caches back the API: `_W_CACHE` (built `$W$` matrices) and `_LU_CACHE` (LU factorizations of `$(I-\rho W)$`).

> All functions in this module return **unfloored** values. The `$\max(0.01,\cdot)$` floor is applied once by the caller, after coupling.

### `halflife_to_phi(halflife_days)`

```python
def halflife_to_phi(halflife_days):
```
Convert a temporal half-life (days) to the AR(1) weight.

- **Returns:** `$\phi = 0.5^{1/h}$` where `$h=\text{float}(\text{halflife\_days})$`.
- **Raises:** `ValueError` when `$h \le 0$` (`"halflife_days must be > 0; got {h}"`).

### `phi_to_halflife(phi)`

```python
def phi_to_halflife(phi):
```
Inverse of `halflife_to_phi`.

- **Returns:** `$\ln(0.5)/\ln(p)$` where `$p=\text{float}(\text{phi})$`.
- **Raises:** `ValueError` when not `$0 < p < 1$` (`"phi must be in (0, 1) to have a finite half-life; got {p}"`).

Round-trips exactly: `phi_to_halflife(halflife_to_phi(10.0)) ≈ 10.0`.

### `build_spatial_weights(zcta_data, method="knn", k=8, length_scale_km=None)`

```python
def build_spatial_weights(zcta_data, method="knn", k=8, length_scale_km=None):
```
Build a row-stochastic csr neighbour matrix `$W$`, aligned to `zcta_data` row order, zero diagonal, each non-isolated row summing to 1. Coordinates are read from the `"latitude"` and `"longitude"` columns.

- **`method`** (`method = (method or "knn").lower()`):
  - `"knn"` — uniform weight `1` on the `k` nearest centroids → `$1/k$` after row-normalization.
  - `"distance"` — `$\exp(-d/\text{ls})$` on the `k` nearest centroids, then row-normalize. `ls = length_scale_km` if truthy else the **median neighbour distance**.
  - `"queen"` / `"rook"` / `"contiguity"` — dispatch to libpysal true border contiguity (`"contiguity"` maps to Queen).
- **`k` clamping:** `kk = int(min(k, n-1))`; if `kk < k` logs a warning but does not error.
- **Returns:** csr `$W$` of shape `$(n,n)$`.
- **Raises:**
  - `ValueError` if `$n = \text{len(zcta\_data)} < 2$` (`"need >= 2 ZCTAs to build neighbour weights; got {n}"`).
  - `ValueError` if `method` not in `("knn","distance")` after the contiguity branch (`"unknown method {method!r}; use 'knn', 'distance', or 'queen'/'rook'"`).
  - `ValueError` for `"distance"` if `ls <= 0` (`"length_scale_km must be > 0 (or None to auto); got {ls}"`).
  - For contiguity: `ImportError` if libpysal is missing; `ValueError` if `zcta_data` is not a `geopandas.GeoDataFrame` with polygon geometry.

### `get_W(zcta_data, method="knn", k=8, length_scale_km=None, cache_key=None)`

```python
def get_W(zcta_data, method="knn", k=8, length_scale_km=None, cache_key=None):
```
Memoized `build_spatial_weights`.

- **`cache_key`** — when `None`, derived from column `"zcta"` as `zc`: `sig = (len(zc), str(zc[0]), str(zc[-1]))` (or `(0,)` if empty), then `cache_key = (method, int(k), length_scale_km, sig)`. Order-sensitive (count + first + last id).
- **Returns:** the cached (or newly built and cached) csr `$W$`.
- **Raises:** nothing beyond what `build_spatial_weights` raises.

### `apply_spacetime_coupling(eta, W, rho=0.0, phi=0.0, normalize=True)`

```python
def apply_spacetime_coupling(eta, W, rho=0.0, phi=0.0, normalize=True):
```
Apply spatial (SAR) then temporal (AR(1)/EWMA) coupling to a **pre-floor** rate grid `eta` of shape `$(n\_days, n\_zctas)$`. Returns a new unfloored `$(n\_days, n\_zctas)$` array.

- **Spatial** (only when `$\rho>0$`): batched LU solve of `$(I-\rho W)X = \eta^\top$`; `normalize=True` multiplies by `$(1-\rho)$`, `normalize=False` does not. When `$\rho=0$` the spatial step is skipped.
- **Temporal** (only when `$\phi>0$`): `$\lambda_t = a\,\tilde\eta_t + \phi\,\lambda_{t-1}$` with `$a=1-\phi$` (normalize) or `$a=1$`, and `$\lambda_{-1}:=\tilde\eta_0$`. When `$\phi=0$` the temporal step is skipped.
- **Early-return identity:** if `not rho and not phi`, returns `eta` (coerced to float64 via `np.asarray`) untouched — no solve, no recursion, byte-identical to the uncoupled DGP. (`rho=0,phi>0` and `rho>0,phi=0` fall through to the real code.)
- **Raises** (in order):
  - `ValueError` if not `$0 \le \rho < 1$` (`"rho must be in [0, 1); got {rho}"`).
  - `ValueError` if not `$0 \le \phi < 1$` (`"phi must be in [0, 1); got {phi}"`).
  - `ValueError` if `eta.ndim != 2` (`"eta must be 2-D (n_days, n_zctas); got shape {eta.shape}"`).
  - In the spatial branch: `ValueError` if `W is None` (`"rho > 0 requires a neighbour matrix W"`); `ValueError` if `W.shape != (n_z, n_z)`.
  - `W` is only required/validated when `$\rho>0$` (unused when `$\rho=0$`).

## 6. How it wires in

**Coupling is applied pre-floor, in the shared rate builder.** In `src/synthetic_causal.py` (`expected_rate_grid`, lines 270–280), the block is read from `cfg.synthetic.spacetime`:

```python
st = cfg.synthetic.get("spacetime", None) or {}
rho = float(st.get("rho", 0.0) or 0.0)
phi = float(st.get("phi", 0.0) or 0.0)
if rho or phi:
    from src.synthetic_spacetime import apply_spacetime_coupling, get_W
    W = get_W(zcta_data, method=st.get("method", "knn"), k=int(st.get("k", 8)),
              length_scale_km=st.get("length_scale_km", None)) if rho else None
    rate = apply_spacetime_coupling(rate, W, rho=rho, phi=phi,
                                    normalize=bool(st.get("normalize", True)))

return np.maximum(0.01, rate)
```

- The additive `rate` (base + seasonal + lat + lon) plus causal `extra` terms is built first; coupling reassigns `rate` while it is still the raw pre-floor value; `np.maximum(0.01, rate)` runs only on the returned line.
- The whole coupling path lives inside `if rho or phi:`, so when both are `0` (or the block is absent) it is a no-op and output is byte-identical to the uncoupled DGP.
- `get_W(...)` is called **only when `rho != 0`**; with pure temporal coupling (`rho=0, phi>0`) `W` is `None` and `get_W` is never invoked.
- Because this same `expected_rate_grid` is the single source of truth for the DGP rate and is also consumed by `ground_truth_erc.py`, the sampled data and the ERC answer key cannot drift.

**One vectorized Poisson draw.** `generate_synthetic_data` now takes a **precomputed** grid and offset:

```python
def generate_synthetic_data(zcta_data, date_list, var_name, rate_grid, offset):
    ...
    counts = np.random.poisson(rate_grid * offset[None, :])
```

A single `np.random.poisson` call over the whole `$(n\_days, n\_zctas)$` grid replaces the old per-day loop; it is byte-identical because `np.random.poisson` consumes the RNG stream in C order (day-by-day, then ZCTA-by-ZCTA) — the same order the old loop drew. The subsequent per-day work only melts `counts` into the long `(zcta, var, date, n)` format and drops zeros (`nz = day_counts > 0`); no extra RNG draws occur. In debug mode the grid is truncated to the leading `len(date_list)` days (`rate_grid = rate_grid[:n_days]`), which is valid because temporal coupling is causal. (If `date_list` is *longer* than the grid it raises a `ValueError`.)

`main` builds the rate and offset **once** and hands them to the single sampler call:

```python
rate_grid = expected_rate_grid(cfg, zcta_data)   # floored rate: background + seasonal + geo + causal + coupling
offset    = offset_vector(cfg, zcta_data)
...
disease_df = generate_synthetic_data(
    zcta_data, days_list, cfg.synthetic.var_name, rate_grid, offset,
)
```

**Manifest & ERC metadata persist the block.**
- `spacetime` is the final entry in `DGP_KEYS` in `src/synthetic_manifest.py`, so the coupling parameters are pulled from the manifest (not the live config) when re-deriving ground truth — the answer key matches the data on disk.
- `build_manifest`'s `dgp` description string wraps the pre-floor rate in `spacetime_coupling(...)` and documents `spacetime_coupling = AR(1)_phi over days of SAR_rho over neighbours (no-op when rho=phi=0)`.
- `ground_truth_erc.py` writes the fully-resolved `spacetime` config into `erc_ground_truth_<var>_<year>.meta.json` alongside `beta_true`, `confounders`, and `interaction_terms`, with a comment noting that `normalize=false` amplifies the represented exposure slope by `1/((1-rho)*(1-phi))`.

## 7. Usage examples

**(a) Config snippet enabling coupling** (`conf/synthetic/config.yaml`):

```yaml
  spacetime:
    rho: 0.5                # spatial neighbour weight in [0, 1)
    phi: 0.93               # temporal memory weight; half-life = ln(0.5)/ln(0.93) ~= 9.55 days
                            #   phi = 0.5 ** (1 / halflife_days); halflife = ln(0.5)/ln(phi)
                            #   e.g. 10-day half-life -> phi = 0.5 ** (1/10) ~= 0.9330
    normalize: true         # true = mean-preserving; false = amplify exposure slope by 1/((1-rho)(1-phi))
    method: knn             # 'knn' | 'distance' (centroid-based) | 'queen'/'rook' (libpysal contiguity)
    k: 8                    # number of nearest neighbours
    length_scale_km: null   # only used by method='distance'; null = median neighbour distance
```

Half-life ↔ phi conversion:

```python
from src.synthetic_spacetime import halflife_to_phi, phi_to_halflife
halflife_to_phi(10.0)   # -> 0.9330...  (10-day half-life)
phi_to_halflife(0.93)   # -> 9.55...    (days)
```

**(b) Regenerate data and recompute the ground-truth ERC:**

```bash
# from the repo root: /n/home09/jkleutgens/LegoLoaderX
python -m src.synthetic_health        # regenerate the synthetic counts with coupling applied
PYTHONPATH=. python -m src.ground_truth_erc year=2010 synthetic.var_name=diabetes   # recompute the closed-form ground-truth ERC + meta.json
```

## 8. Testing

**Run the pytest suite:**

```bash
# from repo root /n/home09/jkleutgens/LegoLoaderX
pytest tests/test_synthetic_spacetime.py
# or
python -m pytest tests/test_synthetic_spacetime.py
```

**Run the module self-test** (numerically verifies, on a fake 200-ZCTA / 120-day grid: row-stochastic `$W$` with zero diagonal; mean-preservation; `normalize=false` amplification vs the exact finite-horizon and asymptotic `$1/((1-\rho)(1-\phi))$` factors; stability at `$\rho=\phi=0.99$`; AR(1)/EWMA closed-form unrolling + half-life round-trip; ERC slope preservation under `normalize=true`):

```bash
python -m src.synthetic_spacetime
```

**What the suite verifies** (the pytest fixture uses a 150-ZCTA grid):

- `test_W_is_row_stochastic_zero_diagonal` — kNN `$W$` is `$(N,N)$`, rows sum to 1, zero diagonal.
- `test_W_distance_weights_closer_neighbours_more` — `method="distance"` `$W$` is still row-stochastic.
- `test_halflife_roundtrip` — `phi_to_halflife(halflife_to_phi(h)) == h`.
- `test_halflife_validation` — `halflife_to_phi(0.0)` and `phi_to_halflife(1.0)` raise `ValueError`.
- `test_rho_phi_validation` — `apply_spacetime_coupling` raises `ValueError` for `rho`/`phi` outside `[0,1)`.
- `test_identity_when_rho_and_phi_zero` — `rho=0 AND phi=0` returns the *same* array object (byte-identical).
- `test_mean_preserving_normalize_true` — a space-and-time-constant field is a fixed point (constant `0.137` unchanged).
- `test_amplification_normalize_false` — a constant field is amplified toward the exact finite-horizon value and asymptotically by `$1/((1-\rho)(1-\phi))$`.
- `test_ar1_matches_closed_form_unrolling` — temporal-only coupling equals the explicit AR(1)/EWMA unrolling.
- `test_stability_near_unit` — `rho=0.99, phi=0.99` produces all-finite output.
- `test_backward_compat_omitted_equals_zero_grid` — omitting the block equals `{rho:0, phi:0}` byte-for-byte; other fields inert when `rho=phi=0`.
- `test_backward_compat_seeded_counts_identical` — byte-for-byte equivalence extends to seeded Poisson counts.
- `test_refactored_sampler_matches_per_day_loop` — one-shot whole-grid Poisson draw reproduces the old per-day loop (same C-order RNG stream).
- `test_partial_configs_valid` — temporal-only and spatial-only configs are valid, finite, and actually change the grid.
- `test_erc_slope_preserved_normalize_true` — `do(exposure)` slope stays exactly `$\beta\,\Delta x$` on unfloored cells.
- `test_erc_slope_amplified_normalize_false` — slope amplified by `$\tfrac{\beta\,\Delta x}{1-\rho}\cdot\tfrac{1-\phi^{365}}{1-\phi}$` on unfloored cells.
- `test_erc_sweep_reuses_caches` — an ERC exposure sweep reuses a single cached `$W$` (`_W_CACHE`) and a single LU factorization (`_LU_CACHE`).

## 9. Notes & caveats

- **Coupling acts on the deterministic rate, not the counts.** Smoothing the latent rate (before the Poisson draw) keeps the ERC closed-form: an additive exposure shift propagates linearly and the g-computation ground truth stays analytic. Coupling the realized *counts* (count feedback) would make the process nonlinear and require Monte Carlo to estimate the ERC — that is intentionally not what this does.
- **Floor nonlinearity.** The `$\max(0.01,\cdot)$` floor is nonlinear, so the slope-preservation and amplification guarantees hold exactly **only on unfloored cells** (`$g_0 > 0.01$`). Cells clamped at the floor do not carry the exact slope.
- **`W` alignment depends on `zcta_data` row order.** `$W$` is built and returned aligned to the input row order, and `get_W`'s auto cache key uses `(count, first zcta, last zcta)`. Reorder the ZCTAs and you get a different (correctly re-aligned) matrix; keep the row order stable across the rate build and the sampler.
- **libpysal is optional.** `knn` and `distance` methods use only sklearn/scipy centroid neighbours (`BallTree` haversine, or a `cKDTree` equirectangular fallback). The `queen`/`rook`/`contiguity` methods require libpysal **and** a `geopandas.GeoDataFrame` with polygon geometry; otherwise they raise `ImportError`/`ValueError` pointing you to `method='knn'/'distance'`.
