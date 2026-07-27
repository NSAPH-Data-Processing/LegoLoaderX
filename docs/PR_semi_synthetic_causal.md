# Semi-synthetic causal DGP for ZCTA-level health counts, with closed-form ground-truth ERC and spatial+temporal rate coupling

## 1. Summary

This PR adds a semi-synthetic, causal data-generating process (DGP) for ZCTA-level daily health counts whose per-cell Poisson rate is written by us as a sum of config-specified terms, so every dose-response coefficient (`beta`, each `gamma_k`, each interaction `delta_j`) is a **known ground truth**. It ships the closed-form ground-truth exposure-response curve (ERC) via g-computation (`do(exposure=a)`), a provenance-stamped manifest that travels with the data, and — the headline addition this session — an optional **spatial (SAR) + temporal (AR(1)/EWMA) coupling** of the latent rate. Because the coupling is linear and deterministic and mean-preserving by default, realistic spatial correlation and temporal persistence are injected without breaking the closed-form ERC, so downstream ERC / g-computation pipelines can be validated against a number we chose rather than a hoped-for estimate.

## 2. Motivation

Validating a causal exposure-response estimator on real data is circular: there is no ground truth to compare the recovered curve against. A semi-synthetic DGP fixes this by generating the outcome from **real covariate values** but with a rate we specify in closed form. The training model later fits on exactly those same covariates, so we can check whether its recovered ERC lands on the injected `beta`, and measure exactly how much bias a naive (unadjusted) fit carries.

Two properties are required for this to be trustworthy:

- **Realistic structure.** Real disease counts are spatially correlated (neighbouring ZCTAs share unobserved drivers) and temporally persistent (rates carry over across days). An independent-across-cells DGP would let estimators exploit an unrealistically clean signal. The new coupling makes ZCTA `i`'s rate on day `t` depend on its neighbours and its own recent past.
- **Closed-form ground truth.** The ERC must remain analytically computable — no Monte Carlo — or the "known truth" is itself an estimate with error bars. This constrains the coupling to be **linear and deterministic** and applied to the latent *rate*, not the realized *counts*.

## 3. Theory — the data-generating process

Let `t` index days and `i`/`z` index the `V` ZCTAs. The full pipeline is: additive pre-floor rate → spatial SAR → temporal AR(1)/EWMA → floor → Poisson with offset.

**(0) Additive pre-floor rate** (`expected_rate_grid`, `src/synthetic_causal.py:227-280`; background block `:259-262`):

```math
\eta_{t,i} = \underbrace{r_{\text{base}} + A_s\sin\!\Big(\tfrac{2\pi t}{365.25}\Big) + \ell_{\text{lat}}\sin\!\Big(\tfrac{(\mathrm{lat}_i-35)}{15}\pi\Big) + \ell_{\text{lon}}\cos\!\Big(\tfrac{(\mathrm{lon}_i+95)}{30}\pi\Big)}_{\text{background (seasonal + geography)}}
```
```math
\quad+\ \beta\,s\!\big(X^{\text{expo}}_{t,i}\big)\ +\ \sum_k \gamma_k\, s_k\!\big(\tilde C^{(k)}_{t,i}\big)\ +\ \sum_j \delta_j \prod_{r \in \text{refs}_j} \text{term}_{r,t,i}
```

- `beta*shape(exposure)` — exposure loaded **raw** (`standardize=False`), so `beta` is per physical PM2.5 unit — the natural ERC target (`_shaped_exposure`, `:127`).
- `sum_k gamma_k*shape_k(std(C_k))` — each confounder is nan-aware **standardized** over the whole file (`_confounder_value`, `:140`; `_load_covar_aligned`, `:94`), so `gamma_k` is a **per-standard-deviation** effect, comparable across covariates on different scales.
- interactions (interaction block in `build_extra_rate`, `:189-212`) add `delta_j * prod(refs_j)` **only when `synthetic.interactions: true`**; each ref is `"exposure"` (shaped) or a confounder `var` (standardized, always `shape="linear"`), encoding confounder×treatment or confounder×confounder effect modification. Not populated in the current config, so `extra` is purely additive.

**Fixed-form shapes** (`_apply_shape`, `:49-78`) — applied inside `expected_rate_grid`, so the ground-truth ERC follows the same curve automatically:

| shape | formula | epidemiology |
|---|---|---|
| `linear`/`identity` (default) | `x` | additive-linear DGP, reproduced byte-for-byte |
| `sqrt` | `sqrt(max(x,0))` | concave/saturating dose-response (sub-linear PM2.5) |
| `log1p` | `log(1+max(x,0))` | concave, more saturating |
| `quadratic`/`u_shape` | `(x-center)**2` | U-shaped risk about `center` (e.g. temperature) |
| `abs`/`v_shape` | `\|x-center\|` | V-shaped risk about `center` |

`center` shifts the U/V minimum (SD units for standardized confounders, raw units for exposure). Both `exposure_shape` and each confounder `shape` are `linear` in the current config.

**(1) Spatial SAR** (per day; `apply_spacetime_coupling`, `src/synthetic_spacetime.py`). Let `W` be a row-stochastic, zero-diagonal neighbour matrix (`W @ 1 = 1`) built from ZCTA centroids by kNN (uniform `1/k`), distance (`exp(-dist_km/length_scale_km)`), or queen/rook contiguity (libpysal):

```math
\tilde\eta_t = (1-\rho)\,(I - \rho W)^{-1}\,\eta_t,\qquad \rho \in [0,1)
```

Each cell is pulled toward a `rho`-weighted average of its neighbours; the inverse makes it *simultaneous* (neighbours-of-neighbours at all orders, geometrically damped).

**(2) Temporal AR(1)/EWMA** (over days; causal, past-only):

```math
\lambda_t = (1-\phi)\,\tilde\eta_t + \phi\,\lambda_{t-1},\qquad \lambda_{-1} := \tilde\eta_0,\qquad \phi \in [0,1)
```

Half-life `l_half = ln(0.5)/ln(phi)` days (`phi=0.93` ≈ 9.55-day memory).

**(3) Floor once, then Poisson** (floor applied by the caller, *after* both couplings — flooring inside the recursion would inject a nonlinearity and break every closed-form property):

```math
r_{t,i} = \max(0.01,\ \lambda_{t,i}),\qquad y_{t,i} \sim \text{Poisson}\big(r_{t,i}\cdot o_i\big)
```

where the offset `o_i = population_normalizer * population_i` (`offset_vector`, `:217-224`).

**`do(exposure=x)` intervention** (`exposure_override`): passing a scalar `x` sets the exposure term to `beta*shape(x)` in **every** `(day, zcta)` cell while confounders are still read from disk unchanged — precisely the g-computation operator: intervene on treatment, hold everything else at its observed distribution, average. `normalize` (default `True`) makes both operators mean-preserving; `normalize=False` drops the `(1-rho)`/`(1-phi)` prefactors (see §5).

## 4. Theory — closed-form ground-truth ERC

**Estimand (g-computation).** The ground truth is the population-average expected outcome under `do(exposure=a)`, using the **same two-loop collapse** the model's ERC targets. Per node, force exposure to `a` everywhere and sum the expected outcome over the forecast window:

```math
F_{\text{true}}(z,a) = \sum_{t \in T_{fc}} \mathbb{E}[Y_{z,t}\mid do(\text{exposure}=a)] = \sum_{t \in T_{fc}} \lambda_{z,t}(a)\,o_z
```
```math
F_{\text{true}}(a) = \tfrac{1}{V}\sum_z F_{\text{true}}(z,a)\ \ (\text{mean, default}) \qquad = \sum_z F_{\text{true}}(z,a)\ \ (\text{sum})
```

The `sum_t` collapse is exact because the observed outcome is a windowed count `Y_z = sum_t Y_{z,t}`, so `E[Y_z] = sum_t E[Y_{z,t}]`. The one-line estimand is written into the meta file (`ground_truth_erc.py:121`):

> `F(a) = node-aggregate over z of [ sum_{t in forecast} E[Y_{z,t}|do(exposure=a)] ]`

The sweep is one comprehension (`ground_truth_erc.py:107-113`):

```python
mu = [ marginal_outcome(expected_rate_grid(cfg, zcta_data, exposure_override=a),
                        offset, forecast=forecast, node_aggregation=node_agg)
       for a in xs ]
```

**Why generator and ground truth cannot drift.** `ground_truth_erc.py` imports and calls the *identical* `expected_rate_grid` / `offset_vector` / `marginal_outcome` that the generator draws its Poisson mean from (`src/synthetic_causal.py:227`, `:217`, `:283`). Nothing about the specific terms (`beta`, shapes, `gamma_k`, interactions, spacetime) is re-implemented in `ground_truth_erc.py`. Change the DGP's functional form in that one function and the ground truth follows automatically. `expected_rate_grid` is the **single source of truth** for the DGP rate.

- Forecast window / node aggregation read from `cfg.synthetic.erc` (`ground_truth_erc.py:85-90`): `forecast_start`/`forecast_len` → `slice`; `node_aggregation` `"mean"`/`"sum"`. The forecast window defaults to the whole year (`marginal_outcome` sums the expected count over all days per node); node aggregation defaults to `"mean"` — both in `marginal_outcome`'s signature and in the config (`erc.node_aggregation: mean`) — so the default ERC point is the per-node annual expected count averaged over nodes.
- Exposure grid: `erc.grid` (`x_min`/`x_max`/`n_points`) if pinned, else 25 points over the exposure's 1st–99th percentile (`:93-101`).

**Manifest (the answer key travels with the data).** The outcome is generated from causal params in `conf/synthetic/config.yaml`; if that file is later edited, a ground truth computed from the *live* config no longer matches the data on disk. So at generation time `write_manifest(cfg)` dumps a JSON snapshot of the fully-resolved `cfg.synthetic` block plus provenance (`schema_version`, `var`, `year`, `generated_at_utc`, `git_commit`, `data_file`, human-readable `dgp` string) **next to the counts parquet** (`sparse_counts_<var>_<year>.parquet` ↔ `sparse_counts_<var>_<year>.manifest.json`; `synthetic_manifest.py`, via `counts_path`/`manifest_path`). Unless `synthetic.erc.use_manifest=false`, `ground_truth_erc.py` (`lines 63-72`) calls `load_manifest` then `apply_dgp_from_manifest`, which overlays **only** `DGP_KEYS`

```
("beta", "exposure_covars_root", "exposure_var_group", "exposure_var", "exposure_shape",
 "confounders", "interactions", "interaction_terms", "poisson_params", "spacetime")
```

onto the live `cfg` via `OmegaConf.merge`. This split is deliberate: **DGP rate parameters** come from the manifest (guaranteed to match the data on disk), while **non-DGP settings** (ERC grid, forecast window, output dir, store paths) stay on the live config so they remain CLI-overridable. If no manifest is found it warns and falls back to the live config.

## 5. Properties & guarantees

Verified numerically by the module `_self_test()`, not merely asserted.

1. **Mean-preserving / stays-low** (`normalize=True`). A field constant in space AND time is a fixed point, because `W` is row-stochastic: `(I - rho*W)^{-1} @ 1 = 1/(1-rho)` cancels the `(1-rho)` prefactor, and the `(1-phi)` EWMA prefactor cancels for time-constant input. Coupling adds correlation without raising the level.
2. **Stability** (`rho<1`). A row-stochastic `W` has spectral radius 1, so `rho*W` has spectral radius `rho<1` and `(I - rho*W)` is invertible for any `rho in [0,1)`; the AR(1) with `phi in [0,1)` is a contraction. Applied sequentially → no joint `rho`+`phi` constraint (finite at `rho=phi=0.99`).
3. **Geometric memory + half-life.** Unrolling the AR(1) gives `lambda_t = (1-phi) sum_{l=0}^{t} phi^l * eta_tilde_{t-l} + phi^{t+1} * lambda_{-1}`; day `t-l` influences day `t` with weight `(1-phi)*phi^l`, half-life `ln(0.5)/ln(phi)`. `halflife_to_phi`/`phi_to_halflife` round-trip exactly.
4. **ERC stays exact** (no simulation). Both operators are linear and deterministic, so an additive `do(exposure=x)` shift (constant in space and time) propagates linearly. With `normalize=True` the constant field is a fixed point, so the marginal slope is unchanged: `g1 - g0 = beta * Δx`. With `normalize=False` the prefactors drop and the steady-state slope is **amplified by `1/((1-rho)(1-phi))`** — still exact/closed-form, but a different (amplified) estimand: pick `normalize` deliberately.
5. **Floor caveat.** `max(0.01, .)` is a nonlinearity, so slope-preservation / amplification hold *exactly only on unfloored cells* (`g0 > 0.01`). Slope tests mask with `g0 > 0.01 + 1e-9` before asserting.

## 6. What changed

**(a) Causal exposure/confounder foundation** (`src/synthetic_causal.py`)
- `_apply_shape` (`:49`) fixed-form non-linearities; `_shaped_exposure` (`:127`) exposure term + `do(exposure=x)`; `_confounder_value` (`:140`) standardized/optionally-shaped confounder; `build_extra_rate` (`:151`) assembles `beta*shape(exposure) + sum gamma_k*shape_k(std(C_k)) + interactions`.
- 5 exposures (PM2.5 `beta=0.01` raw + primary ERC target; NO2 `0.020`; O3 `0.015`; tmmx `0.020`; rmax `0.010`) and 5 yearly confounders (income `-0.020`, poverty `0.030`, over-65 `0.030`, Black `0.015`, median age `0.020`), grouped by role in config only so the dataloader can split them. A naive PM2.5-only fit does **not** recover `beta=0.01` because PM2.5 is correlated with the other terms; only a fully-adjusted model closes the back-door paths.

**(b) Ground-truth ERC + manifest**
- `ground_truth_erc.py` traces `F_true(a)` by sweeping `do(exposure=a)` through the shared `expected_rate_grid` / `offset_vector` / `marginal_outcome`.
- `synthetic_manifest.py` (`build_manifest`, `write_manifest`, `load_manifest`, `apply_dgp_from_manifest`, `DGP_KEYS`, `counts_path`/`manifest_path`) writes the provenance-stamped answer key beside the parquet.

**(c) NEW spatial + temporal coupling** (`src/synthetic_spacetime.py`, `docs/synthetic_spacetime.md`)
- Public API: `halflife_to_phi(halflife_days)`, `phi_to_halflife(phi)`, `build_spatial_weights(zcta_data, method="knn", k=8, length_scale_km=None)`, `get_W(zcta_data, method="knn", k=8, length_scale_km=None, cache_key=None)` (memoized), `apply_spacetime_coupling(eta, W, rho=0.0, phi=0.0, normalize=True)` (SAR then AR(1); returns **unfloored** `(n_days, n_zctas)`).
- Validates `rho, phi in [0,1)`, requires 2-D `eta`, validates `W` shape only when `rho>0`. `rho=0 AND phi=0` fast path returns `eta` untouched. Spatial solve is a single batched sparse LU solve `(I - rho*W) X = eta.T` cached in `_LU_CACHE`; `W` cached in `_W_CACHE` so an ERC sweep never rebuilds/refactorizes. New `spacetime` block in `conf/synthetic/config.yaml`.

**(d) Refactor of `generate_synthetic_data`** (`src/synthetic_health.py`)
- Now consumes the shared `expected_rate_grid` + `offset_vector` and Poisson-samples counts in **one vectorized draw** from the precomputed floored rate grid, removing the duplicated day-by-day rate rebuild (was diverging from the ERC's rate). It also calls `write_manifest(cfg)` after generation.

## 7. Backward compatibility

Omitting the `spacetime` block, or setting `rho=0, phi=0`, reproduces the original DGP **byte-for-byte**: `expected_rate_grid` takes the `rho or phi` branch only when non-zero (`:273`), and `apply_spacetime_coupling`'s `rho=0 AND phi=0` fast path returns `eta` untouched. This produces the identical rate grid and identical seeded counts. Covered by the regression tests in the suite below (`test_backward_compat_omitted_equals_zero_grid`, `test_backward_compat_seeded_counts_identical`).

## 8. Testing

`tests/test_synthetic_spacetime.py` (offline unit tests) plus the in-module `_self_test()`.

Verified:
- **Backward-compat regression (headline):** `rho=0, phi=0` (or omitting the block) reproduces the original rate grid and seeded counts byte-for-byte (`test_backward_compat_omitted_equals_zero_grid`, `test_backward_compat_seeded_counts_identical`); the refactored single-draw sampler also matches the old per-day loop (`test_refactored_sampler_matches_per_day_loop`).
- Mean-preserving / stays-low (`test_mean_preserving_normalize_true`: a space-and-time-constant field is a fixed point under `normalize=True`).
- Stability / finiteness up to `rho=phi=0.99` (`test_stability_near_unit`).
- Geometric memory (`test_ar1_matches_closed_form_unrolling`) + `halflife_to_phi`/`phi_to_halflife` round-trip (`test_halflife_roundtrip`).
- ERC slope preservation on unfloored cells (`test_erc_slope_preserved_normalize_true`: `normalize=True` → `g1-g0 = beta*Δx`; `test_erc_slope_amplified_normalize_false`: `normalize=False` → amplified by `1/((1-rho)(1-phi))`), masking `g0 > 0.01 + 1e-9`.
- ERC sweep reuses one `W` and one LU factorization (`test_erc_sweep_reuses_caches`).

Run commands:

```bash
pytest tests/test_synthetic_spacetime.py -v
python -m src.synthetic_spacetime          # numerical self-test of the guaranteed properties
```

## 9. Files changed

`git diff --stat origin/main..HEAD` — 6 commits (tip `25cce34`), 11 files, +1720 / −50. The index is clean (nothing staged); the only working-tree entry is the untracked `climhealth-fm/` (see below), which is intentionally excluded.

| File | Role |
|---|---|
| `src/synthetic_spacetime.py` | NEW spatial (SAR) + temporal (AR(1)/EWMA) coupling of the pre-floor rate; mean-preserving so the ERC slope stays exactly β; `rho=0,phi=0` = uncoupled DGP byte-for-byte |
| `src/synthetic_causal.py` | Known causal rate terms (`beta*shape(exposure) + sum gamma_k*shape(std(C_k)) + interactions`) + `expected_rate_grid`/`offset_vector`/`marginal_outcome` shared by generator and ERC |
| `src/synthetic_health.py` | Health-count generator; single vectorized Poisson draw from the floored rate grid + offset (refactored onto `synthetic_causal`), plus `write_manifest` |
| `src/synthetic_manifest.py` | Writes JSON manifest (resolved `cfg.synthetic` + provenance/git commit) next to the counts parquet; overlays only `DGP_KEYS` back onto live cfg |
| `src/ground_truth_erc.py` | Computes/saves the closed-form ground-truth ERC via `do(exposure=a)` using the same rate grid as the generator |
| `conf/synthetic/config.yaml` | Hydra DGP config: paths, seed, β/γ_k, shapes, interactions, new `spacetime` and `erc` blocks |
| `tests/test_synthetic_spacetime.py` | Offline unit tests; headline is the `rho=0,phi=0` byte-for-byte regression |
| `docs/synthetic_spacetime.md` | Derives the SAR + AR(1)/EWMA math, mean-preserving property, and why the ERC stays closed-form |
| `jobs/generate_5x5.sbatch` | SLURM: generate the semi-synthetic denominator + disease counts per year, optionally the ground-truth ERC |
| `jobs/synthetic_pipeline.sbatch` | SLURM: Stage 1 synthetic Medicare raw inputs (`snakefile_synthetic.smk`) → Stage 2 dense feature store (`snakefile_health.smk`) |
| `.gitignore` | +1 line (`outputs/*`) |

**`climhealth-fm/`** is an untracked **nested git repository** (its own `.git/`), tracked by neither the outer repo (`git ls-files climhealth-fm/` is empty) nor ignored; it contributes nothing to `origin/main..HEAD` and is **intentionally excluded** from this PR.

## 10. How to run

```bash
# Regenerate the 5-exposure + 5-confounder semi-synthetic counts (writes counts parquet + manifest)
sbatch jobs/generate_5x5.sbatch

# End-to-end: Stage 1 synthetic Medicare raw inputs -> Stage 2 dense feature store
sbatch jobs/synthetic_pipeline.sbatch

# Recompute the closed-form ground-truth ERC (reads the manifest that generated the data)
python -m src.ground_truth_erc
# To match a specific model run, override the forecast window / grid on the CLI, e.g.:
python -m src.ground_truth_erc synthetic.erc.forecast_start=<d> synthetic.erc.forecast_len=<n>
```

## 11. Reviewer notes

- **`expected_rate_grid` is the single source of truth** (`src/synthetic_causal.py:227`). Both the generator (via `synthetic_health.generate_synthetic_data`) and the ground truth (`ground_truth_erc.py`) draw from it; there is no second rate implementation. Any DGP change belongs here and only here — confirm nothing re-derives the rate downstream.
- **`normalize=false` changes the represented estimand.** It amplifies the exposure slope by `1/((1-rho)(1-phi))`. Still exact and closed-form, but it is a different curve; check the config's `spacetime.normalize` matches the intended estimand.
- **`W` row-alignment depends on `zcta_data` row order.** `build_spatial_weights` builds `W` from `zcta_data` centroids; `eta`'s columns and `offset`'s entries must be in the same ZCTA order for the coupling and offset to align. Confirm the row order is stable between rate construction and offset application.
- **Floor is applied exactly once, after both couplings** (`:280`); `apply_spacetime_coupling` returns unfloored values by design. Slope-exactness holds only on cells with `g0 > 0.01` — the caveat the ERC slope tests mask for.
