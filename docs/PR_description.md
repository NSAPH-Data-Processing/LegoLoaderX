# A semi-synthetic benchmark with a known exposure–response curve

**Read first:** [`docs/semi_synthetic_theory.pdf`](semi_synthetic_theory.pdf) (3 pages) — why the
construction is what it is. This file is the PR summary; [`PR_semi_synthetic_causal.md`](PR_semi_synthetic_causal.md)
is the exhaustive per-function appendix, which is *not* the place to start.

## What and why

We cannot tell whether the model recovers a causal exposure–response curve, because on real data
the true curve is exactly the unknown we are trying to estimate. This PR builds data where we chose
the answer: **real covariates** (real PM2.5, gridMET, census) with a **synthetic outcome** drawn
from a rate we write down. The model trains on precisely the numbers that produced its target, so
"did it recover β?" becomes a question with a checkable answer.

The per-capita Poisson rate is

```
λ = max(floor,  seasonal + geographic  +  β·PM2.5  +  Σ_k γ_k · standardized(C_k))
Y ~ Poisson(λ · population_normalizer · population)
```

`β = 0.01` per µg/m³ is the ground truth. The `γ_k` are 4 daily gridMET terms and 10 yearly census
terms. Everything else follows from this one expression.

## The three arms of confounding

Confounding needs the confounder to affect *both* the outcome and the exposure. A benchmark with
only the first arm is trivially solved by naive regression and discriminates nothing.

| arm | mechanism | known? |
|---|---|---|
| confounder → outcome | `γ_k` in the rate | ✅ set in config |
| exposure → outcome | `β` in the rate | ✅ set in config — the target |
| confounder → exposure | `synthetic.exposure_generation` | ✅ when generated; whatever the real data holds otherwise |

Both worlds ship, selected by one config switch (`synthetic.exposure_group` / `env_group`). **Real
exposures are the default** — more realistic, and the model reads `pm25_ushap`/`gridmet` under
exactly those names with no changes on its side. Generated exposures make `R²(exposure ~
confounders) ≈ f_drivers` true by construction, so the confounding strength is a dial rather than
an accident.

## The one genuinely non-obvious result

The model forecasts a *window*: from start day `t₀` it takes `W` input days and emits `H` leads off
each, so prediction `(t₀, τ, k)` targets calendar day `t₀ + τ + k`. Two things follow that are easy
to get wrong:

1. An interior day is summed **`W·H`** times, not `H`. Correcting by "×H because it predicts a
   window" is short by a factor of `W`.
2. The intervention touches only that sample's own `W` input days. Since the dose–response is
   same-day, every prediction with `τ + k ≥ W` targets a day whose exposure stayed factual and
   carries **exactly zero** causal contrast. At `W=H=30` only **52%** of the summed predictions can
   move; at `W=3, H=5` only **40%**.

A perfectly calibrated model scored against a globally-intervened ground truth therefore looks like
it recovered only that fraction. `src/ground_truth_gcomp.py` implements both readings
(`intervention_scope=window|global`) so the comparison can be made deliberately rather than by
accident.

## Exact vs. approximate

The ground truth is **not** Monte Carlo. The rate is deterministic given the covariates, and the
Poisson draw adds noise to realised counts but not to their mean — so we evaluate λ under the
intervention directly, floor included. Verified numerically: the level and shift slopes match their
closed forms to **0.3%**, and the residual is exactly the ~1% of cells pinned at `rate_floor`.

The two curves are also cross-checked against each other. Their slope ratio must equal the
offset-weighted mean exposure, since a δ-shift is a level-shift scaled by the exposure's own
magnitude — they agree to **0.08%** (8.856 vs 8.863 µg/m³).

## What to review, in order

1. **`expected_rate_grid` is the single source of truth** (`src/synthetic_causal.py`). The generator
   and the ground truth both draw from it; there is no second rate implementation. Confirm nothing
   downstream re-derives the rate.
2. **Are the injected coefficients the ones we want?** `β = 0.01/µg/m³` and the 14 live `γ_k` in
   `conf/synthetic/config.yaml` are the entire causal content of the benchmark.
3. **`f_noise = 1 − f_drivers − f_struct` is the positivity knob** (generated-exposure mode only).
   It is the exposure variation remaining *conditional on* the confounders; driving it to zero makes
   the curve unidentifiable at any sample size. The generator raises rather than allow it.
4. **The manifest, not the config, is the answer key.** Every dataset gets a resolved-parameter JSON
   beside its parquet, and the ground truth reads it back. This is load-bearing: per-disease
   `poisson_params` overrides mean the live config's defaults are wrong for most diseases (diabetes
   uses `base_rate=0.15`, not the config's `0.11`).
5. **Units are part of the estimand.** The loader serves covariates standardized, so multiplying
   that channel by δ implements `X → μ + δ(X−μ)`, not `X → δX`. These can differ in *sign*.
   `gcomp.shift_space` selects which one; it must match the evaluation.

## Known limitations (deliberate, not oversights)

- All 30 `climate_types` variables are **identically zero at source**, so the two climate `γ` terms
  are dead. Harmless for identification — there is no confounding to adjust away — but they should
  not be read as variables the model must adjust for.
- Poisson noise uses one seeded stream, so residuals are correlated across diseases and years.
- The outcome is additive and same-day, which rewards estimators matched to that structure.

## How to run

```bash
sbatch jobs/regenerate_outcomes.sbatch    # Stage 1: sparse counts + manifests
sbatch jobs/format_health.sbatch          # Stage 2: dense .npy the dataloader reads
PYTHONPATH=. python -m src.ground_truth_gcomp year=2013 synthetic.var_name=diabetes \
    synthetic.gcomp.window=30 synthetic.gcomp.n_forecast=30   # match the model run
```

Add `jobs/generate_exposures.sbatch` as Stage 0 only in generated-exposure mode. `window` /
`n_forecast` **must** match the model's `dataset.window` / `model.max_forecast_steps`.

Coverage: 27 diseases × 2011–2020. Tests: `tests/test_gcomp_windows.py` (19),
`test_synthetic_exposure.py` (10), `test_synthetic_spacetime.py`, `test_rate_floor_diagnostics.py` (6).
