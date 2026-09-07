"""Rolling-forecast estimand helpers (``src/synthetic_causal``): window multiplicities, the
intervened/carried split, and the multiplicative-shift intervention.

Fully offline — the weight helpers are pure arithmetic, and the shift test builds a tiny fake
covar store on a tmp path so nothing touches ``data/covars``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.synthetic_causal import (  # noqa: E402
    build_extra_rate,
    forecast_window_weights,
    windowed_outcome,
)


# --------------------------------------------------------------------------- weights

def test_interior_day_is_counted_window_times_forecast():
    """An interior day lands in every (tau, k) pair, not just every lead — the naive
    "multiply by T_fc" correction is short by a factor of `window`."""
    w_all, _ = forecast_window_weights(365, window=3, n_forecast=5)
    assert w_all.max() == 3 * 5
    assert w_all[50] == 15  # well inside the year


def test_intervened_fraction_is_the_triangular_count():
    """Only (tau, k) with tau + k <= window-1 target a day whose own exposure was scaled."""
    W, TF, n_days = 3, 5, 365
    w_all, w_int = forecast_window_weights(n_days, W, TF)
    expected_pairs = sum(1 for tau in range(W) for k in range(TF) if tau + k <= W - 1)
    assert expected_pairs == 6
    assert w_int.max() == expected_pairs
    assert w_int.sum() / w_all.sum() == pytest.approx(expected_pairs / (W * TF))
    assert w_int.sum() / w_all.sum() == pytest.approx(0.4)


@pytest.mark.parametrize("W,TF", [(1, 1), (1, 5), (3, 5), (7, 14), (5, 5)])
def test_totals_match_direct_enumeration(W, TF):
    n_days = 365
    w_all, w_int = forecast_window_weights(n_days, W, TF)
    n_t0 = n_days - (W + TF - 2)
    assert w_all.sum() == pytest.approx(n_t0 * W * TF)
    assert w_int.sum() == pytest.approx(
        n_t0 * sum(1 for tau in range(W) for k in range(TF) if tau + k <= W - 1)
    )
    # w_int can never exceed w_all, day by day
    assert np.all(w_int <= w_all)


def test_window_equals_forecast_one_is_the_plain_daily_sum():
    """window=1, n_forecast=1 degenerates to "each day once, and each day intervened"."""
    w_all, w_int = forecast_window_weights(10, window=1, n_forecast=1)
    assert w_all.tolist() == [1.0] * 10
    assert w_int.tolist() == w_all.tolist()


def test_drop_incomplete_uses_every_start_day_whose_window_fits():
    """The last kept t0 must land exactly on the final day — no window may spill past the year,
    and none that fits may be silently dropped."""
    n_days, W, TF = 20, 3, 5
    w_all, _ = forecast_window_weights(n_days, W, TF, drop_incomplete=True)
    assert w_all[-1] > 0.0                       # last day IS reachable, by t0 = n_days-1-max_lag
    assert int(np.flatnonzero(w_all)[-1]) == n_days - 1
    assert w_all.sum() == pytest.approx((n_days - (W + TF - 2)) * W * TF)


def test_edges_ramp_and_never_exceed_the_interior():
    w_all, _ = forecast_window_weights(365, window=3, n_forecast=5)
    head = w_all[:7]
    assert np.all(np.diff(head) >= 0)            # ramps up
    assert head[0] == 1.0                        # only (t0=0, tau=0, k=0) hits day 0
    assert w_all.max() == 15


# --------------------------------------------------------------------------- collapse

def _grids(n_days=12, n_z=4):
    rng = np.random.default_rng(0)
    return rng.random((n_days, n_z)) + 0.5, rng.random((n_days, n_z)) + 0.5


def test_global_scope_ignores_the_factual_grid():
    lam_i, lam_f = _grids()
    off = np.array([1.0, 2.0, 3.0, 4.0])
    w_all, w_int = forecast_window_weights(12, 3, 2)
    got = windowed_outcome(lam_i, None, off, w_all, w_int, "mean")
    want = float(((w_all @ lam_i) * off).mean())
    assert got == pytest.approx(want)


def test_window_scope_blends_intervened_and_carried_days():
    lam_i, lam_f = _grids()
    off = np.array([1.0, 2.0, 3.0, 4.0])
    w_all, w_int = forecast_window_weights(12, 3, 2)
    got = windowed_outcome(lam_i, lam_f, off, w_all, w_int, "mean")
    want = float(((w_int @ lam_i + (w_all - w_int) @ lam_f) * off).mean())
    assert got == pytest.approx(want)


def test_factual_intervention_is_a_no_op_under_both_scopes():
    """delta=1 must return the factual value however the scope is set — G(1) = factual."""
    lam, _ = _grids()
    off = np.array([1.0, 2.0, 3.0, 4.0])
    w_all, w_int = forecast_window_weights(12, 3, 2)
    ref = windowed_outcome(lam, None, off, w_all, w_all, "mean")
    assert windowed_outcome(lam, lam, off, w_all, w_int, "mean") == pytest.approx(ref)


def test_sum_aggregation_is_mean_times_n_nodes():
    lam, fac = _grids()
    off = np.array([1.0, 2.0, 3.0, 4.0])
    w_all, w_int = forecast_window_weights(12, 3, 2)
    m = windowed_outcome(lam, fac, off, w_all, w_int, "mean")
    s = windowed_outcome(lam, fac, off, w_all, w_int, "sum")
    assert s == pytest.approx(m * len(off))


# --------------------------------------------------------------------------- shift intervention

N_Z, YEAR, N_DAYS = 40, 2011, 365


@pytest.fixture
def store(tmp_path):
    """A minimal covar store holding one daily exposure, plus the cfg that reads it."""
    zctas = [f"{i:05d}" for i in range(N_Z)]
    pd.DataFrame({"zcta": zctas}).to_parquet(tmp_path / "idx2zcta.parquet")
    rng = np.random.default_rng(7)
    x = (rng.random((N_DAYS, N_Z)) * 8 + 4).astype(np.float32)
    d = tmp_path / "expo" / "PM25"
    d.mkdir(parents=True)
    np.save(d / f"PM25__{YEAR}.npy", x)

    cfg = OmegaConf.create({
        "year": YEAR,
        "synthetic": {
            "beta": 0.01,
            "exposure_covars_root": str(tmp_path),
            "exposure_var_group": "expo",
            "exposure_var": "PM25",
            "exposure_shape": "linear",
            "confounders": [],
            "interactions": False,
        },
    })
    return cfg, pd.DataFrame({"zcta": zctas}), x


def test_delta_one_reproduces_the_factual_rate_term(store):
    cfg, zcta_data, _ = store
    factual = build_extra_rate(cfg, zcta_data)
    shifted = build_extra_rate(cfg, zcta_data, exposure_scale=1.0)
    np.testing.assert_allclose(factual, shifted, rtol=0, atol=0)


def test_raw_shift_scales_the_exposure_term_linearly(store):
    cfg, zcta_data, x = store
    beta = float(cfg.synthetic.beta)
    for delta in (0.5, 1.25, 2.0):
        got = build_extra_rate(cfg, zcta_data, exposure_scale=delta)
        np.testing.assert_allclose(got, beta * delta * x, rtol=1e-6)


def test_standardized_shift_fans_out_about_the_center_not_the_origin(store):
    """delta on the loader's normalized channel means X -> m + delta*(X-m), which is a
    materially different (and possibly opposite-signed) intervention from X -> delta*X."""
    cfg, zcta_data, x = store
    beta, delta, m = float(cfg.synthetic.beta), 1.5, float(x.mean())
    got = build_extra_rate(cfg, zcta_data, exposure_scale=delta, scale_center=m)
    np.testing.assert_allclose(got, beta * (m + delta * (x - m)), rtol=1e-6)
    raw = build_extra_rate(cfg, zcta_data, exposure_scale=delta)
    assert not np.allclose(got, raw)


def test_override_and_scale_together_is_rejected(store):
    cfg, zcta_data, _ = store
    with pytest.raises(ValueError, match="not both"):
        build_extra_rate(cfg, zcta_data, exposure_override=8.0, exposure_scale=1.5)


def test_scaling_never_mutates_the_module_level_covar_cache(store):
    """The shift builds a new array; a second factual call must still see the raw values."""
    cfg, zcta_data, x = store
    beta = float(cfg.synthetic.beta)
    build_extra_rate(cfg, zcta_data, exposure_scale=3.0)
    np.testing.assert_allclose(build_extra_rate(cfg, zcta_data), beta * x, rtol=1e-6)
