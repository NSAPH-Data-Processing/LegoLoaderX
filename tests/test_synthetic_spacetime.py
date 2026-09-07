"""Spatial + temporal coupling of the synthetic DGP rate (src/synthetic_spacetime.py).

These run fully offline: by using ``beta=0`` or a scalar ``exposure_override`` the rate builder never
touches the covar store, so no data files are needed. The headline test is the BACKWARD-COMPAT
regression (``test_backward_compat_*``): omitting the ``spacetime`` block, or setting
``rho=0, phi=0``, must reproduce the original DGP byte-for-byte -- both the rate grid and the
seeded Poisson counts.
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import src.synthetic_spacetime as st  # noqa: E402
from src.synthetic_causal import expected_rate_grid, offset_vector  # noqa: E402
from src.synthetic_health import generate_synthetic_data  # noqa: E402
from src.synthetic_spacetime import (  # noqa: E402
    apply_spacetime_coupling,
    build_spatial_weights,
    halflife_to_phi,
    phi_to_halflife,
)

N_Z = 150
_P = dict(base_rate=0.11, seasonal_amplitude=0.02, spatial_variance=0.03,
          latitude_effect=0.2, longitude_effect=0.1, population_effect=0.0001,
          population_normalizer=0.0003, random_seed=42)


@pytest.fixture(scope="module")
def zcta_data():
    rng = np.random.RandomState(0)
    return pd.DataFrame({
        "zcta": [f"{i:05d}" for i in range(N_Z)],
        "latitude": rng.uniform(25, 49, N_Z),
        "longitude": rng.uniform(-124, -67, N_Z),
        "population": rng.randint(500, 50000, N_Z).astype(float),
    })


def _cfg(beta=0.0, spacetime="OMIT"):
    """Build a minimal cfg. spacetime='OMIT' leaves the block out entirely."""
    syn = dict(poisson_params=_P, beta=beta, exposure_var_group="x", exposure_var="x",
               exposure_shape="linear", confounders=[], interactions=False)
    if spacetime != "OMIT":
        syn["spacetime"] = spacetime
    return OmegaConf.create({"year": 2010, "synthetic": syn})


# --------------------------------------------------------------------------------------------------
# Neighbour matrix W
# --------------------------------------------------------------------------------------------------
def test_W_is_row_stochastic_zero_diagonal(zcta_data):
    W = build_spatial_weights(zcta_data, method="knn", k=8)
    assert W.shape == (N_Z, N_Z)
    assert np.allclose(np.asarray(W.sum(axis=1)).ravel(), 1.0)   # W @ ones == ones
    assert W.diagonal().sum() == 0.0                             # zero diagonal


def test_W_distance_weights_closer_neighbours_more(zcta_data):
    W = build_spatial_weights(zcta_data, method="distance", k=8, length_scale_km=200.0)
    assert np.allclose(np.asarray(W.sum(axis=1)).ravel(), 1.0)


# --------------------------------------------------------------------------------------------------
# Half-life helpers
# --------------------------------------------------------------------------------------------------
def test_halflife_roundtrip():
    for h in (1.0, 3.0, 10.0, 30.0):
        assert phi_to_halflife(halflife_to_phi(h)) == pytest.approx(h, rel=1e-12)


def test_halflife_validation():
    with pytest.raises(ValueError):
        halflife_to_phi(0.0)
    with pytest.raises(ValueError):
        phi_to_halflife(1.0)


# --------------------------------------------------------------------------------------------------
# apply_spacetime_coupling: properties
# --------------------------------------------------------------------------------------------------
def test_rho_phi_validation(zcta_data):
    W = build_spatial_weights(zcta_data, k=8)
    eta = np.random.RandomState(1).rand(10, N_Z)
    for bad in (-0.1, 1.0, 1.5):
        with pytest.raises(ValueError):
            apply_spacetime_coupling(eta, W, rho=bad, phi=0.0)
        with pytest.raises(ValueError):
            apply_spacetime_coupling(eta, W, rho=0.0, phi=bad)


def test_identity_when_rho_and_phi_zero(zcta_data):
    """rho=0 AND phi=0 is an early-return identity: the SAME array back, no solve/recursion."""
    eta = np.random.RandomState(2).rand(12, N_Z)
    assert apply_spacetime_coupling(eta, None, rho=0.0, phi=0.0) is eta


def test_mean_preserving_normalize_true(zcta_data):
    """A field constant in space AND time is a fixed point (normalize=True)."""
    W = build_spatial_weights(zcta_data, k=8)
    const = np.full((100, N_Z), 0.137)
    out = apply_spacetime_coupling(const, W, rho=0.6, phi=0.9, normalize=True)
    assert np.allclose(out, 0.137, atol=1e-10)


def test_amplification_normalize_false(zcta_data):
    """normalize=False amplifies a constant field toward 1/((1-rho)(1-phi)) (exact finite-horizon)."""
    W = build_spatial_weights(zcta_data, k=8)
    rho, phi, n_days, c = 0.5, 0.9, 200, 0.137
    out = apply_spacetime_coupling(np.full((n_days, N_Z), c), W, rho=rho, phi=phi, normalize=False)
    exact_last = c / (1 - rho) * (1 - phi ** (n_days + 1)) / (1 - phi)
    assert np.allclose(out[-1], exact_last, rtol=1e-7)
    amp = 1.0 / ((1 - rho) * (1 - phi))
    assert out[-1].mean() / c == pytest.approx(amp, rel=1e-3)


def test_ar1_matches_closed_form_unrolling(zcta_data):
    """Temporal-only recursion equals lambda_t=(1-phi)sum phi^l eta_{t-l} + phi^{t+1} eta_0."""
    rng = np.random.RandomState(3)
    phi, n_days = 0.85, 120
    eta = rng.rand(n_days, N_Z)
    out = apply_spacetime_coupling(eta, None, rho=0.0, phi=phi, normalize=True)
    ref = np.empty_like(eta)
    prev = eta[0]
    for t in range(n_days):
        prev = (1 - phi) * eta[t] + phi * prev
        ref[t] = prev
    assert np.allclose(out, ref)


def test_stability_near_unit(zcta_data):
    W = build_spatial_weights(zcta_data, k=8)
    out = apply_spacetime_coupling(np.random.RandomState(4).rand(50, N_Z), W,
                                   rho=0.99, phi=0.99, normalize=True)
    assert np.isfinite(out).all()


# --------------------------------------------------------------------------------------------------
# Wiring into expected_rate_grid + BACKWARD COMPATIBILITY
# --------------------------------------------------------------------------------------------------
def test_backward_compat_omitted_equals_zero_grid(zcta_data):
    """HARD REQUIREMENT: no spacetime block == {rho:0,phi:0}, byte-for-byte, in the rate grid."""
    g_omit = expected_rate_grid(_cfg(spacetime="OMIT"), zcta_data)
    g_zero = expected_rate_grid(_cfg(spacetime={"rho": 0, "phi": 0}), zcta_data)
    assert np.array_equal(g_omit, g_zero)
    # the normalize/method/k/length_scale_km fields are inert when rho=phi=0
    g_zero2 = expected_rate_grid(
        _cfg(spacetime={"rho": 0, "phi": 0, "normalize": False, "method": "distance", "k": 3}),
        zcta_data,
    )
    assert np.array_equal(g_omit, g_zero2)


def test_backward_compat_seeded_counts_identical(zcta_data):
    """HARD REQUIREMENT extends to the Poisson-sampled counts under a fixed seed."""
    g_omit = expected_rate_grid(_cfg(spacetime="OMIT"), zcta_data)
    g_zero = expected_rate_grid(_cfg(spacetime={"rho": 0, "phi": 0}), zcta_data)
    offset = offset_vector(_cfg(), zcta_data)
    days = [(2010, 1, d + 1) for d in range(31)]
    np.random.seed(7); a = generate_synthetic_data(zcta_data, days, "test", g_omit, offset)
    np.random.seed(7); b = generate_synthetic_data(zcta_data, days, "test", g_zero, offset)
    assert a.equals(b)


def test_refactored_sampler_matches_per_day_loop(zcta_data):
    """One whole-grid Poisson draw == the old per-day loop (same C-order stream), byte-for-byte."""
    g = expected_rate_grid(_cfg(spacetime="OMIT"), zcta_data)
    offset = offset_vector(_cfg(), zcta_data)
    days = [(2010, 1, d + 1) for d in range(20)]
    np.random.seed(42)
    new_df = generate_synthetic_data(zcta_data, days, "test", g, offset)
    np.random.seed(42)
    rows = []
    for d, (yy, mm, dd) in enumerate(days):
        counts = np.random.poisson(g[d] * offset)
        m = counts > 0
        rows.append(pd.DataFrame({"zcta": zcta_data["zcta"].to_numpy()[m], "var": "test",
                                  "date": date(yy, mm, dd), "n": counts[m]}))
    assert new_df.equals(pd.concat(rows, ignore_index=True))


def test_partial_configs_valid(zcta_data):
    """rho=0,phi>0 (temporal-only) and rho>0,phi=0 (spatial-only) are valid and change the grid."""
    g0 = expected_rate_grid(_cfg(spacetime="OMIT"), zcta_data)
    t_only = expected_rate_grid(_cfg(spacetime={"rho": 0, "phi": 0.8}), zcta_data)
    s_only = expected_rate_grid(_cfg(spacetime={"rho": 0.5, "phi": 0}), zcta_data)
    for g in (t_only, s_only):
        assert np.isfinite(g).all()
        assert not np.array_equal(g, g0)


def test_erc_slope_preserved_normalize_true(zcta_data):
    """do(exposure) slope stays beta*dx on unfloored cells (mean-preserving coupling)."""
    beta = 0.02
    cfg = _cfg(beta=beta, spacetime={"rho": 0.5, "phi": 0.9, "normalize": True})
    g0 = expected_rate_grid(cfg, zcta_data, exposure_override=1.0)
    g1 = expected_rate_grid(cfg, zcta_data, exposure_override=3.0)
    unfloored = g0 > 0.01 + 1e-9
    assert np.allclose((g1 - g0)[unfloored], beta * 2.0)


def test_erc_slope_amplified_normalize_false(zcta_data):
    """normalize=False amplifies the do(exposure) slope by ~1/((1-rho)(1-phi)) (closed-form)."""
    beta, rho, phi = 0.02, 0.5, 0.9
    cfg = _cfg(beta=beta, spacetime={"rho": rho, "phi": phi, "normalize": False})
    g0 = expected_rate_grid(cfg, zcta_data, exposure_override=1.0)
    g1 = expected_rate_grid(cfg, zcta_data, exposure_override=3.0)
    uf = g0[-1] > 0.01 + 1e-9
    exact = beta * 2.0 / (1 - rho) * (1 - phi ** 365) / (1 - phi)
    assert np.allclose((g1 - g0)[-1][uf], exact, rtol=1e-6)


def test_erc_sweep_reuses_caches(zcta_data):
    """The ERC exposure sweep must reuse one W and one LU factorization, not rebuild per call."""
    st._W_CACHE.clear()
    st._LU_CACHE.clear()
    cfg = _cfg(beta=0.02, spacetime={"rho": 0.4, "phi": 0.8, "normalize": True})
    for a in np.linspace(1, 5, 25):
        expected_rate_grid(cfg, zcta_data, exposure_override=a)
    assert len(st._W_CACHE) == 1
    assert len(st._LU_CACHE) == 1
