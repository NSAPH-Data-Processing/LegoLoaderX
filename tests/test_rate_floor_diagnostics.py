"""Configurable rate floor (``synthetic.poisson_params.rate_floor``) and the lambda-distribution
diagnostic (``describe_rate_grid``). Fully offline: beta=0 and no confounders, so the rate builder
never touches the covar store.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.synthetic_causal import describe_rate_grid, expected_rate_grid  # noqa: E402

N_Z = 200


@pytest.fixture(scope="module")
def zcta_data():
    import pandas as pd
    rng = np.random.RandomState(0)
    return pd.DataFrame({
        "zcta": [f"{i:05d}" for i in range(N_Z)],
        "latitude": rng.uniform(25, 49, N_Z),
        "longitude": rng.uniform(-124, -67, N_Z),
        "population": rng.randint(500, 50000, N_Z).astype(float),
    })


def _cfg(rate_floor="OMIT"):
    # background swings (lat/lon effects) push some pre-floor cells well below 0.01, so the floor
    # is actually exercised -> min(grid) == floor.
    p = dict(base_rate=0.11, seasonal_amplitude=0.02, latitude_effect=0.2, longitude_effect=0.1,
             population_normalizer=0.0003)
    if rate_floor != "OMIT":
        p["rate_floor"] = rate_floor
    syn = dict(poisson_params=p, beta=0.0, exposure_var_group="x", exposure_var="x",
               exposure_shape="linear", confounders=[], interactions=False)
    return OmegaConf.create({"year": 2010, "synthetic": syn})


# --------------------------------------------------------------------------------------------------
# Configurable floor
# --------------------------------------------------------------------------------------------------
def test_default_floor_is_001(zcta_data):
    """Omitting rate_floor keeps the historical 0.01 clamp (backward compatible)."""
    g = expected_rate_grid(_cfg(), zcta_data)
    assert g.min() == pytest.approx(0.01)     # some cells are clamped, so the min sits exactly at the floor
    assert g.max() > 0.01                      # ... but not everything is floored


def test_rate_floor_config_raises_min(zcta_data):
    """A higher rate_floor lifts the clamp and leaves cells already above it untouched."""
    g_lo = expected_rate_grid(_cfg(rate_floor=0.01), zcta_data)
    g_hi = expected_rate_grid(_cfg(rate_floor=0.05), zcta_data)
    assert g_lo.min() == pytest.approx(0.01)
    assert g_hi.min() == pytest.approx(0.05)
    above = g_lo > 0.05                         # cells already above the higher floor
    assert np.allclose(g_lo[above], g_hi[above])
    # every cell in g_hi is at least the new floor, and >= its g_lo value (monotone in the floor)
    assert (g_hi >= 0.05 - 1e-12).all()
    assert (g_hi >= g_lo - 1e-12).all()


def test_floor_default_matches_explicit(zcta_data):
    """Omitting rate_floor == setting it to 0.01, byte-for-byte."""
    assert np.array_equal(expected_rate_grid(_cfg(), zcta_data),
                          expected_rate_grid(_cfg(rate_floor=0.01), zcta_data))


# --------------------------------------------------------------------------------------------------
# describe_rate_grid diagnostic
# --------------------------------------------------------------------------------------------------
def test_describe_rate_grid_floor_fraction_and_percentiles():
    grid = np.array([[0.01, 0.01, 0.50], [0.01, 0.20, 0.30]])   # 3 of 6 cells at the floor
    s = describe_rate_grid(grid, floor=0.01)
    assert s["n_cells"] == 6
    assert s["frac_at_floor"] == pytest.approx(3 / 6)
    assert s["floor"] == pytest.approx(0.01)
    assert s["percentiles"][50] == pytest.approx(np.median(grid))
    assert s["percentiles"][100] == pytest.approx(0.50)
    assert s["mean"] == pytest.approx(grid.mean())


def test_describe_rate_grid_with_offset_sparsity():
    grid = np.full((4, 5), 0.1)
    offset = np.full(5, 10.0)                                    # expected count = 0.1*10 = 1.0 everywhere
    s = describe_rate_grid(grid, offset=offset, floor=0.01)
    assert s["expected_count"]["median"] == pytest.approx(1.0)
    assert s["implied_sparsity_pct"] == pytest.approx(100 * np.exp(-1.0), rel=1e-9)
    assert s["frac_at_floor"] == pytest.approx(0.0)             # 0.1 is above the 0.01 floor


def test_describe_rate_grid_no_floor_key_when_floor_none():
    s = describe_rate_grid(np.full((3, 3), 0.2))               # floor omitted
    assert "frac_at_floor" not in s and "expected_count" not in s
    assert s["n_cells"] == 9
