"""Synthetic exposure generator (``src/synthetic_exposure.py``).

Fully offline: ``build_one_exposure`` takes the standardized driver arrays as an argument, so these
tests fabricate in-memory drivers + a fake ZCTA grid and never touch the covar store. The
``_save_as_covar`` round-trip test writes to a tmp dir with its own ``idx2zcta.parquet``.
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

from src.synthetic_causal import _load_covar_aligned  # noqa: E402
from src.synthetic_exposure import (  # noqa: E402
    _r2_on_drivers,
    _save_as_covar,
    build_one_exposure,
)

N_Z = 300
YEAR = 2011          # non-leap -> 365 days
N_DAYS = 365


@pytest.fixture(scope="module")
def zcta_data():
    rng = np.random.RandomState(0)
    return pd.DataFrame({
        "zcta": [f"{i:05d}" for i in range(N_Z)],
        "latitude": rng.uniform(25, 49, N_Z),
        "longitude": rng.uniform(-124, -67, N_Z),
        "population": rng.randint(500, 50000, N_Z).astype(float),
    })


@pytest.fixture(scope="module")
def drivers():
    """Three standardized, time-constant (yearly-like) drivers, each broadcast to (N_DAYS, N_Z)."""
    rng = np.random.RandomState(1)
    out = {}
    for name in ("d1", "d2", "d3"):
        v = rng.standard_normal(N_Z)
        v = (v - v.mean()) / v.std()
        out[name] = np.repeat(v[None, :], N_DAYS, axis=0).astype(np.float64)
    return out


def _cfg(method="knn", k=8):
    return OmegaConf.create(
        {"year": YEAR, "synthetic": {"exposure_generation": {"method": method, "k": k,
                                                             "length_scale_km": None}}}
    )


def _spec(**over):
    s = dict(name="PM25", var_group="synth_exposure", target_mean=10.0, target_sd=5.0,
             nonnegative=True, f_drivers=0.6, f_struct=0.2, rho=0.7, phi=0.9,
             weights={"d1": 1.0, "d2": -0.5})
    s.update(over)
    return s


def test_shape_dtype(zcta_data, drivers):
    e = build_one_exposure(_cfg(), zcta_data, YEAR, _spec(), drivers, base_seed=123, e_idx=0)
    assert e.shape == (N_DAYS, N_Z)
    assert e.dtype == np.float32


def test_r2_matches_f_drivers(zcta_data, drivers):
    """OLS(exposure ~ all drivers) R^2 ~= f_drivers by construction (independent struct/noise)."""
    for f in (0.4, 0.6):
        e = build_one_exposure(_cfg(), zcta_data, YEAR, _spec(f_drivers=f, f_struct=0.2, nonnegative=False),
                               drivers, base_seed=7, e_idx=0)
        r2 = _r2_on_drivers(e, drivers)
        assert abs(r2 - f) < 0.03, f"R^2={r2} not near f_drivers={f}"


def test_target_mean_sd(zcta_data, drivers):
    """Without clipping, realized mean/sd match the targets."""
    e = build_one_exposure(_cfg(), zcta_data, YEAR, _spec(target_mean=3.0, target_sd=2.0, nonnegative=False),
                           drivers, base_seed=1, e_idx=0)
    assert np.nanmean(e) == pytest.approx(3.0, abs=0.05)
    assert np.nanstd(e) == pytest.approx(2.0, abs=0.05)


def test_nonnegative_clip(zcta_data, drivers):
    e = build_one_exposure(_cfg(), zcta_data, YEAR, _spec(target_mean=2.0, target_sd=5.0, nonnegative=True),
                           drivers, base_seed=1, e_idx=0)
    assert np.nanmin(e) >= 0.0


def test_reproducible_same_seed(zcta_data, drivers):
    a = build_one_exposure(_cfg(), zcta_data, YEAR, _spec(), drivers, base_seed=42, e_idx=2)
    b = build_one_exposure(_cfg(), zcta_data, YEAR, _spec(), drivers, base_seed=42, e_idx=2)
    assert np.array_equal(a, b)


def test_distinct_seed_per_exposure(zcta_data, drivers):
    """Different e_idx -> different structure/noise draw -> different arrays (not collinear)."""
    a = build_one_exposure(_cfg(), zcta_data, YEAR, _spec(), drivers, base_seed=42, e_idx=0)
    b = build_one_exposure(_cfg(), zcta_data, YEAR, _spec(), drivers, base_seed=42, e_idx=1)
    assert not np.array_equal(a, b)


def test_spatial_structure_present(zcta_data, drivers):
    """rho>0 induces positive neighbour correlation beyond the (spatially-smooth) driver signal."""
    from src.synthetic_spacetime import get_W
    e = build_one_exposure(_cfg(), zcta_data, YEAR, _spec(f_drivers=0.0, f_struct=0.9, rho=0.8, phi=0.0,
                                                          nonnegative=False, weights={}),
                           drivers, base_seed=3, e_idx=0)
    W = get_W(zcta_data, method="knn", k=8)
    day = e[10].astype(np.float64)
    neigh = W @ day
    assert np.corrcoef(day, neigh)[0, 1] > 0.2


def test_temporal_structure_present(zcta_data, drivers):
    """phi>0 induces positive lag-1 autocorrelation; phi=0 (all iid noise) gives ~0."""
    e = build_one_exposure(_cfg(), zcta_data, YEAR, _spec(f_drivers=0.0, f_struct=0.8, rho=0.0, phi=0.9,
                                                          nonnegative=False, weights={}),
                           drivers, base_seed=3, e_idx=0)
    x0, x1 = e[:-1].astype(np.float64), e[1:].astype(np.float64)
    ac = np.nanmean(((x0 - x0.mean(0)) * (x1 - x1.mean(0))).mean(0) / (x0.std(0) * x1.std(0)))
    assert ac > 0.3

    e_iid = build_one_exposure(_cfg(), zcta_data, YEAR, _spec(f_drivers=0.0, f_struct=0.0, rho=0.0, phi=0.0,
                                                              nonnegative=False, weights={}),
                               drivers, base_seed=3, e_idx=0)
    x0, x1 = e_iid[:-1].astype(np.float64), e_iid[1:].astype(np.float64)
    ac0 = np.nanmean(((x0 - x0.mean(0)) * (x1 - x1.mean(0))).mean(0) / (x0.std(0) * x1.std(0)))
    assert abs(ac0) < 0.1


def test_f_noise_guard(zcta_data, drivers):
    """f_drivers + f_struct >= 1 leaves no independent noise -> ValueError (positivity guard)."""
    with pytest.raises(ValueError):
        build_one_exposure(_cfg(), zcta_data, YEAR, _spec(f_drivers=0.7, f_struct=0.35),
                           drivers, base_seed=1, e_idx=0)


def test_save_as_covar_roundtrip(zcta_data, drivers, tmp_path):
    """Saving to covars order then reading via _load_covar_aligned is identity on mainland cells;
    ZCTAs absent from the generator grid are NaN in the raw file."""
    # covars grid = the generator's zctas plus 20 extras that the generator never produces
    extra = [f"{i:05d}" for i in range(N_Z, N_Z + 20)]
    covars_zcta = list(zcta_data["zcta"]) + extra
    root = tmp_path / "covars"
    root.mkdir()
    pd.DataFrame({"zcta": covars_zcta}).to_parquet(root / "idx2zcta.parquet", index=False)

    e = build_one_exposure(_cfg(), zcta_data, YEAR, _spec(nonnegative=False), drivers, base_seed=5, e_idx=0)
    fpath, n_missing = _save_as_covar(e, zcta_data, str(root), "synth_exposure", "PM25", YEAR)
    assert n_missing == 0                                       # all generator zctas are in the covars grid

    raw = np.load(fpath)
    assert raw.shape == (N_DAYS, len(covars_zcta))
    assert np.isnan(raw[:, N_Z:]).all()                        # the 20 extra columns are all NaN
    assert np.isfinite(raw[:, :N_Z]).all()

    back = _load_covar_aligned(str(root), "synth_exposure", "PM25", YEAR,
                               zcta_data["zcta"], N_DAYS, temporal_res="daily", standardize=False)
    assert np.allclose(back, e, atol=1e-5)                     # round-trip identity on mainland cells
