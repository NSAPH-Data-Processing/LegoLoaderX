"""Validate the dense mmaps against their source and idx2zcta order.

Covers covariates (float32 / NaN-missing), outcomes (int16 / 0-missing) and the
denominator (float32 / NaN-missing): shape & dtype, value parity on a few cells,
and the per-store missing-value rule. Uses small real slices.
"""

from __future__ import annotations

import glob
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
IDX = ROOT / "data/covars/idx2zcta.parquet"
TMMX = ROOT / "data/covars/gridmet/tmmx/tmmx__2015.npy"
SRC = (
    ROOT / "data/input/lego/environmental/meteorology__gridmet/core/zcta_daily/"
    "meteorology__gridmet__core__zcta_daily__2015.parquet"
)

# health outcome + denominator (synthetic): a single (var, year) we know exists
OUT_VAR, OUT_YEAR = "diabetes", 2010
OUT = ROOT / f"data/health/ccw/{OUT_VAR}/{OUT_VAR}__{OUT_YEAR}.npy"
OUT_SRC = str(
    ROOT / f"data/input/lego/medicare_synthetic/medpar_outcomes/ccw/zcta_daily/"
    f"sparse_counts_{OUT_VAR}_{OUT_YEAR}.parquet"
)
DENOM = ROOT / f"data/health/denom/denom__{OUT_YEAR}.npy"
DENOM_SRC = str(
    ROOT / f"data/input/lego/medicare_synthetic/mbsf_medpar_denom/zcta_yearly/"
    f"counts_{OUT_YEAR}.parquet"
)


def _need(p: Path):
    if not p.exists():
        pytest.skip(f"{p} missing; build it first (snakemake -s snakefile.smk)")


@pytest.fixture
def idx2zcta():
    _need(IDX)
    return pd.read_parquet(IDX)["zcta"].tolist()


@pytest.fixture
def health_idx2zcta():
    p = ROOT / "data/health/idx2zcta.parquet"
    _need(p)
    return pd.read_parquet(p)["zcta"].tolist()


def test_covars_shape_dtype(idx2zcta):
    _need(TMMX)
    a = np.load(TMMX)
    assert a.dtype == np.float32
    assert a.shape == (365, len(idx2zcta))  # 2015 is not a leap year


def test_covars_value_parity(idx2zcta):
    _need(TMMX)
    _need(SRC)
    a = np.load(TMMX)
    z2i = {z: i for i, z in enumerate(idx2zcta)}
    df = duckdb.execute(f"""
        SELECT zcta, date, tmmx FROM read_parquet('{SRC}')
        WHERE date IN (DATE '2015-01-01', DATE '2015-07-15', DATE '2015-12-31')
    """).df()
    df = df[df["zcta"].isin(z2i)].head(50)
    assert len(df) > 0
    for _, r in df.iterrows():
        doy = (pd.Timestamp(r["date"]) - pd.Timestamp(2015, 1, 1)).days
        assert np.isclose(a[doy, z2i[r["zcta"]]], np.float32(r["tmmx"]), equal_nan=True)


def test_covars_missing_is_nan(idx2zcta):
    _need(TMMX)
    _need(SRC)
    a = np.load(TMMX)
    z2i = {z: i for i, z in enumerate(idx2zcta)}
    present = set(duckdb.execute(f"SELECT DISTINCT zcta FROM read_parquet('{SRC}')").df()["zcta"])
    missing = [z for z in idx2zcta if z not in present]
    if not missing:
        pytest.skip("no idx2zcta zcta is absent from the 2015 source")
    assert np.isnan(a[:, z2i[missing[0]]]).all()


def test_outcomes_shape_dtype(health_idx2zcta):
    _need(OUT)
    a = np.load(OUT)
    assert a.dtype == np.int16
    assert a.shape == (365, len(health_idx2zcta))  # 2010 is not a leap year


def test_outcomes_value_parity(health_idx2zcta):
    _need(OUT)
    if not glob.glob(OUT_SRC):
        pytest.skip(f"{OUT_SRC} missing")
    a = np.load(OUT)
    z2i = {z: i for i, z in enumerate(health_idx2zcta)}
    df = duckdb.execute(f"""
        SELECT zcta, date, n FROM '{OUT_SRC}'
        WHERE n > 0 ORDER BY date LIMIT 50
    """).df()
    df = df[df["zcta"].isin(z2i)]
    assert len(df) > 0
    for _, r in df.iterrows():
        doy = (pd.Timestamp(r["date"]) - pd.Timestamp(OUT_YEAR, 1, 1)).days
        assert a[doy, z2i[r["zcta"]]] == np.int16(r["n"])


def test_outcomes_missing_is_zero(health_idx2zcta):
    _need(OUT)
    if not glob.glob(OUT_SRC):
        pytest.skip(f"{OUT_SRC} missing")
    a = np.load(OUT)
    z2i = {z: i for i, z in enumerate(health_idx2zcta)}
    present = set(duckdb.execute(f"SELECT DISTINCT zcta FROM '{OUT_SRC}'").df()["zcta"])
    missing = [z for z in health_idx2zcta if z not in present]
    if not missing:
        pytest.skip("no idx2zcta zcta is absent from the source")
    assert (a[:, z2i[missing[0]]] == 0).all()


def test_denom_shape_dtype(health_idx2zcta):
    _need(DENOM)
    a = np.load(DENOM)
    assert a.dtype == np.float32
    assert a.shape == (len(health_idx2zcta),)


def test_denom_value_parity(health_idx2zcta):
    _need(DENOM)
    _need(Path(DENOM_SRC))
    a = np.load(DENOM)
    z2i = {z: i for i, z in enumerate(health_idx2zcta)}
    df = duckdb.execute(f"SELECT zcta, n_bene FROM '{DENOM_SRC}' LIMIT 50").df()
    df = df[df["zcta"].isin(z2i)]
    assert len(df) > 0
    for _, r in df.iterrows():
        assert np.isclose(a[z2i[r["zcta"]]], np.float32(r["n_bene"]))


def test_denom_missing_is_nan(health_idx2zcta):
    _need(DENOM)
    _need(Path(DENOM_SRC))
    a = np.load(DENOM)
    z2i = {z: i for i, z in enumerate(health_idx2zcta)}
    present = set(duckdb.execute(f"SELECT DISTINCT zcta FROM '{DENOM_SRC}'").df()["zcta"])
    missing = [z for z in health_idx2zcta if z not in present]
    if not missing:
        pytest.skip("no idx2zcta zcta is absent from the denom source")
    assert np.isnan(a[z2i[missing[0]]])
