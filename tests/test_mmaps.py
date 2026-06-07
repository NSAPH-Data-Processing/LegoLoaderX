"""Validate the dense covariate mmaps against the source and idx2zcta order.

Shape/dtype, value parity on a few (zcta, day) cells, and the missing-value rule
(a ZCTA absent from the source year is all-NaN). Uses a small real slice.
"""

from __future__ import annotations

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


def _need(p: Path):
    if not p.exists():
        pytest.skip(f"{p} missing; build it first (snakemake -s snakefile.smk)")


@pytest.fixture
def idx2zcta():
    _need(IDX)
    return pd.read_parquet(IDX)["zcta"].tolist()


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
