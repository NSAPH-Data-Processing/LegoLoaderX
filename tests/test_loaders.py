"""End-to-end checks for the mmap dataset classes against the generated store.

Parity vs the raw `.npy` files, idx2zcta node gather + identity fast-path, the
summary-stats round-trip (compute → save → load → normalize), and the composed
HealthXDataset shapes. Skips if the store hasn't been built.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from legoloaderx import XDataset, HealthDataset, HealthXDataset

ROOT = Path(__file__).resolve().parent.parent
COVARS = ROOT / "data/covars"
HEALTH = ROOT / "data/health"


def _need(p: Path):
    if not p.exists():
        pytest.skip(f"{p} missing; build the store first")


def _covar_nodes(var="tmmx", vg="gridmet", year=2010, k=6):
    """A few continental nodes that actually have data for (vg, var, year)."""
    nodes = pd.read_parquet(COVARS / "idx2zcta.parquet")["zcta"].tolist()
    mm = np.load(COVARS / vg / var / f"{var}__{year}.npy", mmap_mode="r")
    good = [nodes[i] for i in range(len(nodes)) if not np.isnan(mm[:, i]).all()][:k]
    return nodes, good


def test_xdataset_parity_and_shape():
    _need(COVARS / "gridmet/tmmx/tmmx__2010.npy")
    nodes, good = _covar_nodes()
    vd = {"gridmet": {"vars": ["tmmx"], "temporal_res": "daily"}}
    ds = XDataset(str(COVARS), vd, good, window=5, summary_stats={}, min_year=2010, max_year=2010)
    x = ds[180].numpy()[:, 0, :]                       # (n_nodes, 5)
    assert x.shape == (len(good), 5)
    mm = np.load(COVARS / "gridmet/tmmx/tmmx__2010.npy", mmap_mode="r")
    cols = [nodes.index(z) for z in good]
    exp = np.asarray(mm[180:185, cols]).T             # window starts at lead idx 180
    assert np.allclose(x, exp, equal_nan=True)


def test_xdataset_identity_fastpath():
    _need(COVARS / "idx2zcta.parquet")
    nodes = pd.read_parquet(COVARS / "idx2zcta.parquet")["zcta"].tolist()
    vd = {"gridmet": {"vars": ["tmmx"], "temporal_res": "daily"}}
    ds = XDataset(str(COVARS), vd, nodes, window=2, summary_stats={}, min_year=2010, max_year=2010)
    assert ds._identity_rows
    assert ds[0].shape[0] == len(nodes)


def test_xdataset_normalization_roundtrip():
    _need(COVARS / "census/population/population__2010.npy")
    nodes = pd.read_parquet(COVARS / "idx2zcta.parquet")["zcta"].tolist()
    cm = np.load(COVARS / "census/population/population__2010.npy", mmap_mode="r")
    good = [nodes[i] for i in range(len(nodes)) if not np.isnan(cm[i])][:5]
    vd = {"census": {"vars": ["population"], "temporal_res": "yearly"}}
    stats = XDataset.compute_summary(str(COVARS), vd, 2010, 2010)
    ds = XDataset(str(COVARS), vd, good, window=1, summary_stats=stats, min_year=2010, max_year=2010)
    raw = np.asarray(cm[[nodes.index(z) for z in good]])
    m, s = stats["census"]["population"]["mean"], stats["census"]["population"]["std"]
    assert np.allclose(ds[0][:, 0, 0].numpy(), (raw - m) / s, equal_nan=True)


def test_healthdataset_parity_and_denom():
    _need(HEALTH / "ccw/diabetes/diabetes__2010.npy")
    _need(HEALTH / "denom/denom__2010.npy")
    nodes = pd.read_parquet(HEALTH / "idx2zcta.parquet")["zcta"].tolist()
    dm = np.load(HEALTH / "denom/denom__2010.npy", mmap_mode="r")
    good = [nodes[i] for i in range(len(nodes)) if not np.isnan(dm[i])][:6]
    vd = {"ccw": {"vars": ["diabetes"], "temporal_res": "daily"}}
    ds = HealthDataset(str(HEALTH), vd, good, window=10, delta_t=20, min_year=2010, max_year=2010, min_bene=10)
    b = ds[100]
    assert tuple(b["outcomes"].shape) == (6, 1, 30)
    assert tuple(b["denom"].shape) == (6, 30)
    mm = np.load(HEALTH / "ccw/diabetes/diabetes__2010.npy", mmap_mode="r")
    cols = [nodes.index(z) for z in good]
    exp = np.asarray(mm[100:130, cols]).T.astype(np.float32)
    got = b["outcomes"][:, 0, :].numpy()
    keep = ~np.isnan(got)                              # masked where denom < min_bene
    assert np.allclose(got[keep], exp[keep])
    exp_denom = np.where(np.isnan(dm[cols]), 0.0, dm[cols]).astype(np.float32)
    exp_denom = np.where(exp_denom < 10, 0.0, exp_denom)
    assert np.allclose(b["denom"][:, 0].numpy(), exp_denom)


def test_healthxdataset_composed_shapes():
    _need(COVARS / "gridmet/tmmx/tmmx__2010.npy")
    _need(HEALTH / "ccw/diabetes/diabetes__2010.npy")
    nodes, good = _covar_nodes()
    vd = {
        "confounders": {"census": {"temporal_res": "yearly", "vars": ["population"]}},
        "treatments": {"gridmet": {"temporal_res": "daily", "vars": ["rmax"]}},
        "outcomes": {"ccw": {"temporal_res": "daily", "vars": ["diabetes"]}},
    }
    ds = HealthXDataset(str(ROOT / "data"), vd, nodes=good, window=7, delta_t=14,
                        summary_stats={}, min_year=2010, max_year=2011)
    b = ds[200]
    assert tuple(b["confounders"].shape) == (len(good), 1, 7)
    assert tuple(b["treatments"].shape) == (len(good), 1, 7)
    assert tuple(b["outcomes"].shape) == (len(good), 1, 21)
    assert tuple(b["denom"].shape) == (len(good), 21)
    assert tuple(b["year"].shape) == (7,)
