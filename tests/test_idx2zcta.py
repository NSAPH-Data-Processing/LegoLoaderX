"""Validate the canonical idx2zcta order.

``idx2zcta`` defines the row-index -> zcta order used to gather *aligned* rows
across covars, treatments, and outcomes at load time. It must be built the same
way from a single source everywhere, so the per-root copies (``data/covars``,
``data/health``) must be identical. See planning docs 06/07.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
COVARS = ROOT / "data" / "covars" / "idx2zcta.parquet"
HEALTH = ROOT / "data" / "health" / "idx2zcta.parquet"


def _load(path: Path) -> pd.DataFrame:
    if not path.exists():
        pytest.skip(f"{path} missing; run the idx2zcta snakemake rule first")
    return pd.read_parquet(path)


def test_idx2zcta_identical_across_roots():
    """covars and health copies must be the same order (same source, same script)."""
    covars = _load(COVARS)
    health = _load(HEALTH)
    assert list(covars.columns) == list(health.columns)
    assert covars["zcta"].tolist() == health["zcta"].tolist()


def test_idx2zcta_is_canonical_order():
    """A single zcta column whose row order is the sorted, unique index order."""
    zctas = _load(COVARS)["zcta"].tolist()
    assert list(_load(COVARS).columns) == ["zcta"]
    assert zctas == sorted(zctas)
    assert len(zctas) == len(set(zctas))
