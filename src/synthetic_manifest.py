"""Run-time manifest: persist the exact DGP parameters *alongside* the generated data.

WHY
---
The synthetic outcome is generated from the causal parameters in ``conf/synthetic/config.yaml``
(``beta``, the ``gamma_k``, shapes, interactions, ``poisson_params`` ...). If that file is later
edited, the record of *what actually produced a given dataset* is lost, and any "ground truth" we
compute from the live config no longer matches the data on disk.

So at generation time we write a **manifest** next to the counts parquet: a JSON snapshot of the
fully-resolved ``cfg.synthetic`` block plus provenance (year, var, timestamp, git commit). The
manifest — not the live config — is the authoritative ground truth for that dataset. Downstream
tools (e.g. ``ground_truth_erc.py``) read it back so the "answer key" is guaranteed to match the
data, even months later or on a different machine.

The manifest travels with the data and is human-readable; open it to see the exact dose-response
(``beta`` + shape), every confounder coefficient, and any interaction terms used.
"""

import json
import logging
import os
import subprocess
from datetime import datetime, timezone

from omegaconf import OmegaConf

LOGGER = logging.getLogger(__name__)

# Counts live here (must match the path the real health pipeline reads). The manifest sits in the
# SAME directory with a parallel name so the two are obviously paired and move together.
COUNTS_DIR = "data/input/lego/medicare_synthetic/medpar_outcomes/ccw/zcta_daily"

MANIFEST_SCHEMA_VERSION = 1


def counts_path(var_name, year):
    """Parquet of synthetic outcome counts for one (var, year)."""
    return f"{COUNTS_DIR}/sparse_counts_{var_name}_{year}.parquet"


def manifest_path(var_name, year):
    """Manifest JSON paired with ``counts_path(var_name, year)`` (same dir, parallel name)."""
    return f"{COUNTS_DIR}/sparse_counts_{var_name}_{year}.manifest.json"


def _git_commit():
    """Best-effort short commit hash for provenance; ``None`` if unavailable (not a git checkout)."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        return out.stdout.strip() or None if out.returncode == 0 else None
    except Exception:  # pragma: no cover - provenance is best-effort, never fatal
        return None


def build_manifest(cfg):
    """Assemble the manifest dict for the current run from ``cfg`` (resolved, no interpolations left)."""
    synthetic = OmegaConf.to_container(cfg.synthetic, resolve=True)
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "var": cfg.synthetic.var_name,
        "year": int(cfg.year),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "data_file": counts_path(cfg.synthetic.var_name, cfg.year),
        "dgp": (
            "counts ~ Poisson(rate * population_normalizer * population);  "
            "rate = max(rate_floor, spacetime_coupling( base + seasonal + lat + lon "
            "+ beta*shape(exposure) + sum_k gamma_k*shape(standardized(C_k)) + interaction_terms ));  "
            "spacetime_coupling = AR(1)_phi over days of SAR_rho over neighbours (no-op when rho=phi=0)"
        ),
        # The full resolved synthetic config = the ground-truth parameters for this dataset.
        "synthetic": synthetic,
    }


def write_manifest(cfg):
    """Write the manifest next to the counts parquet; return its path."""
    path = manifest_path(cfg.synthetic.var_name, cfg.year)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(build_manifest(cfg), f, indent=2)
    LOGGER.info(f"Wrote DGP manifest (ground-truth parameters) -> {path}")
    return path


def load_manifest(var_name, year, path=None):
    """Load the manifest for one (var, year), or ``None`` if it does not exist.

    ``path`` overrides the default location (e.g. when data lives in lab storage).
    """
    p = path or manifest_path(var_name, year)
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


# The keys under ``synthetic`` that define the data-generating rate. Only these are pulled from a
# manifest when re-deriving the ground truth; everything else (ERC grid, forecast window, output
# dir, store paths) is taken from the *live* config so it can still be overridden on the CLI.
DGP_KEYS = (
    "beta", "exposure_covars_root", "exposure_var_group", "exposure_var", "exposure_shape",
    "confounders", "interactions", "interaction_terms", "poisson_params", "spacetime",
)


def apply_dgp_from_manifest(cfg, manifest):
    """Return ``cfg`` with its DGP parameters (``DGP_KEYS`` under ``synthetic``) replaced by the
    manifest's, so the ground truth is computed from the parameters that actually generated the
    data rather than the (possibly edited) live config. Non-DGP settings are left untouched.
    """
    syn = manifest.get("synthetic", {})
    overlay = {k: syn[k] for k in DGP_KEYS if k in syn}
    return OmegaConf.merge(cfg, OmegaConf.create({"synthetic": overlay}))
