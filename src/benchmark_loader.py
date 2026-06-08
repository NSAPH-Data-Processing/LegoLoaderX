"""Measure HealthXDataset read speed over the dense .npy mmap store.

A standalone tool (kept out of the Dataset classes): per-sample read latency (a first
pass, then a warm pass over the same indices) and DataLoader throughput across
num_workers. Quantifies the contiguous-mmap-read win over the old per-(var, day)
parquet reads (issue #50). No heavy deps — peak RSS via stdlib ``resource``.

    python src/benchmark_loader.py                       # short smoke run
    python src/benchmark_loader.py n_samples=50 n_batches=20 num_workers=[0,4,8,16]

See 12_benchmarking.md for the perf-backends approach this distils.
"""
import json
import resource
import time
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from legoloaderx import HealthXDataset


def _materialise(sample):
    """Force the lazy mmap reads to actually page in (else we'd time index math only)."""
    for v in sample.values():
        if torch.is_tensor(v):
            v.float().sum().item()


def _pct(ms):
    a = np.asarray(ms, dtype=float)
    return {
        "p50": float(np.percentile(a, 50)),
        "p95": float(np.percentile(a, 95)),
        "mean": float(a.mean()),
        "min": float(a.min()),
        "max": float(a.max()),
        "n": int(a.size),
    }


def _peak_rss_mb():
    # ru_maxrss is KiB on Linux
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def build_dataset(cfg):
    nodes = pd.read_parquet(f"{cfg.data_root}/covars/idx2zcta.parquet")["zcta"].tolist()
    if cfg.n_nodes != "all":
        nodes = nodes[: int(cfg.n_nodes)]
    var_dict = OmegaConf.to_container(cfg.var_dict, resolve=True)
    ds = HealthXDataset(
        root_dir=cfg.data_root,
        var_dict=var_dict,
        nodes=nodes,
        window=cfg.window,
        delta_t=cfg.delta_t,
        summary_stats={},  # benchmarking the read path, not normalization
        min_year=cfg.min_year,
        max_year=cfg.max_year,
    )
    return ds, len(nodes)


def measure_latency(ds, n_samples, seed):
    """Pass 1 (cold-ish) then a warm pass over the SAME indices -> page-cache effect."""
    rng = np.random.default_rng(seed)
    idxs = [int(rng.integers(0, len(ds))) for _ in range(n_samples)]
    cold, warm = [], []
    for i in idxs:
        t = time.perf_counter(); _materialise(ds[i]); cold.append((time.perf_counter() - t) * 1e3)
    for i in idxs:
        t = time.perf_counter(); _materialise(ds[i]); warm.append((time.perf_counter() - t) * 1e3)
    return {"cold_ms": _pct(cold), "warm_ms": _pct(warm)}


def measure_throughput(ds, cfg):
    """For each num_workers, draw n_batches and report sustained throughput."""
    cells = []
    for nw in cfg.num_workers:
        kw = dict(batch_size=cfg.batch_size, shuffle=True, num_workers=nw)
        if nw > 0:
            kw["prefetch_factor"] = cfg.prefetch
        loader = DataLoader(ds, **kw)
        seen = 0
        t0 = time.perf_counter()
        for nb, batch in enumerate(loader, 1):
            _materialise(batch)
            seen += cfg.batch_size
            if nb >= cfg.n_batches:
                break
        elapsed = time.perf_counter() - t0
        sps = seen / elapsed if elapsed > 0 else 0.0
        cells.append({
            "num_workers": nw,
            "batch_size": cfg.batch_size,
            "samples_per_sec": sps,
            "ms_per_sample": 1e3 / sps if sps > 0 else float("inf"),
            # 1 sample = 1 lead-date = 1 day; 365 = one data-year (span-independent unit)
            "seconds_per_data_year": 365.0 / sps if sps > 0 else float("inf"),
            "elapsed_s": elapsed,
        })
        del loader
    return cells


@hydra.main(config_path="../conf/benchmark", config_name="config", version_base=None)
def main(cfg: DictConfig):
    ds, n_nodes = build_dataset(cfg)
    print(f"[benchmark] HealthXDataset len={len(ds)} nodes={n_nodes} "
          f"window={cfg.window} delta_t={cfg.delta_t} years={cfg.min_year}-{cfg.max_year}")

    lat = measure_latency(ds, cfg.n_samples, cfg.seed)
    print(f"  read latency (n={cfg.n_samples}): "
          f"cold p50={lat['cold_ms']['p50']:.1f}ms p95={lat['cold_ms']['p95']:.1f}ms | "
          f"warm p50={lat['warm_ms']['p50']:.1f}ms p95={lat['warm_ms']['p95']:.1f}ms")

    print(f"  {'nw':>3} {'bs':>3} {'samples/s':>10} {'ms/sample':>10} {'s/data-yr':>10}")
    cells = measure_throughput(ds, cfg)
    for c in cells:
        print(f"  {c['num_workers']:>3} {c['batch_size']:>3} {c['samples_per_sec']:>10.2f} "
              f"{c['ms_per_sample']:>10.2f} {c['seconds_per_data_year']:>10.1f}")

    out = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "len": len(ds),
        "n_nodes": n_nodes,
        "latency": lat,
        "throughput": cells,
        "peak_rss_mb": _peak_rss_mb(),
        "config": OmegaConf.to_container(cfg, resolve=True),
    }
    results_path = Path(HydraConfig.get().runtime.output_dir) / "benchmark_results.json"
    results_path.write_text(json.dumps(out, indent=2))
    print(f"  peak_rss={out['peak_rss_mb']:.0f}MB  ->  wrote {results_path}")


if __name__ == "__main__":
    main()
