"""Main benchmarking script for HealthXDataset.

Two modes, selected by ``mode`` (formats | scaling | all):

  formats   For each on-disk format (daily_parquet, yearly_mmap_dense,
            yearly_mmap_sparse) time `n_samples` random ``__getitem__`` reads
            and record read latency, RSS, and on-disk size.
  scaling   On yearly_mmap_dense, sweep (num_workers, batch_size, prefetch) and
            measure DataLoader throughput over `n_batches` batches per cell.

Everything is driven by Hydra (``conf/benchmark.yaml``). The ``var_dict`` is a
shared config group (``conf/var_dict/``) mirroring the climhealth workload, so
the same file feeds the loaders and this benchmark. Override on the CLI, e.g.

    python benchmarking/benchmark.py var_dict=lego_small mode=formats
    python benchmarking/benchmark.py n_samples=100 batch_sizes=[32,64] n_batches=15

Build the formats first via
    snakemake -s snakefile.smk         --cores N --config format=<fmt>
    snakemake -s snakefile_health.smk  --cores N --config format=<fmt>
"""
from __future__ import annotations
import gc, json, socket, sys, time
from pathlib import Path


import hydra
import numpy as np
import psutil
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from legoloaderx import HealthXDataset


REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "benchmarking" / "results"
PLOTS_DIR = REPO_ROOT / "benchmarking" / "plots"

ROLES = ("outcomes", "treatments", "confounders")
FORMATS = ["daily_parquet", "yearly_mmap_dense", "yearly_mmap_sparse"]


def build_var_dict(cfg):
    """Extract the three HealthXDataset roles from the shared var_dict group
    (ignoring extra keys like ``data_dict`` that come with the climhealth file)."""
    return {role: OmegaConf.to_container(cfg.var_dict[role], resolve=True) for role in ROLES}


def var_dict_size(var_dict):
    return {role: sum(len(g["vars"]) for g in var_dict[role].values()) for role in ROLES}


def get_canonical_nodes(min_year, max_year):
    from legoloaderx.utils import get_unique_ids
    z = str(REPO_ROOT / "data" / "input" / "lego" / "geoboundaries"
            / "us_geoboundaries__census" / "us_uniqueid__census" / "zcta_yearly")
    n, _ = get_unique_ids(z, min_year, max_year)
    return sorted(n)


def percentiles(xs):
    a = np.asarray(xs)
    return {"p50": float(np.percentile(a, 50)),
            "p95": float(np.percentile(a, 95)),
            "p99": float(np.percentile(a, 99)),
            "mean": float(a.mean()),
            "min": float(a.min()), "max": float(a.max()), "n": int(a.size)}


def dir_size(p: Path):
    """Return (logical_bytes, physical_bytes, file_count) under ``p``."""
    if not p.exists():
        return 0, 0, 0
    lg = ph = n = 0
    for f in p.rglob("*"):
        if f.is_file():
            st = f.stat()
            lg += st.st_size
            ph += st.st_blocks * 512
            n += 1
    return lg, ph, n


def materialise(sample):
    for v in sample.values():
        if hasattr(v, "float"):
            _ = v.float().sum().item()


def make_dataset(cfg, var_dict, fmt, canonical, data_root):
    return HealthXDataset(
        root_dir=str(data_root),
        var_dict=var_dict,
        nodes=canonical,
        window=cfg.window,
        delta_t=cfg.delta_t,
        file_format=fmt,
        min_year=cfg.min_year,
        max_year=cfg.max_year,
    )


# --------------------------------------------------------------------------
# mode: formats
# --------------------------------------------------------------------------

def run_formats(cfg, var_dict, canonical, data_root):
    results = []
    for fmt in FORMATS:
        print(f"\n[formats:{fmt}]")
        disks = {}
        for role in ROLES:
            role_dir = "health" if role == "outcomes" else "covars"
            for vg, vd in var_dict[role].items():
                for v in vd["vars"]:
                    if fmt == "yearly_mmap_sparse" and role != "outcomes":
                        continue  # covariates always read dense
                    lg, ph, nf = dir_size(data_root / role_dir / vg / v / fmt)
                    disks[f"{role}/{vg}/{v}"] = {
                        "disk_logical_mb": lg / 1e6,
                        "disk_physical_mb": ph / 1e6,
                        "n_files": nf,
                    }
        tot_lg = sum(d["disk_logical_mb"] for d in disks.values())
        tot_ph = sum(d["disk_physical_mb"] for d in disks.values())
        tot_nf = sum(d["n_files"] for d in disks.values())
        print(f"  disk total: logical={tot_lg:8.2f} MB  physical={tot_ph:8.2f} MB  files={tot_nf}")

        try:
            ds = make_dataset(cfg, var_dict, fmt, canonical, data_root)
            rng = np.random.default_rng(cfg.seed)
            proc = psutil.Process()
            rss0 = proc.memory_info().rss / 1e6

            for w in range(4):  # warm-up
                t0 = time.perf_counter()
                materialise(ds[int(rng.integers(0, len(ds)))])
                print(f"    warmup {w+1}/4: {(time.perf_counter()-t0)*1000.0:8.1f} ms", flush=True)

            times, rss = [], []
            for s in range(cfg.n_samples):
                idx = int(rng.integers(0, len(ds)))
                t0 = time.perf_counter()
                materialise(ds[idx])
                dt = (time.perf_counter() - t0) * 1000.0
                times.append(dt)
                rss.append(proc.memory_info().rss / 1e6)
                print(f"    sample {s+1}/{cfg.n_samples}: {dt:8.1f} ms", flush=True)
        except Exception as e:
            print(f"  read FAILED: {e}")
            results.append({"format": fmt, "error": str(e), "disks": disks})
            continue

        pct = percentiles(times)
        rss_mb = {"init": rss0, "mean_during": float(np.mean(rss)),
                  "peak_during": float(np.max(rss)),
                  "growth_during": float(np.max(rss) - rss0)}
        print(f"  read p50={pct['p50']:7.2f} ms  p95={pct['p95']:7.2f} ms  (n={pct['n']})")
        print(f"  rss  init={rss_mb['init']:.0f} MB  mean={rss_mb['mean_during']:.0f} MB  "
              f"peak={rss_mb['peak_during']:.0f} MB  growth={rss_mb['growth_during']:.0f} MB")
        results.append({"format": fmt, "read_ms": pct, "rss_mb": rss_mb, "disks": disks,
                        "disk_logical_mb_total": tot_lg, "disk_physical_mb_total": tot_ph,
                        "n_files_total": tot_nf})
    return results


def plot_formats(results, size, cfg):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    for r in results:
        if "read_ms" not in r:
            continue
        ax.scatter([r["disk_physical_mb_total"]], [r["read_ms"]["p50"]],
                   s=170, edgecolor="black", linewidth=0.5, zorder=3)
        ax.annotate(r["format"], (r["disk_physical_mb_total"], r["read_ms"]["p50"]),
                    textcoords="offset points", xytext=(7, 7), fontsize=9)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Total physical disk (MB, log)")
    ax.set_ylabel("HealthXDataset read p50 (ms, log)")
    ax.set_title(f"Storage-format trade-off (var_dict: "
                 f"{size['outcomes']} outcomes + {size['treatments']} treatments + "
                 f"{size['confounders']} confounders, window={cfg.window}, "
                 f"delta_t={cfg.delta_t}, n={cfg.n_samples})")
    ax.grid(True, alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "format_benchmarks.png", dpi=140)
    plt.close(fig)
    print(f"→ wrote {PLOTS_DIR / 'format_benchmarks.png'}")


# --------------------------------------------------------------------------
# mode: scaling
# --------------------------------------------------------------------------

def _tree_rss_mb(proc):
    """Total RSS (MB) of `proc` + all its children, and the worker count.

    With num_workers>0 each DataLoader worker is a SEPARATE process, so the
    main process RSS does not see worker memory. We sum the whole tree so the
    reported peak reflects what a training job actually consumes (the relevant
    number for the per-worker memory budget).
    """
    total = proc.memory_info().rss
    kids = proc.children(recursive=True)
    for c in kids:
        try:
            total += c.memory_info().rss
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return total / 1e6, len(kids)


def time_cell(ds, num_workers, batch_size, prefetch_factor, persistent_workers, n_batches):
    proc = psutil.Process()
    kw = dict(batch_size=batch_size, shuffle=True, num_workers=num_workers,
              pin_memory=False,
              persistent_workers=(persistent_workers and num_workers > 0))
    if num_workers > 0:
        kw["prefetch_factor"] = prefetch_factor
    loader = DataLoader(ds, **kw)

    rss0, _ = _tree_rss_mb(proc)
    batch_times, rss = [], [rss0]
    max_kids = 0
    t_start = last = time.perf_counter()
    seen = nb = 0
    for batch in loader:
        now = time.perf_counter()
        batch_times.append((now - last) * 1000.0)
        last = now
        seen += batch_size
        nb += 1
        tree_rss, n_kids = _tree_rss_mb(proc)   # main + worker processes
        rss.append(tree_rss)
        max_kids = max(max_kids, n_kids)
        if nb >= n_batches:
            break
    elapsed = time.perf_counter() - t_start
    sps = seen / elapsed if elapsed > 0 else 0.0

    peak = float(np.max(rss))
    # per-worker RSS budget: total tree peak divided across the live worker
    # processes (>=1 to avoid div-by-zero for the nw=0 single-process case).
    per_worker = peak / max(max_kids, 1)

    del loader
    gc.collect()
    return {
        "num_workers": num_workers, "batch_size": batch_size,
        "prefetch_factor": prefetch_factor if num_workers > 0 else None,
        "persistent_workers": bool(persistent_workers and num_workers > 0),
        "n_batches": nb, "samples_seen": seen,
        "samples_per_sec": sps, "ms_per_sample": 1000.0 / sps if sps > 0 else float("inf"),
        # One sample = one lead-date = one day of training data; 365 samples = one
        # data-year. This normalises out the dataset's year span (a full epoch
        # scales with it, this does not).
        "seconds_per_data_year": 365.0 / sps if sps > 0 else float("inf"),
        "elapsed_s": elapsed,
        "per_batch_ms_after_first": percentiles(batch_times[1:]) if len(batch_times) > 1 else None,
        # RSS is the whole process tree (main + workers) in MB.
        "rss_mb": {"init": rss0, "mean_during": float(np.mean(rss)),
                   "peak_during": peak, "growth_during": peak - rss0,
                   "peak_per_worker": per_worker, "max_worker_procs": max_kids},
    }


def run_scaling(cfg, var_dict, canonical, data_root):
    print("\n[scaling] building dataset (yearly_mmap_dense)...")
    t0 = time.perf_counter()
    ds = make_dataset(cfg, var_dict, "yearly_mmap_dense", canonical, data_root)
    init_ms = (time.perf_counter() - t0) * 1000.0
    print(f"  init_ms={init_ms:.1f}, len={len(ds)}")

    rng = np.random.default_rng(cfg.seed)
    for _ in range(8):  # warm OS page cache
        _ = ds[int(rng.integers(0, len(ds)))]

    print(f"  {'nw':>3} {'bs':>3} {'pf':>3} {'pers':>5}  "
          f"{'sps':>7}  {'s/yr':>7}  {'rss_peak':>9}  {'rss/wkr':>8}  {'procs':>5}")
    cells = []
    for nw in cfg.num_workers:
        for bs in cfg.batch_sizes:
            pfs = [None] if nw == 0 else list(cfg.prefetch)
            for pf in pfs:
                r = time_cell(ds, nw, bs, pf or 2, nw > 0, cfg.n_batches)
                cells.append(r)
                m = r["rss_mb"]
                print(f"  {nw:>3} {bs:>3} {str(pf or '—'):>3} {str(r['persistent_workers']):>5}  "
                      f"{r['samples_per_sec']:>7.2f}  {r['seconds_per_data_year']:>7.1f}  "
                      f"{m['peak_during']:>9.0f}  {m['peak_per_worker']:>8.0f}  {m['max_worker_procs']:>5}")
    best = min((c for c in cells if c["samples_per_sec"] > 0),
               key=lambda c: c["seconds_per_data_year"], default=None)
    if best:
        print(f"  best: {best['seconds_per_data_year']:.1f} s per data-year "
              f"(nw={best['num_workers']}, bs={best['batch_size']}, "
              f"pf={best['prefetch_factor']})")
    return {"init_ms": init_ms, "n_lead_dates": len(ds), "cells": cells}


def plot_scaling(scaling, size, cfg, n_nodes):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    cells = scaling["cells"]
    fig, ax = plt.subplots(figsize=(8, 5))
    for bs in cfg.batch_sizes:
        for pf in cfg.prefetch:
            xs = sorted({c["num_workers"] for c in cells if c["batch_size"] == bs})
            ys = []
            for x in xs:
                m = [c for c in cells if c["num_workers"] == x and c["batch_size"] == bs
                     and (x == 0 or c["prefetch_factor"] == pf)]
                ys.append(m[0]["samples_per_sec"] if m else 0.0)
            ax.plot(xs, ys, marker="o", label=f"bs={bs}, prefetch={pf}")
    ax.set_xlabel("num_workers")
    ax.set_ylabel("samples / second (HealthXDataset)")
    ax.set_title(f"DataLoader scaling — {n_nodes} ZCTAs, window={cfg.window}, "
                 f"delta_t={cfg.delta_t}, var_dict "
                 f"({size['outcomes']}+{size['treatments']}+{size['confounders']})")
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "scaling.png", dpi=140)
    plt.close(fig)
    print(f"→ wrote {PLOTS_DIR / 'scaling.png'}")


# --------------------------------------------------------------------------

@hydra.main(config_path="../conf", config_name="benchmark", version_base=None)
def main(cfg: DictConfig):
    data_root = Path(cfg.data_root).resolve()
    var_dict = build_var_dict(cfg)
    size = var_dict_size(var_dict)
    canonical = get_canonical_nodes(cfg.min_year, cfg.max_year)

    print(f"[benchmark] host={socket.gethostname()} mode={cfg.mode} data_root={data_root}")
    print(f"  var_dict: outcomes={size['outcomes']}, treatments={size['treatments']}, "
          f"confounders={size['confounders']}")
    print(f"  canonical zctas: {len(canonical)}; years {cfg.min_year}-{cfg.max_year}; "
          f"window={cfg.window}, delta_t={cfg.delta_t}")

    out = {
        "host": socket.gethostname(),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "mode": cfg.mode,
        "data_root": str(data_root),
        "var_dict_size": size,
        "config": OmegaConf.to_container(cfg, resolve=True),
        "n_nodes": len(canonical),
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if cfg.mode in ("formats", "all"):
        print(f"\n=== format comparison (n_samples={cfg.n_samples}) ===")
        fmt_results = run_formats(cfg, var_dict, canonical, data_root)
        out["formats"] = fmt_results
        (RESULTS_DIR / "format_benchmarks.json").write_text(json.dumps(out, indent=2))
        print(f"→ wrote {RESULTS_DIR / 'format_benchmarks.json'}")
        try:
            plot_formats(fmt_results, size, cfg)
        except Exception as e:
            print(f"format plot failed: {e}")

    if cfg.mode in ("scaling", "all"):
        print(f"\n=== scaling sweep (n_batches={cfg.n_batches}) ===")
        scaling = run_scaling(cfg, var_dict, canonical, data_root)
        out["scaling"] = scaling
        (RESULTS_DIR / "scaling.json").write_text(json.dumps(out, indent=2))
        print(f"→ wrote {RESULTS_DIR / 'scaling.json'}")
        try:
            plot_scaling(scaling, size, cfg, len(canonical))
        except Exception as e:
            print(f"scaling plot failed: {e}")


if __name__ == "__main__":
    main()
