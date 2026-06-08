import calendar
import json
import os
from datetime import date

import hydra
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig
from torch.utils.data import Dataset


def _days_in_year(year):
    return 366 if calendar.isleap(year) else 365


class XDataset(Dataset):
    """Covariate / treatment loader over the dense per-(var, year) ``.npy`` store.

    Each file is ``(n_days, n_zctas)`` (daily) or ``(n_zctas,)`` (yearly), float32,
    NaN where the source is missing, in ``idx2zcta`` row order. A lead-date window is a
    contiguous row-slice of the time axis (planning doc 10).

    Normalization (the X-only concern) lives entirely on this class:
    ``summary_stats`` may be a path/dict to load, or ``None`` to compute from the raw
    store at init; ``{}`` disables normalization. ``__getitem__`` always applies
    ``(x - mean)/std`` (NaNs stay NaN).
    """

    # single canonical normalization-stats artifact per read root (cf. idx2zcta.parquet)
    SUMMARY_STATS_FILENAME = "summary_statistics.json"

    def __init__(
        self,
        root_dir,
        var_dict,        # {var_group: {"vars": [...], "temporal_res": ...}}
        nodes,           # list of zctas to gather (a subset/reorder of idx2zcta)
        window,          # temporal window size
        summary_stats=None,  # path|dict -> load; None -> compute from store; {} -> no normalization
        transform=None,      # not implemented right now
        min_year=2000,
        max_year=2020,
    ):
        self.root_dir = root_dir
        self.transform = transform
        self.var_dict = var_dict
        self.window = window
        self.min_year = min_year
        self.max_year = max_year

        self.vars = [f"{vg}_{var}" for vg, g in var_dict.items() for var in g["vars"]]
        self.var_to_idx = {var: i for i, var in enumerate(self.vars)}
        self.nodes = nodes

        all_dates = pd.date_range(f"{min_year}-01-01", f"{max_year}-12-31", freq="D")
        self.yyyymmdd = [f"{d.year}{d.month:02d}{d.day:02d}" for d in all_dates]
        # window-1 of history is needed before the first lead date
        self.lead_dates = self.yyyymmdd[window - 1:]

        # Resolve requested nodes -> stored column index once, via the canonical order.
        idx2zcta = pd.read_parquet(f"{root_dir}/idx2zcta.parquet")["zcta"].tolist()
        zcta2idx = {z: i for i, z in enumerate(idx2zcta)}
        node_idx = np.array([zcta2idx.get(z, -1) for z in nodes], dtype=np.int64)
        if (node_idx < 0).any():
            missing = int((node_idx < 0).sum())
            raise ValueError(f"{missing} of {len(nodes)} requested nodes not in {root_dir}/idx2zcta.parquet")
        self._node_store_idx = node_idx
        # Fast path: requested nodes == full stored order -> skip the column gather.
        self._identity_rows = (
            len(node_idx) == len(idx2zcta)
            and bool(np.array_equal(node_idx, np.arange(len(idx2zcta))))
        )

        # Normalization stats: load if given (path or dict), else compute once from the raw
        # store. Computing reads the .npy files directly (never __getitem__), so it can't
        # form a normalization cycle (cf. #44).
        if summary_stats is not None:
            self.summary_stats = self.load_summary_stats(summary_stats)
        else:
            self.summary_stats = self.compute_summary(root_dir, var_dict, min_year, max_year)

    def __len__(self):
        return len(self.lead_dates)

    def _year_chunks(self, idx, span):
        """Split the [idx, idx+span) day-window at calendar-year boundaries.

        Yields (year, dst_offset, doy0, length): per-year mmaps cover one year, so a
        multi-year window reads one contiguous slice per year.
        """
        days = self.yyyymmdd[idx:idx + span]
        cursor = 0
        while cursor < len(days):
            day0 = days[cursor]
            year = int(day0[:4])
            doy0 = (date(year, int(day0[4:6]), int(day0[6:8])) - date(year, 1, 1)).days
            length = min(_days_in_year(year) - doy0, len(days) - cursor)
            yield year, cursor, doy0, length
            cursor += length

    def __getitem__(self, idx):
        out = np.full((len(self.nodes), len(self.vars), self.window), np.nan, dtype=np.float32)
        node_idx = self._node_store_idx

        for var_group_name, var_group in self.var_dict.items():
            temporal_res = var_group["temporal_res"]
            for var in var_group["vars"]:
                var_index = self.var_to_idx[f"{var_group_name}_{var}"]
                mean, std = self.get_var_summy(var_group_name, var)

                for year, dst, doy0, length in self._year_chunks(idx, self.window):
                    path = f"{self.root_dir}/{var_group_name}/{var}/{var}__{year}.npy"
                    if not os.path.exists(path):
                        continue
                    mm = np.load(path, mmap_mode="r")
                    if temporal_res == "yearly":
                        # mm shape (n_zctas,): broadcast the single value across `length` days.
                        vec = mm if self._identity_rows else mm[node_idx]
                        chunk = np.asarray(vec, dtype=np.float32)
                        if std != 1 or mean != 0:
                            chunk = (chunk - mean) / std
                        out[:, var_index, dst:dst + length] = chunk[:, None]
                    elif temporal_res == "daily":
                        # mm shape (n_days, n_zctas): a day-window is a contiguous row-slice.
                        slc = np.ascontiguousarray(mm[doy0:doy0 + length, :])  # (length, n_zctas)
                        sel = slc if self._identity_rows else slc[:, node_idx]
                        chunk = sel.T.astype(np.float32, copy=False)           # (n_nodes, length)
                        if std != 1 or mean != 0:
                            chunk = (chunk - mean) / std
                        out[:, var_index, dst:dst + length] = chunk
                    else:
                        raise ValueError(f"unsupported temporal_res {temporal_res!r} (daily/yearly only)")

        tensor = torch.from_numpy(out)
        if self.transform:
            tensor = self.transform(tensor)
        return tensor

    # ----- normalization stats (the X-only concern) -----

    @staticmethod
    def summary_stats_path(root_dir):
        """Canonical per-root location of the normalization stats (cf. idx2zcta.parquet)."""
        return os.path.join(root_dir, XDataset.SUMMARY_STATS_FILENAME)

    @staticmethod
    def load_summary_stats(src):
        """Return the nested stats dict from a path or dict; fail loud on a missing path."""
        if src is None:
            return None
        if isinstance(src, (str, os.PathLike)):
            if not os.path.exists(src):
                raise FileNotFoundError(f"summary_stats not found: {src}")
            with open(src) as f:
                return json.load(f)
        return dict(src)

    def get_var_summy(self, var_group_name, var_name):
        """(mean, std) for a variable; (0, 1) when stats are absent (a no-op normalize)."""
        if not self.summary_stats:
            return 0.0, 1.0
        entry = self.summary_stats.get(var_group_name, {}).get(var_name)
        if entry is None:
            return 0.0, 1.0
        std = entry.get("std")
        return entry.get("mean", 0.0), (std if std else 1.0)

    @staticmethod
    def compute_summary(root_dir, var_dict, min_year, max_year):
        """Per-variable (mean, std, frac_nan) reduced directly from the raw .npy store over
        [min_year, max_year], nan-aware, float64 accumulation. Reads each (var, year) file
        once and never goes through __getitem__, so it is node/window-independent and cannot
        form a normalization cycle (cf. #44). Returns nested {vg: {var: {...}}}."""
        summary = {}
        for vg, group in var_dict.items():
            summary[vg] = {}
            for var in group["vars"]:
                total_sum = total_ss = 0.0
                n = n_nan = 0
                for year in range(min_year, max_year + 1):
                    path = f"{root_dir}/{vg}/{var}/{var}__{year}.npy"
                    if not os.path.exists(path):
                        continue
                    a = np.load(path).astype(np.float64, copy=False).ravel()
                    nan = np.isnan(a)
                    n_nan += int(nan.sum())
                    valid = a[~nan]
                    total_sum += float(valid.sum())
                    total_ss += float(np.dot(valid, valid))
                    n += int(valid.size)
                if n > 0:
                    mean = total_sum / n
                    std = float(np.sqrt(max(total_ss / n - mean * mean, 0.0)))
                    frac_nan = n_nan / (n + n_nan)
                else:
                    mean, std, frac_nan = 0.0, 1.0, 1.0
                summary[vg][var] = {"mean": mean, "std": std, "frac_nan": frac_nan}
        return summary

    @staticmethod
    def save_summary_stats(summary, root_dir):
        """Persist the stats dict to the canonical per-root path; returns the path."""
        os.makedirs(root_dir, exist_ok=True)
        path = XDataset.summary_stats_path(root_dir)
        with open(path, "w") as f:
            json.dump(summary, f, indent=2)
        return path


@hydra.main(config_path="../conf/dataloader", config_name="config", version_base=None)
def compute_stats(cfg: DictConfig):
    """CLI: compute + persist the covariate normalization stats.

    Only var_groups flagged ``valid_normalize`` contribute. Reduces the raw ``.npy`` store
    over ``[min_year, max_year]`` and writes the single ``summary_statistics.json`` at
    ``data_dir`` (consumed back via ``XDataset(summary_stats=...)``).
    """
    import yaml
    var_dict = {}
    for vg in cfg.var_groups:
        with open(f"conf/var_group/{vg}.yaml", "r") as f:
            vg_cfg = yaml.safe_load(f)
        if vg_cfg.get("valid_normalize", False):
            var_dict[vg] = {
                "vars": vg_cfg["vars"],
                "temporal_res": vg_cfg["min_temporal_res"],
                "spatial_res": vg_cfg["min_spatial_res"],
            }

    stats = XDataset.compute_summary(cfg.data_dir, var_dict, cfg.min_year, cfg.max_year)
    path = XDataset.save_summary_stats(stats, cfg.data_dir)
    n_vars = sum(len(g["vars"]) for g in var_dict.values())
    print(f"Saved summary statistics for {n_vars} vars across {len(var_dict)} var_groups to {path}")


if __name__ == "__main__":
    compute_stats()
