import calendar
import os
from datetime import date

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


def _days_in_year(year):
    return 366 if calendar.isleap(year) else 365


class HealthDataset(Dataset):
    """Outcome + denominator loader over the dense per-(var, year) ``.npy`` store.

    Outcomes: int16 ``(n_days, n_zctas)`` same-day event counts, ``0`` where absent.
    Denominator: float32 ``(n_zctas,)`` beneficiary counts, ``NaN`` where absent from
    source. ``__getitem__`` returns
    ``{"outcomes": (n_nodes, n_vars, window+delta_t), "denom": (n_nodes, window+delta_t)}``,
    with outcome counts NaN-masked wherever the denom is in-source but below ``min_bene``.
    A day-window is a contiguous row-slice of the time axis (planning doc 10).
    """

    def __init__(
        self,
        root_dir,
        var_dict,   # {var_group: {"vars": [...], "temporal_res": "daily"}}
        nodes,      # list of zctas (a subset/reorder of idx2zcta)
        window,
        delta_t,
        min_year=2000,
        max_year=2020,
        min_bene=10,
    ):
        assert delta_t is not None and delta_t >= 0, "delta_t must be a non-negative integer"
        self.root_dir = root_dir
        self.var_dict = var_dict
        self.window = window
        self.delta_t = delta_t
        self.min_bene = min_bene

        self.vars = [f"{vg}_{var}" for vg, g in var_dict.items() for var in g["vars"]]
        self.var_to_idx = {var: i for i, var in enumerate(self.vars)}
        self.nodes = nodes

        all_dates = pd.date_range(f"{min_year}-01-01", f"{max_year}-12-31", freq="D")
        self.yyyymmdd = [f"{d.year}{d.month:02d}{d.day:02d}" for d in all_dates]
        self.lead_dates = self.yyyymmdd[window - 1:-delta_t] if delta_t > 0 else self.yyyymmdd[window - 1:]

        # Resolve requested nodes -> stored column index once, via the canonical order.
        idx2zcta = pd.read_parquet(f"{root_dir}/idx2zcta.parquet")["zcta"].tolist()
        zcta2idx = {z: i for i, z in enumerate(idx2zcta)}
        node_idx = np.array([zcta2idx.get(z, -1) for z in nodes], dtype=np.int64)
        if (node_idx < 0).any():
            missing = int((node_idx < 0).sum())
            raise ValueError(f"{missing} of {len(nodes)} requested nodes not in {root_dir}/idx2zcta.parquet")
        self._node_store_idx = node_idx
        self._identity_rows = (
            len(node_idx) == len(idx2zcta)
            and bool(np.array_equal(node_idx, np.arange(len(idx2zcta))))
        )

    def __len__(self):
        return len(self.lead_dates)

    def _year_chunks(self, idx, span):
        days = self.yyyymmdd[idx:idx + span]
        cursor = 0
        while cursor < len(days):
            day0 = days[cursor]
            year = int(day0[:4])
            doy0 = (date(year, int(day0[4:6]), int(day0[6:8])) - date(year, 1, 1)).days
            length = min(_days_in_year(year) - doy0, len(days) - cursor)
            yield year, cursor, doy0, length
            cursor += length

    def _counts(self, idx):
        span = self.window + self.delta_t
        counts = np.zeros((len(self.nodes), len(self.vars), span), dtype=np.float32)
        node_idx = self._node_store_idx
        # Files are (n_days, n_zctas): a day-window is a contiguous row-slice.
        for var_group_name, var_group in self.var_dict.items():
            for var in var_group["vars"]:
                var_index = self.var_to_idx[f"{var_group_name}_{var}"]
                for year, dst, doy0, length in self._year_chunks(idx, span):
                    path = f"{self.root_dir}/{var_group_name}/{var}/{var}__{year}.npy"
                    if not os.path.exists(path):
                        continue
                    mm = np.load(path, mmap_mode="r")
                    slc = np.ascontiguousarray(mm[doy0:doy0 + length, :])  # (length, n_zctas)
                    sel = slc if self._identity_rows else slc[:, node_idx]
                    counts[:, var_index, dst:dst + length] = sel.T.astype(np.float32, copy=False)
        return torch.from_numpy(counts)

    def _denom_and_mask_counts(self, idx, counts):
        span = self.window + self.delta_t
        dates = self.yyyymmdd[idx:idx + span]
        denom = torch.zeros((len(self.nodes), span), dtype=torch.float32)

        cache = {}  # per-call, keyed by year
        for date_idx, day in enumerate(dates):
            year = int(day[:4])
            payload = cache.get(year)
            if payload is None:
                payload = self._load_denom_year(year)
                cache[year] = payload
            if payload is False:
                continue
            n_bene, zero_mask = payload
            denom[:, date_idx] = n_bene
            counts[zero_mask, :, date_idx] = torch.nan  # too few beneficiaries -> mask outcomes
        return denom

    def _load_denom_year(self, year):
        """Return (n_bene, zero_mask) tensors for the year, or False if the file is absent.

        Denom is float32 with NaN = absent-from-source (0.0 = a genuine zero). Absent rows
        contribute 0 and are not masked; in-source rows below ``min_bene`` are set to 0 and
        flag their outcome counts for NaN-masking.
        """
        path = f"{self.root_dir}/denom/denom__{year}.npy"
        if not os.path.exists(path):
            return False
        arr = np.load(path, mmap_mode="r")
        arr = arr if self._identity_rows else arr[self._node_store_idx]
        arr = np.asarray(arr, dtype=np.float32)
        in_source = ~np.isnan(arr)
        n_bene = np.where(in_source, arr, 0.0).astype(np.float32)
        zero_mask = in_source & (n_bene < self.min_bene)
        n_bene[zero_mask] = 0.0
        return torch.from_numpy(n_bene), torch.from_numpy(zero_mask)

    def __getitem__(self, idx):
        counts = self._counts(idx)
        denom = self._denom_and_mask_counts(idx, counts)
        return {"outcomes": counts, "denom": denom}


def main():
    root_dir = "data/health"
    var_dict = {"ccw": {"vars": ["anemia", "asthma"], "temporal_res": "daily"}}
    nodes = pd.read_parquet(f"{root_dir}/idx2zcta.parquet")["zcta"].tolist()[:50]
    dataset = HealthDataset(root_dir, var_dict, nodes, window=30, delta_t=180,
                            min_year=2000, max_year=2014)
    loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0)
    batch = next(iter(loader))
    print(f"HealthDataset: outcomes={tuple(batch['outcomes'].shape)} denom={tuple(batch['denom'].shape)}")


if __name__ == "__main__":
    main()
