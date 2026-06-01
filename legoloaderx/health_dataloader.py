import os
import calendar
from datetime import date
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
import pyarrow.parquet as pq
import scipy.sparse

from legoloaderx.utils import load_idx2zcta


def _days_in_year(year):
    return 366 if calendar.isleap(year) else 365


class HealthDataset(Dataset):
    def __init__(
        self,
        root_dir,
        var_dict, # var_dict is structured as {var_group: {"vars": [...], "temporal_res": ...}}
        nodes, # List of zctas
        window,
        delta_t,
        file_format="daily_parquet",
        min_year: int = 2000,
        max_year: int = 2020,
        min_bene: int = 10,
    ):
        assert delta_t is not None and delta_t >= 0, "delta_t must be a non-negative integer"
        assert file_format in ("daily_parquet", "yearly_mmap_dense", "yearly_mmap_sparse"), \
            f"unknown file_format {file_format!r}"
        self.root_dir = root_dir
        self.file_format = file_format

        self.var_dict = var_dict
        # Pull the vars for each var_group in var_dict
        self.vars = [
            f"{var_group_name}_{var}"
            for var_group_name, var_group in var_dict.items()
            for var in var_group["vars"]
        ]
        self.var_to_idx = {var: i for i, var in enumerate(self.vars)}

        self.nodes = nodes
        self.node_to_idx = {node: i for i, node in enumerate(self.nodes)}

        all_dates = pd.date_range(f"{min_year}-01-01", f"{max_year}-12-31", freq="D")
        self.yyyymmdd = [f"{d.year}{d.month:02d}{d.day:02d}"  for d in all_dates]

        self.delta_t = delta_t
        self.lead_dates = self.yyyymmdd[window-1:-delta_t] if delta_t > 0 else self.yyyymmdd[window-1:]

        self.window = window
        self.min_bene = min_bene

        # For non-parquet formats we mmap per-year arrays stored as
        # (n_days, n_zctas). Resolve requested nodes → stored-index column
        # indices once at init.
        self._node_store_idx = None
        self._identity_rows = False
        if file_format != "daily_parquet":
            idx2zcta = load_idx2zcta(self.root_dir)
            zcta2idx = {z: i for i, z in enumerate(idx2zcta)}
            node_idx = np.array([zcta2idx.get(z, -1) for z in nodes], dtype=np.int64)
            if (node_idx < 0).any():
                missing = int((node_idx < 0).sum())
                raise ValueError(
                    f"{missing} of {len(nodes)} requested nodes not in idx2zcta.txt at {self.root_dir}"
                )
            self._node_store_idx = node_idx
            # Fast path: when the requested nodes are exactly the stored idx2zcta list
            # in order, the per-read node-gather is a no-op and can be skipped.
            self._identity_rows = (
                len(node_idx) == len(idx2zcta)
                and bool(np.array_equal(node_idx, np.arange(len(idx2zcta))))
            )

    def __len__(self):
        return len(self.lead_dates)

    # ----- daily_parquet path: original main behaviour, with /{file_format}/ subdir -----

    def _counts_parquet(self, idx):
        counts = torch.zeros((len(self.nodes), len(self.vars), self.window + self.delta_t), dtype=torch.float32)

        for var_group_name, var_group in self.var_dict.items():
            dates = self.yyyymmdd[idx:idx + self.window + self.delta_t]

            for var in var_group["vars"]:
                var_index = self.var_to_idx[f"{var_group_name}_{var}"]

                for date_idx, day in enumerate(dates):
                    file = f"{self.root_dir}/{var_group_name}/{var}/{self.file_format}/{var}__{day}.parquet"
                    if not os.path.exists(file):
                        continue

                    table = pq.read_table(file).to_pandas()
                    if table.empty:
                        continue
                    table["zcta_index"] = table["zcta"].apply(lambda z: self.node_to_idx.get(z, -1))
                    table = table[table["zcta_index"] != -1]

                    zcta_index = torch.LongTensor(table["zcta_index"].values)
                    n = torch.FloatTensor(table["n"].values)
                    counts[zcta_index, var_index, date_idx] = n

        return counts

    # ----- yearly_mmap_* path -----

    def _year_chunks(self, idx, span):
        days = self.yyyymmdd[idx:idx + span]
        cursor = 0
        while cursor < len(days):
            day0 = days[cursor]
            year = int(day0[:4])
            year_days = _days_in_year(year)
            doy0 = (
                date(int(day0[:4]), int(day0[4:6]), int(day0[6:8]))
                - date(year, 1, 1)
            ).days
            length = min(year_days - doy0, len(days) - cursor)
            yield year, cursor, doy0, length
            cursor += length

    def _counts_mmap(self, idx):
        span = self.window + self.delta_t
        counts = np.zeros((len(self.nodes), len(self.vars), span), dtype=np.float32)
        node_idx = self._node_store_idx
        identity = self._identity_rows
        sparse = (self.file_format == "yearly_mmap_sparse")
        ext = "npz" if sparse else "npy"

        # Files are (n_days, n_zctas): a day-window is a contiguous row-slice.
        for var_group_name, var_group in self.var_dict.items():
            for var in var_group["vars"]:
                var_index = self.var_to_idx[f"{var_group_name}_{var}"]
                for year, dst, doy0, length in self._year_chunks(idx, span):
                    path = f"{self.root_dir}/{var_group_name}/{var}/{self.file_format}/{var}__{year}.{ext}"
                    if not os.path.exists(path):
                        continue
                    if sparse:
                        mm = scipy.sparse.load_npz(path)
                        slc = mm[doy0:doy0 + length, :].toarray()      # (length, n_zctas)
                    else:
                        mm = np.load(path, mmap_mode="r")
                        slc = np.ascontiguousarray(mm[doy0:doy0 + length, :])  # (length, n_zctas)
                    # gather requested nodes on the column axis (skip if identity),
                    # then place as (n_nodes, length).
                    sel = slc if identity else slc[:, node_idx]
                    counts[:, var_index, dst:dst + length] = sel.T.astype(np.float32, copy=False)

        return torch.from_numpy(counts)

    # ----- denom (format-aware) -----

    def _denom_and_mask_counts(self, idx, counts):
        span = self.window + self.delta_t
        dates = self.yyyymmdd[idx:idx + span]
        denom = torch.zeros((len(self.nodes), span), dtype=torch.float32)

        # Local per-call denom cache (matches main's pattern — scoped to one
        # __getitem__ call so it doesn't grow across calls).
        denoms_this_call = {}
        for date_idx, day in enumerate(dates):
            year_int = int(day[:4])
            payload = denoms_this_call.get(year_int)
            if payload is None:
                payload = self._load_denom_year(year_int)
                denoms_this_call[year_int] = payload
            if payload is False:
                continue
            zcta_idx_t, n_bene_t, zero_mask_t = payload
            denom[zcta_idx_t, date_idx] = n_bene_t
            counts[zcta_idx_t[zero_mask_t], ..., date_idx] = torch.nan

        return denom

    def _load_denom_year(self, year):
        """Return (zcta_idx_tensor, n_bene_tensor, zero_mask_tensor) for the given year.
        zero_mask flags rows that are in source AND have n_bene < min_bene (so counts get NaN-masked).
        """
        if self.file_format == "daily_parquet":
            path = f"{self.root_dir}/denom/{self.file_format}/denom__{year}.parquet"
            if not os.path.exists(path):
                return False
            df = pq.read_table(path).to_pandas()
            df["zcta_index"] = df["zcta"].map(lambda z: self.node_to_idx.get(z, -1))
            df = df[df["zcta_index"] != -1]
            n_bene = df["n_bene"].values.astype(np.float32)
            zcta_idx = df["zcta_index"].values.astype(np.int64, copy=False)
            zero_mask = n_bene < self.min_bene
            n_bene[zero_mask] = 0
            return (
                torch.from_numpy(zcta_idx).long(),
                torch.from_numpy(n_bene),
                torch.from_numpy(zero_mask),
            )

        # yearly_mmap_*: dense int32 .npy with sentinel -1 for "missing from source".
        # Denom is always written under /denom/yearly_mmap_dense/ even when outcomes
        # use sparse, since denom is a single-vector-per-year and doesn't benefit
        # from sparsity.
        path = f"{self.root_dir}/denom/yearly_mmap_dense/denom__{year}.npy"
        if not os.path.exists(path):
            return False
        arr = np.load(path, mmap_mode="r")[self._node_store_idx]
        in_source = arr >= 0
        n_bene = np.where(in_source, arr, 0).astype(np.float32)
        zero_mask = in_source & (n_bene < self.min_bene)
        n_bene[zero_mask] = 0
        rows = np.arange(len(self.nodes), dtype=np.int64)
        return (
            torch.from_numpy(rows),
            torch.from_numpy(n_bene),
            torch.from_numpy(zero_mask),
        )

    def __getitem__(self, idx):
        if self.file_format == "daily_parquet":
            counts = self._counts_parquet(idx)
        else:
            counts = self._counts_mmap(idx)

        denom = self._denom_and_mask_counts(idx, counts)

        return {
            "outcomes": counts,
            "denom": denom,
        }


def main():
    root_dir = "data/health"
    var_dict = {
        "ccw": {
            "vars": ["anemia", "asthma"],
            "temporal_res": "daily"
        }
    }
    nodes_list = ["00601", "00602"]
    window = 30
    delta_t = 180
    min_year = 2000
    max_year = 2014

    dataset = HealthDataset(root_dir, var_dict, nodes_list, window, delta_t=delta_t, min_year=min_year, max_year=max_year)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0)

    for batch in dataloader:
        print(batch['outcomes'].shape)

if __name__ == "__main__":
    main()
