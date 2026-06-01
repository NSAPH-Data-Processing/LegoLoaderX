import calendar
import logging
import os
import time
import json
from datetime import date

import hydra
import numpy as np
import torch
import pandas as pd
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import pyarrow.parquet as pq
from legoloaderx.utils import compute_summary, load_summary_stats, get_var_summy, get_unique_ids, load_idx2zcta


def _days_in_year(year):
    return 366 if calendar.isleap(year) else 365


class XDataset(Dataset):
    def __init__(
        self,
        root_dir,
        var_dict, #var_dict is structured as {var_group: {"vars": [...], "temporal_res": ...}}
        nodes,  # List of zctas (required)
        window,   # Window size for temporal data (required)
        file_format="daily_parquet",  # daily_parquet | yearly_mmap_dense
        transform=None, # not implemented right now
        min_year = 2000,
        max_year = 2020,
        normalize=False,  # Optional path or dict of summary stats
    ):
        assert file_format in ("daily_parquet", "yearly_mmap_dense"), \
            f"unknown file_format {file_format!r} for XDataset (sparse only applies to outcomes)"
        self.root_dir = root_dir
        self.file_format = file_format
        self.transform = transform
        self.var_dict = var_dict
        self.window = window

        if not normalize:
            self.summary_stats = None
        else:
            self.summary_stats = load_summary_stats(os.path.join(self.root_dir, "summary_statistics/summary_statistics.json"))

        # Pull the vars for each var_group in var_dict
        self.vars = [f"{var_group_name}_{var}" for var_group_name, var_group in var_dict.items() for var in var_group["vars"]]
        self.var_to_idx = {var: i for i, var in enumerate(self.vars)}

        # Handle nodes (zctas)
        self.nodes = nodes
        self.node_to_idx = {node: i for i, node in enumerate(self.nodes)}

        all_dates = pd.date_range(f"{min_year}-01-01", f"{max_year}-12-31", freq="D")
        self.yyyymmdd = [f"{d.year}{d.month:02d}{d.day:02d}"  for d in all_dates]

        # For windowed data, we need to start from window-1 to have enough history
        self.lead_dates = self.yyyymmdd[window-1:]

        # For daily_parquet: per-var_group row→zcta assignment cache (existing behaviour).
        self.row_to_zcta_assignments = {}

        # For yearly_mmap_dense: resolve requested nodes → stored-index column index once
        # at init (files are (n_days, n_zctas)).
        self._node_store_idx = None
        self._identity_rows = False
        if file_format == "yearly_mmap_dense":
            idx2zcta = load_idx2zcta(self.root_dir)
            zcta2idx = {z: i for i, z in enumerate(idx2zcta)}
            node_idx = np.array([zcta2idx.get(z, -1) for z in nodes], dtype=np.int64)
            if (node_idx < 0).any():
                missing = int((node_idx < 0).sum())
                raise ValueError(
                    f"{missing} of {len(nodes)} requested nodes not in idx2zcta.txt at {self.root_dir}"
                )
            self._node_store_idx = node_idx
            # Fast path: requested nodes == stored idx2zcta order -> skip gather.
            self._identity_rows = (
                len(node_idx) == len(idx2zcta)
                and bool(np.array_equal(node_idx, np.arange(len(idx2zcta))))
            )

    def __len__(self):
        return len(self.lead_dates)

    def __getitem__(self, idx):
        if self.file_format == "daily_parquet":
            tensor = self._getitem_parquet(idx)
        else:
            tensor = self._getitem_mmap(idx)

        if self.transform:
            tensor = self.transform(tensor)

        return tensor

    # ----- daily_parquet path: original main behaviour, with /{file_format}/ subdir -----

    def _getitem_parquet(self, idx):
        dates = self.yyyymmdd[idx:idx + self.window]
        tensor = torch.full((len(self.nodes), len(self.vars), self.window), fill_value=torch.nan, dtype=torch.float32)

        for var_group_name, var_group in self.var_dict.items():
            temporal_res = var_group["temporal_res"]

            for var in var_group["vars"]:
                var_index = self.var_to_idx[f"{var_group_name}_{var}"]

                for date_idx, date_str in enumerate(dates):
                    if temporal_res == "yearly":
                        file_date_str = date_str[:4]
                    elif temporal_res == "monthly":
                        file_date_str = date_str[:6]
                    else:  # daily
                        file_date_str = date_str

                    filename = f"{self.root_dir}/{var_group_name}/{var}/{self.file_format}/{var}__{file_date_str}.parquet"

                    if var_group_name not in self.row_to_zcta_assignments:
                        if not os.path.exists(filename):
                            logging.warning(f"File {filename} does not exist. Filling with NaNs.")
                            continue

                        table = pq.read_table(filename, columns=["zcta"]).to_pandas()
                        table["zcta_index"] = table["zcta"].apply(lambda z: self.node_to_idx.get(z, -1))
                        row_filter = (table["zcta_index"] != -1).values
                        zcta_index = torch.tensor(table["zcta_index"][row_filter].values, dtype=torch.long)
                        self.row_to_zcta_assignments[var_group_name] = (zcta_index, row_filter)
                    else:
                        zcta_index, row_filter = self.row_to_zcta_assignments[var_group_name]

                    if os.path.exists(filename):
                        table = pq.read_table(filename, columns=[var]).to_pandas()
                    else:
                        logging.warning(f"File {filename} does not exist. Filling with NaNs.")
                        table = pd.DataFrame(columns=[var])

                    if not table.empty:
                        values = torch.tensor(table[var][row_filter].values, dtype=torch.float32)
                        mean, std = get_var_summy(self.summary_stats, var_group_name, var)
                        mask = ~torch.isnan(values)
                        values[mask] = (values[mask] - mean) / std
                        tensor[zcta_index, var_index, date_idx] = values

        return tensor

    # ----- yearly_mmap_dense path -----

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

    def _getitem_mmap(self, idx):
        out = np.full((len(self.nodes), len(self.vars), self.window), np.nan, dtype=np.float32)
        node_idx = self._node_store_idx

        for var_group_name, var_group in self.var_dict.items():
            temporal_res = var_group["temporal_res"]
            for var in var_group["vars"]:
                var_index = self.var_to_idx[f"{var_group_name}_{var}"]
                mean, std = get_var_summy(self.summary_stats, var_group_name, var)

                for year, dst, doy0, length in self._year_chunks(idx, self.window):
                    path = f"{self.root_dir}/{var_group_name}/{var}/{self.file_format}/{var}__{year}.npy"
                    if not os.path.exists(path):
                        continue
                    mm = np.load(path, mmap_mode="r")
                    if temporal_res == "yearly":
                        # mm shape: (n_zctas,). Broadcast the single value across `length` days.
                        vec = mm if self._identity_rows else mm[node_idx]
                        chunk = np.asarray(vec).astype(np.float32, copy=False)
                        if std != 1 or mean != 0:
                            chunk = (chunk - mean) / std
                        out[:, var_index, dst:dst + length] = chunk[:, None]
                    elif temporal_res == "daily":
                        # file is (n_days, n_zctas): day-window is a contiguous row-slice.
                        slc = np.ascontiguousarray(mm[doy0:doy0 + length, :])  # (length, n_zctas)
                        sel = slc if self._identity_rows else slc[:, node_idx]
                        chunk = sel.T.astype(np.float32, copy=False)           # (n_nodes, length)
                        if std != 1 or mean != 0:
                            chunk = (chunk - mean) / std
                        out[:, var_index, dst:dst + length] = chunk
                    else:
                        raise ValueError(f"unsupported temporal_res {temporal_res!r} for yearly_mmap_dense")

        return torch.from_numpy(out)




@hydra.main(config_path="../conf/dataloader", config_name="config", version_base=None)
def main(cfg: DictConfig):
    import yaml
    var_dict = {}

    # iterate through variable groups and collect names of all variables
    for vg in cfg.var_groups:
        var_dict[vg] = {}
        with open(f"conf/var_group/{vg}.yaml", "r") as f:
            vg_cfg = yaml.safe_load(f)
            # get variable names
            var_dict[vg]["vars"] = vg_cfg["vars"]
            # store spatial and temporal res
            var_dict[vg]["temporal_res"] = vg_cfg["min_temporal_res"]
            var_dict[vg]["spatial_res"] = vg_cfg["min_spatial_res"]
            f.close()

    root_dir = cfg.data_dir
    zcta_dir = f"{cfg.data_dir}/lego/geoboundaries/us_geoboundaries__census/us_uniqueid__census/zcta_yearly"
    unique_zctas, _ = get_unique_ids(zcta_dir, cfg.min_year, cfg.max_year)

    # initialize dataset
    dataset = XDataset(
        root_dir=root_dir,
        transform=None,
        var_dict=var_dict,
        nodes=unique_zctas,
        normalize = cfg.normalize if hasattr(cfg, 'normalize') else False,
        window=cfg.window if hasattr(cfg, 'window') else 7,  # Default window if not specified
        min_year = cfg.min_year, 
        max_year = cfg.max_year
    )

    # adapt to dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        num_workers=0,
        # pin_memory=True,
        # persistent_workers=True,
    )

    compute_summary(dataloader, output_dir=f"{cfg.data_dir}/{cfg.summary_stats_dir}")


if __name__ == "__main__":
    main()
