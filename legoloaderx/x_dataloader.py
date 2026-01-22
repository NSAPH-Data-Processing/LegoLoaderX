import logging
import os
import time
import json

import hydra
import torch
import pandas as pd
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import pyarrow.parquet as pq
from utils import compute_summary, load_summary_stats, get_var_summy, get_unique_ids



class XDataset(Dataset):
    def __init__(
        self,
        root_dir,
        var_dict, #var_dict is structured as {var_group: {"vars": [...], "temporal_res": ...}}
        nodes,  # List of zctas (required)
        window,   # Window size for temporal data (required)
        transform=None, # not implemented right now
        min_year = 2000,
        max_year = 2020,
        normalize=False,  # Optional path or dict of summary stats
    ):
        self.root_dir = root_dir
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

        # For node assignment after reading parquet
        self.row_to_zcta_assignments = {}

    def __len__(self):
        return len(self.lead_dates)

    def __getitem__(self, idx):
        # Get the date range for the window
        # end_dates[idx] corresponds to yyyymmdd[idx + window - 1]
        dates = self.yyyymmdd[idx:idx + self.window]  # Get the last 'window' dates
        
        # Initialize tensor for this variable across the window
        tensor = torch.full((len(self.nodes), len(self.vars), self.window), fill_value=torch.nan, dtype=torch.float32)

        for var_group_name, var_group in self.var_dict.items():
            temporal_res = var_group["temporal_res"]
            
            for var in var_group["vars"]:
                # Get the index for the variable
                var_index = self.var_to_idx[f"{var_group_name}_{var}"]

                for date_idx, date_str in enumerate(dates):
                    # Adjust date string based on temporal resolution
                    if temporal_res == "yearly":
                        file_date_str = date_str[:4]
                    elif temporal_res == "monthly":
                        file_date_str = date_str[:6]
                    else:  # daily
                        file_date_str = date_str
                    
                    filename = f"{self.root_dir}/{var_group_name}/{var}/{var}__{file_date_str}.parquet"
            
                    # # Read the parquet file
                    if var_group_name not in self.row_to_zcta_assignments:
                        if not os.path.exists(filename):
                            logging.warning(f"File {filename} does not exist. Filling with NaNs.")
                            continue

                        table = pq.read_table(filename, columns=["zcta"]).to_pandas()
                        table["zcta_index"] = table["zcta"].apply(lambda z: self.node_to_idx.get(z, -1))
                        # Filter out rows where zcta is not in node_to_idx
                        row_filter = (table["zcta_index"] != -1).values
                        zcta_index = torch.tensor(table["zcta_index"][row_filter].values, dtype=torch.long)
                        self.row_to_zcta_assignments[var_group_name] = (zcta_index, row_filter)
                    else:
                        zcta_index, row_filter = self.row_to_zcta_assignments[var_group_name]

                    if not os.path.exists(filename):
                        logging.warning(f"File {filename} does not exist. Filling with NaNs.")
                        table = pd.DataFrame(columns=[var])
                    else:
                        table = pq.read_table(filename, columns=[var]).to_pandas()
                        
                    if not table.empty:
                        values = torch.tensor(table[var][row_filter].values, dtype=torch.float32)
                        # apply normalization if stats available
                        mean, std = get_var_summy(self.summary_stats, var_group_name, var)
                        mask = ~torch.isnan(values)
                        values[mask] = (values[mask] - mean) / std
                        tensor[zcta_index, var_index, date_idx] = values

        if self.transform:
            tensor = self.transform(tensor)

        return tensor


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
