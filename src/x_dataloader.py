import time

import hydra
import numpy as np
import torch
import pandas as pd
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import duckdb



class XDataset(Dataset):
    def __init__(
        self,
        root_dir,
        var_dict, #var_dict is structured as {var_group: {"vars": [...], "temporal_res": ...}}
        nodes,  # List of zctas (required)
        window,   # Window size for temporal data (required)
        transform=None, # not implemented right now
        min_year = 2000,
        max_year = 2020
    ):
        self.root_dir = root_dir
        self.transform = transform
        self.var_dict = var_dict
        self.window = window

        # Pull the vars for each var_group in var_dict
        self.vars = [f"{var_group_name}_{var}" for var_group_name, var_group in var_dict.items() for var in var_group["vars"]]
        self.var_to_idx = {var: i for i, var in enumerate(self.vars)}
        
        # Handle nodes (zctas)
        self.nodes = nodes
        self.node_string = ",".join(f"'{node}'" for node in self.nodes)  # For SQL queries
        self.node_to_idx = {node: i for i, node in enumerate(self.nodes)}
        
        all_dates = pd.date_range(f"{min_year}-01-01", f"{max_year}-12-31", freq="D")
        self.yyyymmdd = [f"{d.year}{d.month:02d}{d.day:02d}"  for d in all_dates]
        
        # For windowed data, we need to start from window-1 to have enough history
        self.end_dates = self.yyyymmdd[window-1:]

    def __len__(self):
        return len(self.end_dates)

    def __getitem__(self, idx):
        # Get the date range for the window
        # end_dates[idx] corresponds to yyyymmdd[idx + window - 1]
        dates = self.yyyymmdd[idx:idx + self.window - 1]  # Get the last 'window' dates
        
        # Initialize tensor for this variable across the window
        counts = torch.zeros((len(self.nodes), len(self.vars), self.window))

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
                    
                    # Query with node filtering
                    vals = duckdb.query(f"SELECT * FROM '{filename}' WHERE zcta IN ({self.node_string})").fetchall()
                    
                    # non-vectorized
                    for z, c in vals:
                        # Get the index for the node
                        z_idx = self.node_to_idx[z]
                        # Update the counts tensor
                        counts[z_idx, var_index, date_idx] = int(c)

        if self.transform:
            counts = self.transform(counts)

        return counts
    
# compute the means and standard deviations without storing all the data
def compute_summary(loader):
    var_dict = loader.dataset.var_dict
    var_lst = [var for source in var_dict.values() for var in source['vars']]
    
    totals_nan = torch.zeros(len(var_lst))
    totals_sum = torch.zeros(len(var_lst))
    totals_ss = torch.zeros(len(var_lst))
    totals_n = torch.zeros(len(var_lst))

    # also keep track of time
    start_time = time.time()

    # iterate through
    for batch in tqdm(loader):
        # Shape: (batch_size, n_nodes, n_vars, window)
        # Sum over batch, nodes, and window dimensions
        totals_nan += torch.isnan(batch).sum(dim=(0, 1, 3))
        totals_n += (~torch.isnan(batch)).sum(dim=(0, 1, 3))
        x = torch.nan_to_num(batch, nan=0.0)
        totals_sum += x.sum(dim=(0, 1, 3))
        totals_ss += (x**2).sum(dim=(0, 1, 3))

    elapsed_time = time.time() - start_time

    # compute means and stds
    means = totals_sum / totals_n
    stds = torch.sqrt(totals_ss / totals_n - means**2)

    # conver to dict with components as keys
    nans_dict = {component: float(nan) for component, nan in zip(var_lst, totals_nan)}
    means_dict = {component: float(mean) for component, mean in zip(var_lst, means)}
    stds_dict = {component: float(std) for component, std in zip(var_lst, stds)}

    # save to file
    summary = {
        "nans": nans_dict,
        "means": means_dict, 
        "stds": stds_dict, 
        "elapsed_time": elapsed_time}
    return(summary)


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

    # # Example var_dict
    # var_dict = {
    #     "census": {
    #         "temporal_res": "yearly",
    #         "vars": ["pop", "income", "poverty"],
    #     },
    #     "gridmet": {
    #         "vars": ["rmax", "rmin", "pr"],
    #         "temporal_res": "daily"
    #     }
    # }

    root_dir = cfg.data_dir

    # initialize dataset
    dataset = XDataset(
        root_dir=root_dir,
        transform=None,
        var_dict=var_dict,
        nodes=cfg.nodes if hasattr(cfg, 'nodes') else ["02301", "02148"],  # Default nodes if not specified
        window=cfg.window if hasattr(cfg, 'window') else 7,  # Default window if not specified
        min_year = cfg.min_year, 
        max_year = cfg.max_year
    )

    # adapt to dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        num_workers=12,
        pin_memory=True,
        persistent_workers=True,
    )

    for batch in dataloader:
        print(batch.shape)

    # summary = compute_summary(loader)
    # print(summary["nans"])
    # print(summary["means"])
    # print(summary["stds"])
    # print(f"Elapsed time: {summary['elapsed_time']:.2f} seconds")


if __name__ == "__main__":
    main()
