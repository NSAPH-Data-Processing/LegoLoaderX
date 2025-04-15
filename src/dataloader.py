import time

import hydra
import numpy as np
import torch
import pandas as pd
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import duckdb
import yaml

class VarsHealthxDataset(Dataset):
    def __init__(
        self,
        root_dir,
        var_dict,
        transform=None, # not implemented right now
        min_year = 2000,
        max_year = 2020
    ):
        self.root_dir = root_dir
        self.transform = transform
        self.var_dict = var_dict
        all_dates = pd.date_range(f"{min_year}-01-01", f"{max_year}-12-31", freq="D")
        self.yyyymmdd = [f"{d.year}{d.month:02d}{d.day:02d}"  for d in all_dates]

    def __len__(self):
        return len(self.yyyymmdd)

    def __getitem__(self, idx):
        date_str = self.yyyymmdd[idx]

        layers = []
        
        for var_group in list(self.var_dict.keys()):
            if self.var_dict[var_group]["temporal_res"] == "yearly":
                date_str = date_str[:4]
            elif self.var_dict[var_group]["temporal_res"] == "monthly":
                date_str = date_str[:6]
            for var in self.var_dict[var_group]["vars"]:
                filename = f"{self.root_dir}/{var_group}/{var}__{date_str}.parquet"
                layers.append(duckdb.query(f"SELECT {var} FROM '{filename}'").fetchdf()[var].tolist())

        tensor = torch.FloatTensor(np.stack(layers, axis=0))

        if self.transform:
            tensor = self.transform(tensor)

        return tensor
    
# compute the means and standard deviations without storing all the data
def compute_stats(loader, var_lst):
    # do not sum nans
    totals_sum = torch.zeros(len(var_lst))
    totals_ss = torch.zeros(len(var_lst))
    totals_n = torch.zeros(len(var_lst))

    # also keep track of time
    start_time = time.time()

    input_size = None
    # iterate through
    for batch in tqdm(loader):
        if input_size is None:
            input_size = batch.shape[2]
        totals_n += (~torch.isnan(batch)).sum(dim=(0, 2))
        x = torch.nan_to_num(batch, nan=0.0)
        totals_sum += x.sum(dim=(0, 2))
        totals_ss += (x**2).sum(dim=(0, 2))

    elapsed_time = time.time() - start_time

    # compute means and stds
    means = totals_sum / totals_n
    stds = torch.sqrt(totals_ss / totals_n - means**2)

    # conver to dict with components as keys
    means_dict = {component: float(mean) for component, mean in zip(var_lst, means)}
    stds_dict = {component: float(std) for component, std in zip(var_lst, stds)}

    # save to file
    summary = {"means": means_dict, 
               "stds": stds_dict, 
               "elapsed_time": elapsed_time, 
               "input_grid_size": input_size}
    return(summary)


@hydra.main(config_path="../conf/dataloader", config_name="config", version_base=None)
def main(cfg: DictConfig):
    """This script tests the dataloader and saves aggregate statistics in a summary file."""

    var_dict = {}
    var_lst = []

    # iterate through variable groups and collect names of all variables
    for vg in cfg.var_groups:
        var_dict[vg] = {}
        with open(f"conf/var_group/{vg}.yaml", "r") as f:
            vg_cfg = yaml.safe_load(f)
            # get variable names
            var_dict[vg]["vars"] = vg_cfg["vars"]
            var_lst += vg_cfg["vars"]
            # store spatial and temporal res
            var_dict[vg]["temporal_res"] = vg_cfg["min_temporal_res"]
            var_dict[vg]["spatial_res"] = vg_cfg["min_spatial_res"]
            f.close()

    root_dir = cfg.data_dir

    # initialize dataset
    dataset = VarsHealthxDataset(
        root_dir=root_dir,
        transform=None,
        var_dict=var_dict,
        min_year = cfg.min_year, 
        max_year = cfg.max_year
    )

    # adapt to dataloader
    loader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )

    # test loader ability
    if cfg.verbose:
        compute_stats(loader, var_lst)



if __name__ == "__main__":
    main()
