import json
from itertools import product
import time

import hydra
import numpy as np
import torch
import torchvision.transforms as transforms
import pandas as pd
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import duckdb
import yaml

# var group
    # varlist
    # temporal_res
    # spatial_res

class VarsHealthxDataset(Dataset):
    def __init__(
        self,
        root_dir,
        var_dict,
        transform=None, # not implemented right now
        years=list(range(2000, 2020)),
    ):
        self.root_dir = root_dir
        self.transform = transform
        self.var_dict = var_dict
        self.yyyymmdd = pd.date_range(f"{min(years)}-01-01", f"{max(years)}-12-31", freq="D")

    def __len__(self):
        return len(self.yyyymmdd)

    def __getitem__(self, idx):
        date_str = self.yyyymmdd[idx]

        # read files for all components from
        layers = []
        
        for var_group in list(self.var_dict.keys()):
            if var_group["temporal_res"] == "yearly":
                date_str = date_str[:4]
            elif var_group["temporal_res"] == "monthly":
                date_str = date_str[:6]
            for var in self.var_dict[var_group]["vars"]:
                # duckdb query
                filename = f"{self.root_dir}/{var_group}__{var}__{date_str}.parquet"
                layers.append(duckdb.query(f"SELECT {var} FROM '{filename}'").fetchdf()[var].tolist())

        tensor = torch.FloatTensor(np.stack(layers, axis=0))

        if self.transform:
            tensor = self.transform(tensor)

        return tensor


@hydra.main(config_path="../conf/dataloader", config_name="config", version_base=None)
def main(cfg: DictConfig):
    """This script tests the dataloader and saves aggregate statistics in a summary file."""

    var_dict = {}
    var_lst = []
    for vg in cfg.vars:
        var_dict[vg] = {}
        with open(f"conf/var_group/{vg}.yaml", "r") as f:
            vg_cfg = yaml.safe_load(f)
            var_dict[vg]["vars"] = vg_cfg["vars"]
            var_lst += vg_cfg["vars"]
            var_dict[vg]["temporal_res"] = vg_cfg["min_temporal_res"]
            var_dict[vg]["spatial_res"] = vg_cfg["min_spatial_res"]
            f.close()

    root_dir = cfg.data_dir
    # transform = None

    dataset = VarsHealthxDataset(
        root_dir=root_dir,
        transform=None,
        var_dict=var_dict,
    )

    loader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )

    # compute the means and standard deviations without storing all the data
    # do not sum nans
    totals_sum = torch.zeros(len(var_lst))
    totals_ss = torch.zeros(len(var_lst))
    totals_n = torch.zeros(len(var_lst))

    # also keep track of time
    start_time = time.time()

    input_size = None
    for batch in tqdm(loader):
        if input_size is None:
            input_size = batch.shape[2:]
        totals_n += (~torch.isnan(batch)).sum(dim=(0, 2, 3))
        x = torch.nan_to_num(batch, nan=0.0)
        totals_sum += x.sum(dim=(0, 2, 3))
        totals_ss += (x**2).sum(dim=(0, 2, 3))

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

    summary_file = "summary.json"

    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)


# def example(
#     root_dir,
#     grid_size=(128, 256),
#     components=["pm25", "no3", "so4", "ss", "nh4", "dust", "bc", "om"],
# ):
#     # load summary stats
#     with open(f"{root_dir}/summary.json", "r") as f:
#         summary = json.load(f)

#     means = [summary["means"][component] for component in components]
#     stds = [summary["stds"][component] for component in components]

#     # two tranforms are needed: first, although the data is already at 0.1 resolution,
#     # it's convenient to further resize to a power of two each dimension
#     # second, normalize the data using the means and stds
#     transform = transforms.Compose(
#         [
#             transforms.Resize(grid_size),
#             transforms.Normalize(mean=means, std=stds),
#         ]
#     )

#     # create torch dataset
#     dataset = ComponentsWashuDataset(
#         root_dir=root_dir,
#         transform=transform,
#         components=components,
#     )

#     # create loader with 4 parallel workers
#     loader = DataLoader(
#         dataset,
#         batch_size=8,
#         shuffle=True,
#         num_workers=4,
#         pin_memory=True,
#         persistent_workers=True,
#     )

#     for batch in loader:
#         # get mask of nans
#         nonnan_mask = (~ torch.isnan(batch)).float()

#         # pad with zero for the computation
#         batch_padded = torch.nan_to_num(batch, nan=0.0)

#         # rest of training logic...
#         # e.g., out = model(batch_padded); loss = F.mse_loss(out * nonnan_mask, target * nonnan_mask)


if __name__ == "__main__":
    main()
    # example("data/input/pm25_components__washu__grid_0_1__dataloader/monthly")
