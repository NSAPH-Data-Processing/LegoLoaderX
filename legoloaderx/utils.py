import json
import os
import torch
import time
import logging
from tqdm import tqdm
import duckdb
import pandas as pd


# compute the means and standard deviations without storing all the data
def compute_summary(loader, output_dir=None):
    """
    Compute summary statistics (mean, std, nan count) for all variables.
    
    Args:
        loader: DataLoader instance
        output_path: Path to save JSON summary. If None, no file is saved.
        separate_by_group: If True, save separate JSON for each variable group.
                          If False, save single JSON with all variables.
    
    Returns:
        dict: Summary statistics in format:
            {
                "variable_name": {
                    "mean": float,
                    "std": float,
                    "nan_count": int
                },
                ...
            }
    """
    var_dict = loader.dataset.var_dict
    
    # Create mapping from variable name to group
    var_to_group = {}
    for var_group_name, var_group in var_dict.items():
        for var in var_group["vars"]:
            var_to_group[var] = var_group_name
    
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

        # if count == 10:
        #     break  # for testing purposes, limit to 10 batches

    elapsed_time = time.time() - start_time

    # compute means and stds
    means = totals_sum / totals_n
    stds = torch.sqrt(totals_ss / totals_n - means**2)

    # Create summary in nested format by variable group (for organizing by group if needed)
    summary_by_group = {}
    for var_group_name in var_dict.keys():
        summary_by_group[var_group_name] = {}
    
    for var, mean, std, nan_count, total_count in zip(var_lst, means, stds, totals_nan, totals_n):
        var_group = var_to_group[var]
        
        frac_nan = float(nan_count) / (float(nan_count) + float(total_count)) if float(total_count) > 0 else 0
        var_summary = {
            "mean": float(mean),
            "std": float(std),
            "frac_nan": frac_nan
        }
        
        summary_by_group[var_group][var] = var_summary
    
    summary_by_group["_metadata"] = {
        "elapsed_time_seconds": elapsed_time}
    
    # Save to JSON file(s) if output_dir is provided
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)

        # Save single JSON with all variables
        summary_file = os.path.join(output_dir, "summary_statistics.json")
        with open(summary_file, "w") as f:
            json.dump(summary_by_group, f, indent=2)
        print(f"Saved summary statistics to {summary_file}")

    return summary_by_group

def load_summary_stats(stats_source):
    """Load summary statistics from a dict or JSON file.

    Supports two shapes:
    1) Flat mapping: {"group_var": {"mean": .., "std": ..}}
    2) Grouped mapping: {"group": {"var": {"mean": .., "std": ..}}}
    """
    if stats_source is None:
        return None

    stats = stats_source
    if isinstance(stats_source, str):
        if not os.path.exists(stats_source):
            logging.warning(f"Summary stats file not found: {stats_source}")
            return None
        with open(stats_source, "r") as f:
            stats = json.load(f)

    # Normalize keys to allow both grouped and flat lookups
    norm_map = {}
    for var_group, vg_dict in stats.items():
        # if isinstance(val, dict) and "mean" in val and "std" in val:
        #     # flat style
        #     norm_map[key] = val
        if isinstance(vg_dict, dict):
            # grouped style
            for var, val in vg_dict.items():
                if "mean" in val and "std" in val:
                    norm_map[f"{var_group}_{var}"] = val
    return norm_map if norm_map else None

def get_var_summy(summary_stats, var_group_name, var_name):
    """Return (mean, std) if available for a variable, else None."""
    if summary_stats is None:
        return 0, 1  # no normalization

    key_full = f"{var_group_name}_{var_name}"
    entry = summary_stats.get(key_full)
    
    if entry is None:
        return 0, 1 # no normalization
    
    mean = entry.get("mean")
    std = entry.get("std")
    if std is None or std == 0:
        return mean, 1 # no std normalization
    return mean, std


def get_unique_ids(unique_fpath, min_yr, max_yr):
    total_uniq = []
    node_lst_dict = {} 
    
    query = duckdb.query(f"SELECT zcta, year, continental_us FROM '{unique_fpath}/*.parquet'")
    all_df = query.fetchdf().set_index('year')
    all_df = all_df[all_df.continental_us]  # Filter for continental US
    for yr in range(min_yr, max_yr+1):
        
        uniq_ids = all_df.loc[yr]['zcta']
        total_uniq.extend(uniq_ids.tolist())
        node_lst_dict[yr] = uniq_ids.tolist()

    return pd.Series(total_uniq).unique().tolist(), node_lst_dict