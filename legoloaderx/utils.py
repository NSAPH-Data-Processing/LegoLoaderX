import json
import os
import torch
import time
import logging
from tqdm import tqdm


# compute the means and standard deviations without storing all the data
def compute_summary(loader, output_path=None, separate_by_group=False):
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

    elapsed_time = time.time() - start_time

    # compute means and stds
    means = totals_sum / totals_n
    stds = torch.sqrt(totals_ss / totals_n - means**2)

    # Create summary in nested format by variable group (for organizing by group if needed)
    summary_by_group = {}
    for var_group_name in var_dict.keys():
        summary_by_group[var_group_name] = {}
    
    summary_all = {}
    
    for var, mean, std, nan_count in zip(var_lst, means, stds, totals_nan):
        var_group = var_to_group[var]
        
        var_summary = {
            "mean": float(mean),
            "std": float(std),
            "nan_count": int(nan_count)
        }
        
        summary_by_group[var_group][var] = var_summary
        summary_all[var] = var_summary
    
    # Save to JSON file(s) if output_path is provided
    if output_path is not None:
        os.makedirs(output_path, exist_ok=True)
        
        if separate_by_group:
            # Save separate JSON for each variable group
            for var_group_name, group_summary in summary_by_group.items():
                group_file = os.path.join(output_path, f"{var_group_name}_summary.json")
                with open(group_file, "w") as f:
                    json.dump(group_summary, f, indent=2)
                print(f"Saved summary for group '{var_group_name}' to {group_file}")
        else:
            # Save single JSON with all variables
            summary_file = os.path.join(output_path, "summary_statistics.json")
            with open(summary_file, "w") as f:
                json.dump(summary_all, f, indent=2)
            print(f"Saved summary statistics to {summary_file}")
    
    return summary_all

def _load_normalization_stats(self, stats_source):
    """Load normalization statistics from a dict or JSON file.

    Supports two shapes:
    1) Flat mapping: {"group_var": {"mean": .., "std": ..}}
    2) Grouped mapping: {"group": {"var": {"mean": .., "std": ..}}}
    """
    if stats_source is None:
        return None

    stats = stats_source
    if isinstance(stats_source, str):
        if not os.path.exists(stats_source):
            logging.warning(f"Normalization stats file not found: {stats_source}")
            return None
        with open(stats_source, "r") as f:
            stats = json.load(f)

    # Normalize keys to allow both grouped and flat lookups
    norm_map = {}
    for key, val in stats.items():
        if isinstance(val, dict) and "mean" in val and "std" in val:
            # flat style
            norm_map[key] = val
        elif isinstance(val, dict):
            # grouped style
            for inner_k, inner_v in val.items():
                if "mean" in inner_v and "std" in inner_v:
                    norm_map[f"{key}_{inner_k}"] = inner_v
    return norm_map if norm_map else None

def _get_norm(self, var_group_name, var_name):
    """Return (mean, std) if available for a variable, else None."""
    if self.normalization_stats is None:
        return None

    key_full = f"{var_group_name}_{var_name}"
    entry = self.normalization_stats.get(key_full)
    if entry is None:
        # fallback to unprefixed variable name if provided
        entry = self.normalization_stats.get(var_name)
    if entry is None:
        return None
    mean = entry.get("mean")
    std = entry.get("std")
    if std is None or std == 0:
        return None
    return mean, std