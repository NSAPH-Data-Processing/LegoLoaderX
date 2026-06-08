import hydra
import pandas as pd
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset

from legoloaderx.x_dataloader import XDataset
from legoloaderx.health_dataloader import HealthDataset


class HealthXDataset(Dataset):
    """Composes confounders + treatments (covars root) with outcomes + denom (health root).

    All sub-datasets share one date axis and node set. ``summary_stats`` is forwarded to the
    covariate datasets (outcomes/denom are never normalized); see ``XDataset`` for its
    semantics (path|dict -> load, None -> compute, {} -> off).
    """

    def __init__(
        self,
        root_dir,
        var_dict,   # {"confounders": {...}, "treatments": {...}, "outcomes": {...}}
        nodes=None,
        window=None,
        delta_t=None,
        summary_stats=None,
        min_year=2000,
        max_year=2020,
    ):
        self.root_dir = root_dir
        self.var_dict = var_dict
        self.nodes = nodes
        self.window = window
        self.min_year = min_year
        self.max_year = max_year

        self.outcomes_dataset = HealthDataset(
            root_dir=f"{root_dir}/health",
            var_dict=var_dict["outcomes"],
            nodes=nodes,
            window=window,
            delta_t=delta_t,
            min_year=min_year,
            max_year=max_year,
        )
        self.delta_t = self.outcomes_dataset.delta_t

        self.confounders_dataset = XDataset(
            root_dir=f"{root_dir}/covars",
            var_dict=var_dict["confounders"],
            nodes=nodes,
            window=window,
            summary_stats=summary_stats,
            min_year=min_year,
            max_year=max_year,
        )
        self.treatments_dataset = XDataset(
            root_dir=f"{root_dir}/covars",
            var_dict=var_dict["treatments"],
            nodes=nodes,
            window=window,
            summary_stats=summary_stats,
            min_year=min_year,
            max_year=max_year,
        )

        self.vars = {
            "confounders": self.confounders_dataset.vars,
            "treatments": self.treatments_dataset.vars,
            "outcomes": self.outcomes_dataset.var_dict,
        }
        self.lead_dates = self.outcomes_dataset.lead_dates
        self.yyyymmdd = self.outcomes_dataset.yyyymmdd

    def __len__(self):
        return len(self.outcomes_dataset.lead_dates)

    def __getitem__(self, idx):
        outcomes = self.outcomes_dataset[idx]
        dates = self.yyyymmdd[idx:idx + self.window]
        year = [int(d[:4]) for d in dates]
        month = [int(d[4:6]) for d in dates]
        day = [int(d[6:8]) for d in dates]
        return {
            "confounders": self.confounders_dataset[idx],
            "treatments": self.treatments_dataset[idx],
            "outcomes": outcomes["outcomes"],
            "denom": outcomes["denom"],
            "index": torch.tensor(idx, dtype=torch.long),
            "year": torch.tensor(year, dtype=torch.long),
            "month": torch.tensor(month, dtype=torch.long),
            "day": torch.tensor(day, dtype=torch.long),
        }


@hydra.main(config_path="../conf/dataloader", config_name="config", version_base=None)
def main(cfg: DictConfig):
    var_dict = {
        "confounders": {
            "census": {"temporal_res": "yearly", "vars": ["population", "median_household_income", "pop_poverty"]},
            "climate_types": {"temporal_res": "yearly", "vars": ["Af", "Am", "Aw", "BSh"]},
        },
        "treatments": {
            "gridmet": {"temporal_res": "daily", "vars": ["rmax", "rmin", "pr"]},
        },
        "outcomes": {
            "ccw": {"temporal_res": "daily", "vars": ["anemia", "asthma", "diabetes"]},
        },
    }

    nodes = pd.read_parquet("data/covars/idx2zcta.parquet")["zcta"].tolist()[:50]
    dataset = HealthXDataset(
        root_dir="data",
        var_dict=var_dict,
        nodes=nodes,
        window=cfg.window if hasattr(cfg, 'window') else 7,
        delta_t=cfg.delta_t if hasattr(cfg, 'delta_t') else 7,
        summary_stats={},  # smoke test: skip the full-store compute
        min_year=cfg.min_year,
        max_year=cfg.max_year,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)
    print({k: tuple(v.shape) for k, v in next(iter(loader)).items()})


if __name__ == "__main__":
    main()
