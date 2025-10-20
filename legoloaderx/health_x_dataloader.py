import torch
from torch.utils.data import DataLoader, Dataset
from legoloaderx.x_dataloader import XDataset
from legoloaderx.health_dataloader import HealthDataset
import hydra
from omegaconf import DictConfig


class HealthXDataset(Dataset):
    def __init__(
            self, 
            root_dir, 
            var_dict, 
            var_type=None,
            nodes=None, 
            window=None, 
            horizons=None, 
            delta_t=None, 
            min_year=2000, 
            max_year=2020):

        self.root_dir = root_dir
        self.var_dict = var_dict
        self.nodes = nodes
        self.window = window
        self.min_year = min_year
        self.max_year = max_year

        self.outcomes_dataset = HealthDataset(
            root_dir=f"{self.root_dir}/health",
            var_dict=self.var_dict["outcomes"],
            nodes=self.nodes,  # Use nodes_list directly
            window=self.window,
            horizons=horizons,
            delta_t=delta_t,
            min_year=self.min_year,
            max_year=self.max_year
        )
        self.horizons = self.outcomes_dataset.horizons
        self.delta_t = self.outcomes_dataset.delta_t
       
        self.confounders_dataset = XDataset(
            root_dir=f"{self.root_dir}/covars",
            var_dict=self.var_dict["confounders"],
            nodes=self.nodes,  # List of zctas or other nodes
            window=self.window,
            min_year=self.min_year,
            max_year=self.max_year
        )
        self.treatments_dataset = XDataset(
            root_dir=f"{self.root_dir}/covars",
            var_dict=self.var_dict["treatments"],
            nodes=self.nodes,  # List of zctas or other nodes
            window=self.window,
            min_year=self.min_year,
            max_year=self.max_year
        )

        self.vars = {
            "confounders": self.confounders_dataset.vars,
            "treatments": self.treatments_dataset.vars,
            "outcomes": self.outcomes_dataset.var_dict
        }

        self.lead_dates = self.outcomes_dataset.lead_dates
        self.yyyymmdd = self.outcomes_dataset.yyyymmdd

    def __len__(self):
        return len(self.outcomes_dataset.lead_dates)
    
    def __getitem__(self, idx):
        confounders = self.confounders_dataset[idx]
        treatments = self.treatments_dataset[idx]
        outcomes = self.outcomes_dataset[idx]

        # extract year, month, date for each date in the window
        dates = self.yyyymmdd[idx:idx + self.window]
        year = [int(date[:4]) for date in dates]
        month = [int(date[4:6]) for date in dates]
        day = [int(date[6:8]) for date in dates]

        return {
            "confounders": confounders,
            "treatments": treatments,
            "outcomes": outcomes["outcomes"],
            "denom": outcomes["denom"],
            "index": torch.tensor(idx, dtype=torch.long),
            "year": torch.tensor(year, dtype=torch.long),
            "month": torch.tensor(month, dtype=torch.long),
            "day": torch.tensor(day, dtype=torch.long)
        }

@hydra.main(config_path="../conf/dataloader", config_name="config", version_base=None)
def main(cfg: DictConfig):
    
    var_dict = {}

    var_dict = {
        "confounders": {
            "census": {
                "temporal_res": "yearly",
                "vars": ["population", "median_household_income", "pop_poverty"],
            },
            "climate_types": {
                "vars": ["Af", "Am", "Aw", "BSh"],
                "temporal_res": "yearly"
            }
        },
        "treatments": {
            "gridmet": {
                "vars": ["rmax", "rmin", "pr"],
                "temporal_res": "daily"
            }
        },
        "outcomes": {
            "ccw": {
                "vars": ["anemia", "asthma", "diabetes"],
                "temporal_res": "daily",   # only daily is supported for outcomes for now
            },
        },
    }

    root_dir = "data/"

    # initialize dataset
    dataset = HealthXDataset(
        root_dir=root_dir,
        var_dict=var_dict,
        nodes=cfg.nodes if hasattr(cfg, 'nodes') else ["02301", "02148"],  # Default nodes if not specified
        window=cfg.window if hasattr(cfg, 'window') else 7,  # Default window if not specified
        delta_t=cfg.delta_t if hasattr(cfg, 'delta_t') else 7,  # Default delta_t if not specified
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

    for batch in dataloader:
        print({k: v.shape for k, v in batch.items()})  # Print shapes of each item in the batch

if __name__ == "__main__":
    main()

