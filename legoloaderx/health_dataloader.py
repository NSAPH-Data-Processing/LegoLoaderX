import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
import duckdb
import pyarrow.parquet as pq

class HealthDataset(Dataset):
    def __init__(
        self,
        root_dir,
        vars, # List of outcomes (e.g., ["anemia", "asthma"])
        nodes, # List of zctas
        window,
        horizons: list[int] | None = None,
        delta_t: int | None = None,
        min_year: int = 2000,
        max_year: int = 2020,
        min_bene: int = 10,
    ):
        assert horizons is not None or delta_t is not None, "Either horizons or delta_t must be provided."
        assert horizons is None or delta_t is None, "Only one of horizons or delta_t can be provided."
        self.root_dir = root_dir

        self.vars = vars
        self.var_to_idx = {var: i for i, var in enumerate(self.vars)}
        
        self.nodes = nodes
        self.node_string = ",".join(f"'{node}'" for node in self.nodes)  # For SQL queries
        self.node_to_idx = {node: i for i, node in enumerate(self.nodes)}

        all_dates = pd.date_range(f"{min_year}-01-01", f"{max_year}-12-31", freq="D")
        self.yyyymmdd = [f"{d.year}{d.month:02d}{d.day:02d}"  for d in all_dates]

        if horizons:
            self.horizon_mode = "horizons"
            self.horizons = list(sorted(horizons))
            if 0 not in self.horizons:
                self.horizons.insert(0, 0)
            self.horizon_string = ",".join(map(str, self.horizons))  # For SQL queries
            self.horizon_to_idx = {h: i for i, h in enumerate(horizons)}
            self.lead_dates = self.yyyymmdd[window-1:]  # Dates for which we have window history
            self.delta_t = None
        else:
            self.horizon_mode = "delta_t"
            self.delta_t = delta_t
            self.lead_dates = self.yyyymmdd[window-1:-delta_t]  # Dates for which we have window history
            self.horizons = None
            self.horizon_string = None
            self.horizon_to_idx = None

        self.window = window
        self.min_bene = min_bene

    def __len__(self):
        return len(self.lead_dates)
    
    def __getcounts_with_horizons(self, idx):
        counts = torch.zeros((len(self.nodes), len(self.vars), len(self.horizons), self.window), dtype=torch.float32)

        for var in self.var_to_idx:
            # Like this start_date[idx] == self.yyyymmdd[idx + window - 1]
            dates = self.yyyymmdd[idx:idx + self.window]

            # Collect all files for the given variable and date range
            var_index = self.var_to_idx[var]

            for date_idx, day in enumerate(dates):
                file = f"{self.root_dir}/{var}/{var}__{day}.parquet"
                table = pq.read_table(file).to_pandas()

                table["zcta_index"] = table["zcta"].apply(lambda z: self.node_to_idx.get(z, -1))
                table["horizon_index"] = table["horizon"].apply(lambda h: self.horizon_to_idx.get(h, -1))
                table = table[(table["zcta_index"] != -1) & (table["horizon_index"] != -1)]  # Filter out nodes not in self.node_to_idx
                zcta_index = torch.LongTensor(table["zcta_index"].values)
                horizon_index = torch.LongTensor(table["horizon_index"].values)
                n = torch.FloatTensor(table["n"].values)

                # Update the counts tensor
                counts[zcta_index, var_index, horizon_index, date_idx] = n

        return counts

    def __getcounts_with_delta_t(self, idx):
        counts = torch.zeros((len(self.nodes), len(self.vars), self.window + self.delta_t), dtype=torch.float32)

        for var in self.var_to_idx:
            # Like this start_date[idx] == self.yyyymmdd[idx + window - 1]
            dates = self.yyyymmdd[idx:idx + self.window + self.delta_t]

            # Collect all files for the given variable and date range
            var_index = self.var_to_idx[var]

            for date_idx, day in enumerate(dates):
                file = f"{self.root_dir}/{var}/{var}__{day}.parquet"
            
                table = pq.read_table(file).to_pandas()
                table = table[table["horizon"] == 0]

                if not table.empty:
                    table["zcta_index"] = table["zcta"].apply(lambda z: self.node_to_idx.get(z, -1))
                    table = table[table["zcta_index"] != -1]  # Filter out nodes not in self.node_to_idx

                    zcta_index = torch.LongTensor(table["zcta_index"].values)
                    n = torch.FloatTensor(table["n"].values)

                    counts[zcta_index, var_index, date_idx] = n

        return counts

    def __getdenom_and_mask_counts(self, idx, counts):
        dates = self.yyyymmdd[idx:idx + self.window]

        _denom_cache = {}
        denom = torch.zeros((len(self.nodes), self.window), dtype=torch.float32)
        for date_idx, day in enumerate(dates):
            year = day[:4]

            if year not in _denom_cache:
                df = pq.read_table(f"{self.root_dir}/denom/denom__{year}.parquet").to_pandas()
                df.loc[df.n_bene < self.min_bene, "n_bene"] = 0  # Mask out small counts
                df["zcta_index"] = df["zcta"].map(lambda z: self.node_to_idx.get(z, -1))
                df = df[df["zcta_index"] != -1]  # Filter out nodes not in self.node_to_idx
                _denom_cache[year] = df

            denom_df = _denom_cache[year]
            denom_counts = torch.FloatTensor(denom_df.n_bene.values)
            idxs = torch.LongTensor(denom_df.zcta_index.values)
            denom[idxs, date_idx] = denom_counts
            counts[denom_counts == 0, ..., date_idx] = torch.nan  # Mask counts where denom is zero


        return denom


    def __getitem__(self, idx):
        if self.horizon_mode == "horizons":
            counts = self.__getcounts_with_horizons(idx)
        else:
            counts = self.__getcounts_with_delta_t(idx)


        denom = self.__getdenom_and_mask_counts(idx, counts)

        return {
            "outcomes": counts,
            "denom": denom,
        }

def main():
    root_dir = "data/health"
    var_dict = ["anemia", "asthma"]
    # print example  zcta list
    nodes_list = ["00601", "00602"]  # Example zcta codes
    window = 30
    horizons = [0, 7, 14, 30]  # Example horizons
    delta_t = 180
    min_year = 2000
    max_year = 2014

    dataset = HealthDataset(root_dir, var_dict, nodes_list, window, delta_t=delta_t, min_year=min_year, max_year=max_year)
    dataset = HealthDataset(root_dir, var_dict, nodes_list, window, horizons=horizons, min_year=min_year, max_year=max_year)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0)

    for batch in dataloader:
        print(batch['outcomes'].shape)  # Should print the shape of the batch tensor

if __name__ == "__main__":
    main()