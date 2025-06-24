def __getitem__(self, idx):
    date = self.date_keys[idx]
    files = self.date_to_files[date]

    tensor = torch.zeros((len(self.icd_codes), len(self.zcta_list), len(self.horizons)), dtype=torch.float32)

    for icd, file in files.items():
        df = pd.read_parquet(file)

        # Filter valid zcta and horizon entries
        df = df[df['zcta'].isin(self.zcta_to_idx) & df['horizon'].isin(self.horizon_to_idx)]
        if df.empty:
            continue

        icd_idx = self.icd_to_idx[icd]
        zcta_idx = df['zcta'].map(self.zcta_to_idx).values
        horizon_idx = df['horizon'].map(self.horizon_to_idx).values
        counts = df['n'].values

        i = torch.full((len(df),), icd_idx, dtype=torch.long)
        j = torch.tensor(zcta_idx, dtype=torch.long)
        k = torch.tensor(horizon_idx, dtype=torch.long)
        v = torch.tensor(counts, dtype=torch.float32)

        tensor[i, j, k] = v  # ✅ idiomatic and readable

    return tensor
