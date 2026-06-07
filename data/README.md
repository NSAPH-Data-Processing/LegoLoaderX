## Data setup

Set up the `data/` tree by running the helper script — it creates the needed
folders and symlinks from `conf/datapaths`:

```bash
python src/create_dir_paths.py datapaths=datapaths_cannon   # or datapaths_local / datapaths_fasse
```

Pick the `datapaths` config that matches your host. That's it.
