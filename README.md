# LegoLoaderX

## Setup

### Conda setup
Make sure you have the necessary conda environment installed.
```
conda env create -f requirements.yaml
```

### Datapaths and symlinks
The default datapaths and associated symlinks are stored in `conf/datapaths/datapaths_cannon.yaml`. To create the necessary symlinks:

```
python3 src/create_dir_paths.p
```

