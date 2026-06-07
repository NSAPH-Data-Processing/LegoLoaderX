# Lego Dataloader X

**Modular, reproducible data loading for health and environmental research. Built on the Lego Data Model.**

**LegoLoaderX** provides PyTorch-compatible Datasets designed for epidemiological and environmental data applications which are built from a standardized access to a vetted internal data warehouse — the **Lego Data Model** — composed of materialized views across air quality, climate, census, and health data.

This package enables deep learning pipelines to easily and reproducibly access complex structured data via a simple, familiar PyTorch interface.

Use cases:

- Health outcome modeling from environmental exposures  
- Deep learning with time-aligned spatial datasets  
- Demographic covariate modeling  
- Multi-source data fusion for predictive modeling


Key points:

- Exposes a modular Dataset class
- Composable loading of **confounders**, **treatments**, and **outcomes**
- Time-windowed and forecast-oriented batch generation
- Supports configuration-driven loading via YAML
- Built on the Lego Data Model — a structured and vetted multi-domain data warehouse
- DuckDB + PyArrow query backend for fast I/O
- Compatible with PyTorch `DataLoader` and batch processing

## Installation

There are **two audiences**, and they install different things:

**1. Using the library** (you just want to `import legoloaderx` in your own project):

```
pip install git+https://github.com/your-org/LegoLoaderX.git
```

This pulls only the minimal runtime deps the dataloader needs (declared in
`setup.py`: numpy, torch, pandas, pyarrow, duckdb, hydra-core). You do **not** get the
data-processing pipeline dependencies — that's intentional, so downstream users aren't
forced to install heavyweights like geopandas/GDAL, snakemake, or transformers.

**2. Running the pipelines** (building the feature store from the Lego Data Model):
see [Generating the Feature Store](#generating-the-feature-store) — this needs the full,
pinned environment in `requirements.txt`, not just the package.

> **How the env files fit together.** `environment.yaml` creates the conda env and its
> only pip step is `pip install -e .`, i.e. it installs `setup.py`'s **minimal core**.
> `requirements.txt` is the **separate, complete** set for running `src/` (adds
> geopandas, scipy, snakemake, transformers, pytest, matplotlib, … with version pins).
> So building the env is **two steps** — see below.

## Dataloader Architecture

```
Lego Data Model 
+
var_dict
 ├── Confounders   ──> XDataset      ┐
 ├── Treatments    ──> XDataset      |  ──> HealthXDataset
 └── Outcomes      ──> HealthDataset ┘        
```

| Component    | Shape                                  |
|--------------|-----------------------------------------|
| Confounders  | `(n_nodes, n_vars, window)`            |
| Treatments   | `(n_nodes, n_vars, window)`            |
| Outcomes     | `(n_nodes, n_vars, len(horizons), window)` *(or)* `(n_nodes, n_vars, window + delta_t)` |

## The Lego Data Model
The Lego Data Model is a system of standardized and composable data views (or "blocks") for:

- Air pollution (e.g., PM2.5, NO₂)
- Climate indicators (e.g., temperature, drought)
- Demographics and census
- Health outcomes and covars

The Lego Data Model is designed to house datasets that are easy to piece together for epi and environmental studies. In this repository we process Lego Data Model materialized views into a feature store. The feature store allows accelerated loading into ML applications. 

## Generating the Feature Store 

Build the conda environment, then install the pipeline dependencies (two steps —
`environment.yaml` only installs the package's core deps via `-e .`):
```
conda env create -f environment.yaml     # creates the legoloaderX env + editable package (core deps only)
conda activate legoloaderX
pip install -r requirements.txt          # pipeline/dev deps: geopandas, snakemake, pytest, ...
```

> **geopandas / GDAL.** geopandas (used by the synthetic-data scripts in `src/`) needs
> system GDAL/PROJ and can fail to pip-install on a bare machine. If you hit
> `ModuleNotFoundError: geopandas`, install it from conda-forge into the same env:
> `conda install -c conda-forge geopandas`.

Modify the configuration files in config/

Attach the pipeline folder structure using `python src/create_dir_paths`

Then run
```
snakemake --cores 4
```

## Overview of synthetic data generation pipeline

Synthetic denom files are generated using official ZCTA population counts from the U.S. Census.

