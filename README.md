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

Install directly from GitHub:

```
pip install git+https://github.com/your-org/LegoLoaderX.git
```

Or for development:

```
git clone https://github.com/your-org/LegoLoaderX.git
cd LegoLoaderX
pip install -e .
```

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

Build the conda environment
```
conda env create -f environment.yaml
```

Modify the configuration files in config/

Attach the pipeline folder structure using `python src/create_dir_paths`

Then run
```
snakemake --cores 4
```

## Overview of synthetic data generation pipeline

Synthetic denom files are generated using official ZCTA population counts from the U.S. Census.

