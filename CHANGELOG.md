# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- PyTorch-compatible Dataset classes for health and environmental data
  - `XDataset` for confounders and treatments
  - `HealthDataset` for health outcomes
  - `HealthXDataset` combining confounders, treatments, and outcomes
- Snakemake-based data preprocessing pipelines
  - Main preprocessing pipeline (`snakefile.smk`)
  - Health data preprocessing (`snakefile_health.smk`)
  - Synthetic data generation (`snakefile_synthgen.smk`, `snakefile_synthetic_health.smk`)
- Synthetic data generation for health research
  - ZCTA-level synthetic denominator files based on US Census population counts
  - Synthetic health outcome generation
- Configuration-driven data loading via YAML files
  - Variable dictionaries for flexible feature selection
  - Data path configuration
  - Dataloader configuration
- DuckDB + PyArrow query backend for fast I/O
- Modular architecture supporting composable data loading
  - Time-windowed batch generation
  - Forecast-oriented data structuring
  - Multi-source data fusion capabilities
- Integration with the Lego Data Model
  - Standardized access to air quality data (PM2.5, NO₂)
  - Climate indicators (temperature, drought)
  - Demographics and census data
  - Health outcomes and covariates
- Setup and installation scripts
  - Conda environment configuration
  - Pip-installable package setup
  - Directory structure creation utilities

[Unreleased]: https://github.com/NSAPH-Data-Processing/LegoLoaderX/compare/HEAD...HEAD
