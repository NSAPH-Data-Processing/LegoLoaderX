#!/bin/bash
#
#SBATCH -J synthetic_pipeline      # job name
#SBATCH -p test                    # partition (queue)
#SBATCH -c 64                      # number of cores
#SBATCH --mem 128GB                # memory per job
#SBATCH -t 11:59:00                # time (HH:MM:SS)
#SBATCH -o logs/slurm/%j.out
#SBATCH -e logs/slurm/%j.err

# End-to-end synthetic dataset generation:
#   Stage 1 (snakefile_synthetic.smk) generates synthetic Medicare raw inputs under
#           data/input/lego/medicare_synthetic/  (sparse counts + denominators).
#   Stage 2 (snakefile_health.smk) processes those into the dense .npy feature store
#           under data/health/  (use_synthetic: true is set in conf/health/snakemake.yaml).
#
# Submit from the repo root:   sbatch jobs/synthetic_pipeline.sbatch

set -euo pipefail

# Conda env to use (change to legoloaderx if you built the env from environment.yaml).
ENV_NAME="healthx"
CORES=64

source ~/.bashrc
conda activate "${ENV_NAME}"

# Make src/ importable by the hydra scripts.
export PYTHONPATH="."

# Release any stale snakemake lock from a previous interrupted run.
snakemake -s snakefile_synthetic.smk --unlock || true
snakemake -s snakefile_health.smk    --unlock || true

# Stage 1: generate the synthetic raw inputs.
snakemake -s snakefile_synthetic.smk --cores "${CORES}"

# Stage 2: build the dense .npy feature store from the synthetic inputs.
snakemake -s snakefile_health.smk --cores "${CORES}"
