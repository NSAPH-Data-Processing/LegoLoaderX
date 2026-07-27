#!/bin/bash
#
#SBATCH -J synthetic_pipeline      # job name
#SBATCH -p test                    # partition (queue)
#SBATCH -c 64                      # number of cores
#SBATCH --mem 128GB                # memory per job
#SBATCH -t 11:59:00                # time (HH:MM:SS)
#SBATCH -o /n/home09/jkleutgens/LegoLoaderX/logs/slurm/%j.out
#SBATCH -e /n/home09/jkleutgens/LegoLoaderX/logs/slurm/%j.err
#
# End-to-end synthetic dataset generation for ALL diseases x ALL years:
#   Stage 1 (snakefile_synthetic.smk) -> sparse counts + denominators under
#           data/input/lego/medicare_synthetic/ for EVERY disease in
#           conf/synthetic/snakemake.yaml:disease_params  x  EVERY year in :years
#   Stage 2 (snakefile_health.smk) -> dense .npy feature store under data/health/
#
# >>> EDIT THE YEARS in conf/synthetic/snakemake.yaml (the 'years:' list). <<<
# NOTE on range: the causal confounders NO2/O3 (aqdh) only exist through 2016, and
# PM2.5/gridmet/census through 2020 -- so the usable range with the current 5+5 config is
# 2010-2016. Generating a year without its covariates will crash build_extra_rate.
#
# Submit from anywhere:  sbatch jobs/synthetic_pipeline.sbatch
# (Only need the raw counts? Comment out the Stage 2 line at the bottom.)

set -euo pipefail

# ---------- configuration ----------
REPO="/n/home09/jkleutgens/LegoLoaderX"
ENV="/n/home09/jkleutgens/.conda/envs/healthx"     # healthx env (used directly, no 'conda activate')
CORES=64
# -----------------------------------

# Put the healthx env's bin first so BOTH snakemake and the 'python src/...' subprocesses it spawns
# resolve to healthx -- no 'conda activate' (which fails in a non-interactive SLURM shell). Run from
# the repo root so 'src', the snakefiles, and data/ paths resolve regardless of the submit directory.
export PATH="${ENV}/bin:${PATH}"
cd "${REPO}"
export PYTHONPATH="${REPO}"
mkdir -p logs/slurm
echo "using: $(which snakemake) | $(which python)"

# Release any stale snakemake lock from a previous interrupted run.
snakemake -s snakefile_synthetic.smk --unlock || true
snakemake -s snakefile_health.smk    --unlock || true

# Stage 1: generate the synthetic raw inputs (counts + denominators) for all diseases x years.
snakemake -s snakefile_synthetic.smk --cores "${CORES}"

# Stage 2: build the dense .npy feature store from the synthetic inputs.
# (Comment out the next line if you only need the raw sparse counts.)
snakemake -s snakefile_health.smk    --cores "${CORES}"
