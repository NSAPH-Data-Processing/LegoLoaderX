#!/bin/bash
#
#SBATCH -J gen_exposures
#SBATCH -p shared                # partition (queue)
#SBATCH -c 8                     # cores
#SBATCH --mem 32GB               # memory
#SBATCH -t 03:00:00              # walltime (HH:MM:SS)
#SBATCH -o /n/home09/jkleutgens/LegoLoaderX/logs/slurm/%j.out
#SBATCH -e /n/home09/jkleutgens/LegoLoaderX/logs/slurm/%j.err
#
# =============================================================================
# STAGE 0 of the semi-synthetic pipeline: generate the 5 SYNTHETIC ENVIRONMENTAL
# EXPOSURES (PM2.5 + tmmx/rmax/pr/vs) as treatment-assignment models with a KNOWN
# dependence on the confounders (src/synthetic_exposure.py).
#
# Output (one dense covar .npy + .meta.json per exposure per year):
#   <exposure_covars_root>/synth_exposure/<var>/<var>__<year>.npy
# where exposure_covars_root = data/covars (the PERSONAL overlay symlink).
#
# The generation parameters (drivers, weights, f_drivers/f_struct, rho/phi, scale)
# live in conf/synthetic/config.yaml -> synthetic.exposure_generation.
#
# ORDER: run this BEFORE synthetic_health.py -- the outcome rate reads these exposure
# files from disk (beta*PM25 + sum gamma*standardized(gridmet)). Regenerating exposures
# after outcomes silently invalidates the ground truth, so regenerate both together.
#
# Submit from the repo root:  sbatch jobs/generate_exposures.sbatch
# =============================================================================

set -uo pipefail

# ---------- configuration (edit these) ----------
REPO="/n/home09/jkleutgens/LegoLoaderX"
ENV_NAME="healthx"
YEARS="2011 2012 2013 2014 2015 2016 2017 2018 2019 2020"   # covariate drivers exist 2011-2020
# ------------------------------------------------

# Conda setup with nounset OFF (system /etc/bashrc references an unset var under set -u).
set +u
source ~/.bashrc
conda activate "${ENV_NAME}"
set -u

cd "${REPO}"
export PYTHONPATH="${REPO}"
mkdir -p logs/slurm

HYDRA_ARGS="hydra.run.dir=. hydra.output_subdir=null"

for y in ${YEARS}; do
    echo "=== generating synthetic exposures for ${y} ==="
    python -m src.synthetic_exposure year="${y}" ${HYDRA_ARGS}
done

echo "=== synthetic exposure generation complete for: ${YEARS} ==="
