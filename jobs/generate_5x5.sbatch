#!/bin/bash
#
#SBATCH -J synth_data           # job name
#SBATCH -p shared               # partition (queue)
#SBATCH -c 8                    # cores
#SBATCH --mem 32GB              # memory
#SBATCH -t 02:00:00             # walltime (HH:MM:SS)
#SBATCH -o logs/slurm/%j.out
#SBATCH -e logs/slurm/%j.err
#
# =============================================================================
# Generate the semi-synthetic outcome data for the ERC validation pipeline.
#
# WHAT IT PRODUCES, per year (both REQUIRED for usable health data, since a
# health rate = counts / denominator):
#   1. denominator parquet   <- synthetic_denom.py   (population at risk per ZCTA)
#   2. disease counts parquet <- synthetic_health.py  (outcome; the Poisson rate is
#        base + season + geo + beta*PM2.5 + sum_k gamma_k*confounder_k, with the
#        coefficients set in conf/synthetic/config.yaml)
#
# WHICH PARTS DO YOU NEED?
#   * Minimum: ONE year of the disease you care about  -> set YEARS to a single year.
#   * More train/eval data: list more years (all configured covariates exist 2000-2016).
#   * Ground-truth ERC ("answer key"): only needed when you want to SCORE a model.
#     Toggle it with RUN_GROUND_TRUTH below; skip it if you just want the data.
#
# Submit from the repo root:  sbatch jobs/generate_5x5.sbatch
# =============================================================================

set -euo pipefail

# ---------- configuration (edit these) ----------
ENV_NAME="healthx"                  # conda env to use (the repo's environment.yaml builds 'legoloaderx')
VAR="diabetes"                      # which disease/outcome to generate
YEARS="2010 2011"                   # years to generate (configured covariates exist 2000-2016)
RUN_GROUND_TRUTH=true               # also compute the ground-truth ERC?  true | false
GT_YEAR="2010"                      # year to compute the ground-truth ERC for (if enabled)
ERC_OUT="/n/dominici_lab/lab/lego_loader_x/jkleutgens/erc_outputs"  # ERC output dir (lab storage, not home)
# ------------------------------------------------

source ~/.bashrc
conda activate "${ENV_NAME}"
export PYTHONPATH="."
mkdir -p logs/slurm

# ---------- generate the data (denominator + outcome counts) ----------
for y in ${YEARS}; do
    echo "=== year ${y}: denominator (population at risk) ==="
    python -m src.synthetic_denom  year="${y}"

    echo "=== year ${y}: outcome counts for ${VAR} (causal rate from config) ==="
    python -m src.synthetic_health year="${y}" synthetic.var_name="${VAR}"
done

# ---------- optional: the ground-truth ERC (answer key for model validation) ----------
if [ "${RUN_GROUND_TRUTH}" = true ]; then
    echo "=== ground-truth ERC for ${VAR} ${GT_YEAR} -> ${ERC_OUT}/erc_ground_truth_${VAR}_${GT_YEAR}.csv ==="
    python -m src.ground_truth_erc year="${GT_YEAR}" synthetic.var_name="${VAR}" \
        synthetic.erc.output_dir="${ERC_OUT}"
fi

echo "=== done. synthetic inputs are under data/input/lego/medicare_synthetic/ ==="
