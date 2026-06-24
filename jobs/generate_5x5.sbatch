#!/bin/bash
#
#SBATCH -J synth_5x5            # job name
#SBATCH -p shared               # partition (queue)
#SBATCH -c 8                    # cores
#SBATCH --mem 32GB              # memory
#SBATCH -t 02:00:00             # walltime (HH:MM:SS)
#SBATCH -o logs/slurm/%j.out
#SBATCH -e logs/slurm/%j.err

# Generate semi-synthetic outcome data whose rate depends on 5 environmental EXPOSURES
# (pm25_ushap/PM25 [ERC target], aqdh/no2, aqdh/o3, gridmet/tmmx, gridmet/rmax) and 5 census
# CONFOUNDERS (median_household_income, pop_poverty, pop_over_65_years, pop_black, median_age),
# all with known coefficients set in conf/synthetic/config.yaml. For each year it writes:
#   - the denominator parquet      (synthetic_denom.py)
#   - the disease sparse-counts parquet (synthetic_health.py, using the 5+5 rate terms)
# then computes the ground-truth ERC for one year.
#
# Submit from the repo root:   sbatch jobs/generate_5x5.sbatch

set -euo pipefail

ENV_NAME="healthx"
VAR="diabetes"                  # disease to generate
YEARS="2010 2011 2012 2013 2014"   # all 10 covariates are present for these years
CHECK_YEAR="2012"               # year to validate + plot the ERC for

source ~/.bashrc
conda activate "${ENV_NAME}"
export PYTHONPATH="."
mkdir -p logs/slurm

echo "=== generating ${VAR} for years: ${YEARS} ==="
for y in ${YEARS}; do
    echo "--- year ${y}: denominator ---"
    python -m src.synthetic_denom  year="${y}"
    echo "--- year ${y}: outcome counts (5 exposures + 5 confounders) ---"
    python -m src.synthetic_health year="${y}" synthetic.var_name="${VAR}"
done

echo "=== ground-truth ERC for ${CHECK_YEAR} -> outputs/erc_ground_truth_${VAR}_${CHECK_YEAR}.csv ==="
python -m src.ground_truth_erc year="${CHECK_YEAR}" synthetic.var_name="${VAR}"

echo "=== done. raw synthetic inputs are under data/input/lego/medicare_synthetic/ ==="
