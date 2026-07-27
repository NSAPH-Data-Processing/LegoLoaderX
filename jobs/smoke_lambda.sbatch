#!/bin/bash
#
#SBATCH -J smoke_lambda           # job name
#SBATCH -p gpu_test               # FASRC GPU test partition (fast, short queue)
#SBATCH --gres=gpu:1              # 1 GPU (this run is CPU-only; the GPU just gets you into the test queue)
#SBATCH -c 4                      # cores (smoke is single-process; must be < 8 per GPU on gpu_test)
#SBATCH --mem 32GB                # memory
#SBATCH -t 00:30:00               # walltime (well under the 12h gpu_test cap)
#SBATCH -o logs/slurm/%j.out
#SBATCH -e logs/slurm/%j.err
#
# =============================================================================
# Smoke run: print the DISTRIBUTION and AVERAGES of the DGP rate lambda for ONE
# disease and ONE year -- NO Poisson sampling, NO data written.
#
# It calls src/synthetic_smoke.py, which builds the rate with the SAME function the
# generator and the ground-truth ERC use (expected_rate_grid), so the numbers you see
# (mean, percentiles, %% of cells at the floor, implied count sparsity) are the true
# per-cell rate the outcome would be drawn from.
#
# This run sets the floor low (rate_floor=0.0001) to reveal how far below 0.01 the rate
# actually wants to go -- if very few cells sit at 0.0001, the DGP rarely needs the clamp.
#
# Submit from the repo root:  sbatch jobs/smoke_lambda.sbatch
# (CPU-only: to skip the GPU queue, change '-p gpu_test' to '-p test'/'-p shared' and
#  delete the --gres line.)
# =============================================================================

set -euo pipefail

# ---------- configuration (edit these) ----------
ENV_NAME="/n/home09/jkleutgens/.conda/envs/healthx"   # conda env (activate by full path)
VAR="diabetes"            # which disease/outcome to summarize
YEAR="2010"               # which year (configured covariates exist 2000-2016)
RATE_FLOOR="0.0001"       # lower clamp on lambda for this run
# ------------------------------------------------

# Initialize conda DIRECTLY -- do NOT rely on ~/.bashrc, which in a non-interactive SLURM shell may
# not set up conda (leaving base Mambaforge python on PATH). set +u: the profile scripts touch unset
# vars that would trip 'set -u'.
CONDA_BASE="/n/sw/Mambaforge-23.3.1-1"
set +u
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"
set -u
echo "using python: $(which python)"   # sanity: should be .../envs/${ENV_NAME}/bin/python

# Always run from the repo root so 'src' is importable and data/ paths resolve, no matter where
# 'sbatch' was submitted from.
REPO="/n/home09/jkleutgens/LegoLoaderX"
cd "${REPO}"
export PYTHONPATH="${REPO}"
mkdir -p logs/slurm

PLOT_PATH="outputs/lambda_hist_${VAR}_${YEAR}.png"   # where the histogram PNG lands (repo outputs/, gitignored)

echo "=== smoke run: lambda distribution/averages for ${VAR} ${YEAR} (rate_floor=${RATE_FLOOR}) ==="
/n/home09/jkleutgens/.conda/envs/healthx/bin/python -m src.synthetic_smoke \
    year="${YEAR}" \
    synthetic.var_name="${VAR}" \
    synthetic.poisson_params.rate_floor="${RATE_FLOOR}" \
    synthetic.diagnostics.plot_path="${PLOT_PATH}" \
    hydra.run.dir=.

echo "=== done. text summary above; histogram plot -> ${PLOT_PATH} ==="
