#!/bin/bash
#
#SBATCH -J format_health
#SBATCH -p gpu_test              # partition (queue)
#SBATCH --gres=gpu:1             # NOT used by this job -- snakemake/numpy/pandas only. The
                                 # gpu_test partition REJECTS jobs that request no GPU
                                 # ("if you have CPU work for this hardware please use test"),
                                 # so a gres request is mandatory to sit here at all. Switch to
                                 # `-p test` (or `-p shared`) and delete this line to stop
                                 # holding an A100 slice for CPU work.
#SBATCH -c 8                     # cores
#SBATCH --mem 32GB               # memory
#SBATCH -t 02:00:00              # walltime (HH:MM:SS)
#SBATCH -o /n/home09/jkleutgens/LegoLoaderX/logs/slurm/%j.out
#SBATCH -e /n/home09/jkleutgens/LegoLoaderX/logs/slurm/%j.err
#
# =============================================================================
# STAGE 2: format the synthetic sparse counts + denominators into the dense
# per-(var, year) .npy store the dataloader reads (snakefile_health.smk):
#   data/health/ccw/<var>/<var>__<year>.npy     (n_days, n_zctas) int16
#   data/health/denom/denom__<year>.npy         (n_zctas,)        float32
#   data/health/idx2zcta.parquet                (row order, matches data/covars)
#
# Reads use_synthetic:true -> lego/medicare_synthetic. Years + disease list come
# from conf/health/snakemake.yaml (currently 2011-2020, 27 diseases). data/health is
# the PERSONAL overlay symlink (see docs/synthetic_exposure.tex).
#
# NOTE: snakemake is NOT in the repo's intended env; it is pip-installed in `healthx`
# (7.32.4 + pulp<2.8). We prepend the env to PATH so both `snakemake` and the bare
# `python` in the snakefile's shell rules resolve to healthx.
#
# ORDER: run AFTER jobs/regenerate_outcomes.sbatch.
# Submit from the repo root:  sbatch jobs/format_health.sbatch
# =============================================================================

set -uo pipefail

# ---------- configuration ----------
REPO="/n/home09/jkleutgens/LegoLoaderX"
ENV_NAME="healthx"
# -----------------------------------

set +u
source ~/.bashrc
conda activate "${ENV_NAME}"
set -u

cd "${REPO}"
export PATH="/n/home09/jkleutgens/.conda/envs/${ENV_NAME}/bin:${PATH}"
export PYTHONPATH="${REPO}"
mkdir -p logs/slurm logs

echo "snakemake: $(which snakemake) ($(snakemake --version))"
echo "python   : $(which python)"

# --forcerun preprocess_health: the sparse-count parquets are read via a glob INSIDE
# preprocessing_health.py and are NOT declared snakemake inputs (only idx2zcta.parquet is), so a
# plain run would report "nothing to do" and skip rebuilding when only the counts changed. Forcing
# the rule guarantees the dense .npy store reflects the latest regenerated counts. idx2zcta + denom
# are left alone (denom is exposure-independent; idx2zcta already matches data/covars).
# --keep-going: one bad (var,year) doesn't halt the rest; --rerun-incomplete: recover partials.
snakemake -s snakefile_health.smk --cores 8 --forcerun preprocess_health --keep-going --rerun-incomplete

echo "=== done. health outputs: ccw=$(find data/health/ccw -name '*.npy' | wc -l) denom=$(find data/health/denom -name '*.npy' | wc -l) ==="
