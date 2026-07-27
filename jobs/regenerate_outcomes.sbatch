#!/bin/bash
#
#SBATCH -J regen_outcomes
#SBATCH -p shared                # partition (queue)
#SBATCH -c 8                     # cores
#SBATCH --mem 48GB               # memory
#SBATCH -t 06:00:00              # walltime (HH:MM:SS)
#SBATCH -o /n/home09/jkleutgens/LegoLoaderX/logs/slurm/%j.out
#SBATCH -e /n/home09/jkleutgens/LegoLoaderX/logs/slurm/%j.err
#
# =============================================================================
# STAGE 1: (re)generate the synthetic OUTCOME counts for all 27 diseases x years
# from the synthetic exposures (src/synthetic_health.py). The Poisson rate is
#   max(rate_floor, base + seasonal + geo + beta*PM25 + sum_k gamma_k*std(C_k))
# where PM25 and the 4 gridmet confounders are now the SYNTHETIC exposures written
# by Stage 0. Per-disease poisson_params come from conf/synthetic/snakemake.yaml.
#
# Output: data/input/lego/medicare_synthetic/medpar_outcomes/ccw/zcta_daily/
#             sparse_counts_<disease>_<year>.parquet (+ .manifest.json)
#
# FORCE OVERWRITE: unlike jobs/generate_all.sbatch this does NOT skip existing
# files -- it must overwrite counts previously built from real PM2.5. The DENOM is
# population-only (exposure-independent) and is NOT regenerated here.
#
# ORDER: run AFTER jobs/generate_exposures.sbatch and BEFORE jobs/format_health.sbatch.
# Submit from the repo root:  sbatch jobs/regenerate_outcomes.sbatch
# =============================================================================

set -uo pipefail

# ---------- configuration (edit these) ----------
REPO="/n/home09/jkleutgens/LegoLoaderX"
ENV_NAME="healthx"
YEARS="2011 2012 2013 2014 2015 2016 2017 2018 2019 2020"
PAR=7                            # concurrent (disease,year) generations (<= cores)
# ------------------------------------------------

set +u
source ~/.bashrc
conda activate "${ENV_NAME}"
set -u

cd "${REPO}"
export PYTHONPATH="${REPO}"
export OMP_NUM_THREADS=1          # 1 thread/process so PAR of them fit the cores
mkdir -p logs/slurm

HYDRA_ARGS="hydra.run.dir=. hydra.output_subdir=null"
COUNTS_DIR="data/input/lego/medicare_synthetic/medpar_outcomes/ccw/zcta_daily"

# disease -> per-disease poisson_params overrides, read from the YAML (single source of truth)
mapfile -t DISEASE_LINES < <(python - <<'PY'
import yaml
d = yaml.safe_load(open("conf/synthetic/snakemake.yaml"))["disease_params"]
for var, params in d.items():
    ov = " ".join(f"synthetic.poisson_params.{k}={v}" for k, v in (params or {}).items())
    print(f"{var}\t{ov}")
PY
)
echo "diseases: ${#DISEASE_LINES[@]} | years: ${YEARS} | PAR=${PAR}"

throttle() { while (( $(jobs -r | wc -l) >= PAR )); do wait -n; done; }
for y in ${YEARS}; do
    for line in "${DISEASE_LINES[@]}"; do
        var="${line%%$'\t'*}"; ov="${line#*$'\t'}"
        ( python -m src.synthetic_health year="$y" synthetic.var_name="$var" $ov ${HYDRA_ARGS} \
            || echo "!! FAILED $var $y" ) &
        throttle
    done
    echo "submitted all diseases for $y"
done
wait

echo "=== done. count files present: $(ls ${COUNTS_DIR}/sparse_counts_*.parquet 2>/dev/null | wc -l) ==="
