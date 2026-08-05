#!/bin/bash
#
#SBATCH -J regen_outcomes
#SBATCH -p gpu_test              # partition (queue)
#SBATCH --gres=gpu:1             # NOT used by this job -- numpy/pandas only. The gpu_test
                                 # partition REJECTS jobs that request no GPU ("if you have CPU
                                 # work for this hardware please use test"), so a gres request is
                                 # mandatory to sit here at all. Switch to `-p test` (or
                                 # `-p shared`) and delete this line to stop holding an A100.
#SBATCH -c 8                     # cores
#SBATCH --mem 32GB               # memory -- was 48GB. PAR defaults to 7 and each concurrent
                                 # generation holds a rate grid + a ~4M-row dataframe, so at
                                 # 32GB either raise this back to 48GB or run with PAR=5.
#SBATCH -t 06:00:00              # walltime (HH:MM:SS)
#SBATCH -o /n/home09/jkleutgens/LegoLoaderX/logs/slurm/%j.out
#SBATCH -e /n/home09/jkleutgens/LegoLoaderX/logs/slurm/%j.err
#
# =============================================================================
# STAGE 1: (re)generate the synthetic OUTCOME counts for all 27 diseases x years
# (src/synthetic_health.py). The Poisson rate is
#   max(rate_floor, base + seasonal + geo + beta*PM25 + sum_k gamma_k*std(C_k))
# where PM25 and the 4 environmental terms come from whichever var_groups
# conf/synthetic/config.yaml -> synthetic.{exposure_group,env_group} point at:
#   pm25_ushap + gridmet   = the REAL exposures (current default)
#   synth_exposure (both)  = the generated ones from jobs/generate_exposures.sbatch
# Per-disease poisson_params come from conf/synthetic/snakemake.yaml.
#
# Output: data/input/lego/medicare_synthetic/medpar_outcomes/ccw/zcta_daily/
#             sparse_counts_<disease>_<year>.parquet (+ .manifest.json)
#
# FORCE OVERWRITE: unlike jobs/generate_all.sbatch this does NOT skip existing files
# -- it overwrites in place, so the previous world's counts (~3.7 GB) are gone. Move
# the directory aside first if you still want them. The manifest written beside each
# parquet records which exposure_group produced it, and the ground truth reads that
# back, so the two can never silently disagree. The DENOM is population-only
# (exposure-independent) and is NOT regenerated here.
#
# ORDER: run BEFORE jobs/format_health.sbatch. Only needs
# jobs/generate_exposures.sbatch first when using the synth_exposure world.
# Submit from the repo root:  sbatch jobs/regenerate_outcomes.sbatch
# =============================================================================

set -uo pipefail

# ---------- configuration (edit these) ----------
REPO="/n/home09/jkleutgens/LegoLoaderX"
ENV_NAME="healthx"
# Env-overridable so the same script can run under sbatch OR directly in an interactive
# allocation (`PAR=5 bash jobs/regenerate_outcomes.sbatch`) without editing the file. Nothing
# in the body uses SLURM variables, so it is a valid standalone bash script.
YEARS="${YEARS:-2011 2012 2013 2014 2015 2016 2017 2018 2019 2020}"
PAR="${PAR:-7}"                  # concurrent (disease,year) generations (<= cores, and each
                                 # holds a ~1-2 GB rate grid + dataframe -> keep PAR*2GB < mem)
# ------------------------------------------------

set +u
source ~/.bashrc
conda activate "${ENV_NAME}"
set -u

cd "${REPO}"
# Belt and braces (same as jobs/format_health.sbatch): `conda activate` is a shell FUNCTION
# defined by conda's init block in ~/.bashrc, and that block is commonly skipped in a
# NON-interactive shell. Running this file directly (`bash jobs/regenerate_outcomes.sbatch`,
# nohup, or any pipeline) therefore leaves `python` pointing at base Mambaforge, which has no
# geopandas -- every generation then dies at import and the `|| echo` below turns that into 270
# identical tracebacks and an exit status of 0. Putting the env on PATH works either way.
export PATH="/n/home09/jkleutgens/.conda/envs/${ENV_NAME}/bin:${PATH}"
export PYTHONPATH="${REPO}"
export OMP_NUM_THREADS=1          # 1 thread/process so PAR of them fit the cores
mkdir -p logs/slurm

# Fail fast and loudly rather than 270 times, quietly. The per-generation `|| echo "!! FAILED"`
# below deliberately swallows failures so one bad (disease, year) does not abort the run -- which
# means an environment problem would otherwise look like a successful job with no output.
echo "python: $(which python)"
if ! python -c "import geopandas, hydra, numpy, pandas" 2>/dev/null; then
    echo "FATAL: '${ENV_NAME}' env not active -- python is $(which python)." >&2
    echo "       Expected /n/home09/jkleutgens/.conda/envs/${ENV_NAME}/bin/python" >&2
    exit 1
fi

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
        ( /n/home09/jkleutgens/.conda/envs/healthx/bin/python -m src.synthetic_health year="$y" synthetic.var_name="$var" $ov ${HYDRA_ARGS} \
            || echo "!! FAILED $var $y" ) &
        throttle
    done
    echo "submitted all diseases for $y"
done
wait

echo "=== done. count files present: $(ls ${COUNTS_DIR}/sparse_counts_*.parquet 2>/dev/null | wc -l) ==="
