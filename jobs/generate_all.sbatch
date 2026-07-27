#!/bin/bash
#
#SBATCH -J gen_all
#SBATCH -p gpu_test
#SBATCH --gres=gpu:1              # required by gpu_test; this job is CPU-only, the GPU sits idle
#SBATCH -c 7
#SBATCH --mem 16GB
#SBATCH -t 11:00:00              # gpu_test allows up to 12h; 8h is plenty for 27 diseases x 11 years
#SBATCH -o /n/home09/jkleutgens/LegoLoaderX/logs/slurm/%j.out
#SBATCH -e /n/home09/jkleutgens/LegoLoaderX/logs/slurm/%j.err
#
# =============================================================================
# Generate synthetic sparse COUNTS + DENOMINATORS for ALL diseases x YEARS,
# WITHOUT snakemake (snakemake isn't installed in the healthx env). The disease
# list and each disease's poisson_params are read from conf/synthetic/snakemake.yaml
# -- the SAME source the snakefile uses -- so nothing is hard-coded here.
#
# Output:
#   data/input/lego/medicare_synthetic/medpar_outcomes/ccw/zcta_daily/
#       sparse_counts_<disease>_<year>.parquet   (+ a .manifest.json each)
#   data/input/lego/medicare_synthetic/mbsf_medpar_denom/zcta_yearly/
#       counts_<year>.parquet
#
# This does NOT build the Stage-2 dense .npy feature store (that path needs snakemake).
# Re-runnable: already-generated (disease, year) outputs are skipped.
#
# >>> EDIT the YEARS line below. With the current aqdh-free 5+12 config the range is
#     2010-2020 (PM2.5/gridmet/census/climate all run through 2020; nothing after 2020).
#     A year without its covariates crashes build_extra_rate.
#
# Submit from anywhere:  sbatch jobs/generate_all.sbatch
# =============================================================================

set -uo pipefail

# ---------- configuration (edit these) ----------
REPO="/n/home09/jkleutgens/LegoLoaderX"
PYBIN="/n/home09/jkleutgens/.conda/envs/healthx/bin/python"
YEARS="2010 2011 2012 2013 2014 2015 2016 2017 2018 2019 2020"   # <-- EDIT (max 2020 with the aqdh-free 5+12 config)
PAR=6                                         # concurrent generations (<= 7 cores; fits 16GB)
# ------------------------------------------------

cd "${REPO}"
export PYTHONPATH="${REPO}"
export OMP_NUM_THREADS=1          # 1 thread/process so PAR of them fit the 7 cores
mkdir -p logs/slurm

HYDRA_ARGS="hydra.run.dir=. hydra.output_subdir=null"   # no per-run .hydra dir (avoids parallel clashes)
COUNTS_DIR="data/input/lego/medicare_synthetic/medpar_outcomes/ccw/zcta_daily"
DENOM_DIR="data/input/lego/medicare_synthetic/mbsf_medpar_denom/zcta_yearly"

# disease -> per-disease hydra overrides, read from the YAML (single source of truth)
mapfile -t DISEASE_LINES < <("${PYBIN}" - <<'PY'
import yaml
d = yaml.safe_load(open("conf/synthetic/snakemake.yaml"))["disease_params"]
for var, params in d.items():
    ov = " ".join(f"synthetic.poisson_params.{k}={v}" for k, v in (params or {}).items())
    print(f"{var}\t{ov}")
PY
)

echo "python : ${PYBIN}"
echo "years  : ${YEARS}"
echo "disease: ${#DISEASE_LINES[@]} diseases | parallel: ${PAR}"

throttle() { while (( $(jobs -r | wc -l) >= PAR )); do wait -n; done; }

for y in ${YEARS}; do
    # denominator (population at risk) for this year
    if [ -f "${DENOM_DIR}/counts_${y}.parquet" ]; then
        echo "skip denom ${y} (exists)"
    else
        ( "${PYBIN}" -m src.synthetic_denom year="${y}" ${HYDRA_ARGS} \
            || echo "!! FAILED denom ${y}" ) &
        throttle
    fi

    # counts for every disease
    for line in "${DISEASE_LINES[@]}"; do
        var="${line%%$'\t'*}"; ov="${line#*$'\t'}"
        if [ -f "${COUNTS_DIR}/sparse_counts_${var}_${y}.parquet" ]; then
            echo "skip ${var} ${y} (exists)"; continue
        fi
        ( "${PYBIN}" -m src.synthetic_health year="${y}" synthetic.var_name="${var}" ${ov} ${HYDRA_ARGS} \
            || echo "!! FAILED ${var} ${y}" ) &
        throttle
    done
done
wait

echo "=== done. sparse_counts files now present: $(ls "${COUNTS_DIR}"/sparse_counts_*.parquet 2>/dev/null | wc -l) ==="
