#!/bin/bash
#
#SBATCH -p sapphire # partition (queue)
#SBATCH -c 16 # number of cores per job in the array
#SBATCH --mem 32GB # memory per job in the array
#SBATCH -t 0-01:00 # time (D-HH:MM)
#SBATCH --array=2000-2020
#SBATCH -o logs/output/%A.%a.out
#SBATCH -e logs/err/%A.%a.err

module load python/3.10.9-fasrc01
source activate legoloaderX

export PYTHONPATH="."

min_year=$SLURM_ARRAY_TASK_ID
max_year=$((min_year + 1))

snakemake --unlock
snakemake --cores 12 --config min_year=$min_year max_year=$max_year var_groups="['census']"
