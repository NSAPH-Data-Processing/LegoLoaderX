#!/bin/bash
#
#SBATCH -p sapphire # partition (queue)
#SBATCH -c 16 # number of cores per job in the array
#SBATCH --mem 32GB # memory per job in the array
#SBATCH -t 0-01:00 # time (D-HH:MM)
#SBATCH --array=2000-2020

min_year=$SLURM_ARRAY_TASK_ID
max_year=$((min_year + 1))

snakemake --cores 12 --config min_year=$min_year max_year=$max_year var_groups="['aqdh']"
