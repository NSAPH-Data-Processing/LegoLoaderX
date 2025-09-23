#!/bin/bash
#
#SBATCH -p test # partition (queue)
#SBATCH -c 65 # number of cores
#SBATCH --mem 128GB # memory per job
#SBATCH -t 11:59:00 # time (HH:MM:SS)
#SBATCH -o logs/slurm/%j.out
#SBATCH -e logs/slurm/%j.err

# Activate conda environment
conda activate legoloaderx

# Set PYTHONPATH to current directory
export PYTHONPATH="."

# Run snakemake with synthetic health data pipeline
snakemake -s snakefile_health.smk --config synthetic=true horizons=[] -c 64
