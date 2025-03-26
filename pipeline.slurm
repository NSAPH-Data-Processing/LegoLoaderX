#!/bin/bash
#
#SBATCH -p shared,test # partition (queue)
#SBATCH -c 6 # number of cores
#SBATCH --mem 30GB # memory per job
#SBATCH -t 0-06:00 # time (D-HH:MM)
#SBATCH -o logs/output/%j.out
#SBATCH -e logs/err/%j.err

# ===================

module load python/3.10.9-fasrc01
source activate legoloaderX

export PYTHONPATH="."
snakemake --unlock
snakemake --cores 4