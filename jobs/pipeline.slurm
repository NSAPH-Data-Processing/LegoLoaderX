#!/bin/bash
#
#SBATCH -p shared # partition (queue)
#SBATCH -c 25 # number of cores
#SBATCH --mem 20GB # memory per job
#SBATCH -t 01-10:00 # time (D-HH:MM)
#SBATCH -o logs/output/%j.out
#SBATCH -e logs/err/%j.err

# ===================

module load python/3.10.9-fasrc01
source activate legoloaderX

export PYTHONPATH="."
snakemake -s snakefile.smk --unlock
snakemake -s snakefile.smk --cores 23