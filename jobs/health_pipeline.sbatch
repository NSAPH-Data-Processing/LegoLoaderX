#!/bin/bash
#
#SBATCH -p fasse # partition (queue)
#SBATCH -c 25 # number of cores
#SBATCH --mem 20GB # memory per job
#SBATCH -t 01-30:00 # time (D-HH:MM)
#SBATCH -o logs/output/%j.out
#SBATCH -e logs/err/%j.err

snakemake -s Snakefile_health.smk --cores 23