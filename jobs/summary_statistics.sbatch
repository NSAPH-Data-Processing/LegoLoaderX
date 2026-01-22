#!/bin/bash
#

#SBATCH -p test # partition (queue)
#SBATCH -c 3 # number of cores
#SBATCH --mem 20GB # memory per job
#SBATCH -t 0-06:00 # time (D-HH:MM)
#SBATCH -o logs/output/%j.out
#SBATCH -e logs/err/%j.err

# ===================

module load python/3.10.9-fasrc01
source activate legoloaderX

export PYTHONPATH="."

python3 legoloaderx/calc_summary_statistics.py