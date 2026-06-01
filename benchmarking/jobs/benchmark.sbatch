#!/bin/bash
#SBATCH --job-name=legoloaderx-benchmark
#SBATCH --output=benchmarking/jobs/benchmark_%j.out
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=12:00:00

# Full-scale HealthXDataset benchmark (format comparison + DataLoader scaling).
# The short interactive run can be done directly on a compute node:
#   python benchmarking/benchmark.py
# Use this SLURM job for the long confidence run (full epoch), e.g.:
#   sbatch benchmarking/jobs/benchmark.sbatch n_samples=200 batch_sizes=[32,64] n_batches=15

set -euo pipefail

source activate healthx 2>/dev/null || conda activate healthx 2>/dev/null || true
cd "${SLURM_SUBMIT_DIR:-.}"

python benchmarking/benchmark.py "$@"
