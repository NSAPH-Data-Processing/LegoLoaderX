#!/bin/bash
# Stage the post-processed mmap store into node-local /dev/shm (RAM) so the
# DataLoader benchmark / training reads from RAM instead of the shared NFS mount
# (NFS read contention is what stalls many-worker loaders). Run ON the compute
# node. The tarball holds BOTH formats (yearly_mmap_dense + yearly_mmap_sparse)
# and idx2zcta. TARBALL / DST overridable via env. Logs to benchmarking/stage_shm.log.
#
#   bash benchmarking/stage_shm.sh
#   python benchmarking/benchmark.py mode=traintime data_root=/dev/shm/legoloaderx_data
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)   # benchmarking/
REPO=$(dirname "$HERE")
TARBALL=${TARBALL:-/n/dominici_lab/lab/lego_loader_x/legoloaderx_mmaps.tar}
DST=${DST:-/dev/shm/legoloaderx_data}
LOG="$HERE/stage_shm.log"
: > "$LOG"
echo "host=$(hostname) tarball=$TARBALL dst=$DST" >> "$LOG"

rm -rf "$DST"; mkdir -p "$DST"
t0=$(date +%s.%N)
tar -C "$DST" -xf "$TARBALL"
t1=$(date +%s.%N)
NF=$(find "$DST" -type f | wc -l); SZ=$(du -sh "$DST" | cut -f1)
echo "unpacked $NF files ($SZ) in $(echo "$t1-$t0" | bc)s" >> "$LOG"

# The tar stores paths as output/... and health_synthetic/...; the loaders expect
# root_dir to contain covars/ and health/ -> symlink so data_root=$DST works as-is.
ln -sfn "$DST/output" "$DST/covars"
ln -sfn "$DST/health_synthetic" "$DST/health"
echo "symlinks: covars->output, health->health_synthetic" >> "$LOG"
echo DONE >> "$LOG"
cat "$LOG"
