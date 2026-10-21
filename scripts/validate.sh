#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export JAX_PLATFORMS=cuda,cpu
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 MPLBACKEND=Agg
export XLA_PYTHON_CLIENT_PREALLOCATE=false
taskset -c 0-3 .venv/bin/python -m baseline.physics > results/physics_gpu.log 2>&1
for method in exact paper compressed checkpointed; do
    scripts/run.sh gradient --spacing-nm 50 --voxel-nm 50 --gradient "$method" \
        --finite-difference --reps 2 --output "results/gradient50_${method}" > "results/gradient50_${method}.log" 2>&1
done
# Execute GPU and CPU timings serially: no benchmark process competes with another.
for backend in gpu cpu; do
    if [[ "$backend" == cpu ]]; then export JAX_PLATFORMS=cpu; else export JAX_PLATFORMS=cuda,cpu; fi
    scripts/run.sh forward --backend "$backend" --spacing-nm 50 --voxel-nm 50 --reps 5 --save-full-fields \
        --output "results/forward50_${backend}" > "results/forward50_${backend}.log" 2>&1
done
export JAX_PLATFORMS=cuda,cpu
for dx in 25 20 10; do
    voxel="$dx"
    if [[ "$dx" == 10 ]]; then voxel=20; fi
    scripts/run.sh forward --spacing-nm "$dx" --voxel-nm "$voxel" --reps 5 \
        --output "results/forward${dx}_gpu" > "results/forward${dx}_gpu.log" 2>&1
done
