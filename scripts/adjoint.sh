#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
# Check the newer frequency-adjoint implementation on the objective it supports.
scripts/run.sh gradient --spacing-nm 50 --voxel-nm 50 --objective modal --gradient adjoint \
    --finite-difference --reps 2 --output results/gradient50_adjoint > results/gradient50_adjoint.log 2>&1
for method in exact adjoint; do
    scripts/run.sh gradient --load results/ring20_optimized/best_params.npz --beta 50 \
        --objective modal --gradient "$method" --exact-interpolation --reps 2 \
        --output "results/gradient20_modal_${method}" > "results/gradient20_modal_${method}.log" 2>&1
done
for backend in cpu gpu; do
    if [[ "$backend" == cpu ]]; then export JAX_PLATFORMS=cpu; else export JAX_PLATFORMS=cuda,cpu; fi
    scripts/run.sh forward --backend "$backend" --spacing-nm 50 --voxel-nm 50 --reps 5 --save-full-fields \
        --output "results/forward50_${backend}" > "results/forward50_${backend}.log" 2>&1
done
.venv/bin/python -m baseline.compare > results/comparison.log 2>&1
.venv/bin/python -m baseline.report
