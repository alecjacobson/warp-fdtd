#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
for method in exact adjoint; do
    scripts/run.sh gradient --load results/ring20_optimized/best_params.npz --beta 50 \
        --objective modal --modal-window hann --gradient "$method" --exact-interpolation --reps 2 \
        --output "results/gradient20_hann_${method}" > "results/gradient20_hann_${method}.log" 2>&1
done
.venv/bin/python -m baseline.compare > results/comparison.log 2>&1
scripts/run.sh forward --spacing-nm 5 --voxel-nm 20 --max-steps 1000 --reps 3 \
    --output results/capacity192m > results/capacity192m.log 2>&1
bash scripts/export_references.sh
.venv/bin/python -m baseline.report
