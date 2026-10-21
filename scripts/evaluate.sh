#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
for duration in 200 300 400; do
    scripts/run.sh forward --load results/ring20_optimized/best_params.npz --beta 50 \
        --time-fs "$duration" --objective modal --exact-interpolation --reps 2 \
        --output "results/quality_${duration}fs" > "results/quality_${duration}fs.log" 2>&1
done
scripts/run.sh forward --load results/ring20_optimized/best_params.npz --beta 50 \
    --pml-cells 20 --pad-domain --time-fs 300 --objective modal --exact-interpolation --reps 2 \
    --output results/quality_thicker_pml > results/quality_thicker_pml.log 2>&1
# Preserve physical PML thickness when halving grid spacing.
scripts/run.sh forward --load results/ring20_optimized/best_params.npz --beta 50 \
    --spacing-nm 10 --pml-cells 20 --time-fs 300 --objective modal --exact-interpolation --reps 2 \
    --output results/quality_10nm > results/quality_10nm.log 2>&1
.venv/bin/python -m baseline.quality_checks --window rectangular > results/quality_checks.log 2>&1
# Capacity/stress measurement only: prefix is too short for physical transmission.
scripts/run.sh forward --spacing-nm 5 --voxel-nm 20 --max-steps 1000 --reps 3 \
    --output results/capacity192m > results/capacity192m.log 2>&1
.venv/bin/python -m baseline.report
