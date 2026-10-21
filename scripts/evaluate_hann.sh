#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
scripts/run.sh forward --scene straight --objective modal --modal-window hann \
    --exact-interpolation --reps 1 --output results/hann_straight20 > results/hann_straight20.log 2>&1
for duration in 200 300 400; do
    scripts/run.sh forward --load results/ring20_optimized/best_params.npz --beta 50 \
        --time-fs "$duration" --objective modal --modal-window hann --exact-interpolation --reps 1 \
        --output "results/hann_${duration}fs" > "results/hann_${duration}fs.log" 2>&1
done
scripts/run.sh forward --load results/ring20_optimized/best_params.npz --beta 50 \
    --pml-cells 20 --pad-domain --time-fs 300 --objective modal --modal-window hann \
    --exact-interpolation --reps 1 --output results/hann_thicker_pml > results/hann_thicker_pml.log 2>&1
scripts/run.sh forward --load results/ring20_optimized/best_params.npz --beta 50 \
    --spacing-nm 10 --pml-cells 20 --time-fs 300 --objective modal --modal-window hann \
    --exact-interpolation --reps 1 --output results/hann_10nm > results/hann_10nm.log 2>&1
.venv/bin/python -m baseline.quality_checks --window hann > results/quality_checks.log 2>&1
.venv/bin/python -m baseline.report
