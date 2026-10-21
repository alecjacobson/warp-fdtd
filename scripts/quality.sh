#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
JAX_PLATFORMS=cpu OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 taskset -c 0-3 .venv/bin/python -m baseline.protocol_checks > results/protocol_checks.log 2>&1
scripts/run.sh forward --spacing-nm 50 --voxel-nm 50 --time-fs 30 --reps 2 --trace \
    --output results/cuda_trace > results/cuda_trace.log 2>&1
scripts/run.sh forward --scene straight --objective modal --exact-interpolation --reps 2 \
    --output results/straight20_calibration > results/straight20_calibration.log 2>&1
# First establish a finite-difference plateau without float32 cancellation.
scripts/run.sh gradient --spacing-nm 50 --voxel-nm 50 --gradient exact --precision float64 \
    --finite-difference --reps 2 --output results/gradient50_exact64 > results/gradient50_exact64.log 2>&1
scripts/run.sh gradient --spacing-nm 50 --voxel-nm 50 --gradient exact \
    --finite-difference --reps 2 --output results/gradient50_exact > results/gradient50_exact.log 2>&1
# Independent quality readout of a deterministic curved guide, before optimization.
scripts/run.sh forward --init ring --beta 50 --objective modal --exact-interpolation \
    --reps 2 --output results/ring20_initial > results/ring20_initial.log 2>&1
scripts/run.sh gradient --spacing-nm 50 --voxel-nm 50 --objective modal --gradient exact \
    --finite-difference --reps 1 --output results/gradient50_modal > results/gradient50_modal.log 2>&1
scripts/run.sh gradient --spacing-nm 50 --voxel-nm 50 --objective modal --gradient adjoint \
    --finite-difference --reps 1 --output results/gradient50_adjoint > results/gradient50_adjoint.log 2>&1
.venv/bin/python -m baseline.compare > results/comparison.log 2>&1
scripts/run.sh optimize --init ring --objective modal --gradient exact --epochs 20 \
    --fixed-beta --beta 50 --exact-interpolation --output results/ring20_optimized > results/ring20_optimized.log 2>&1
bash scripts/evaluate_hann.sh
.venv/bin/python -m baseline.report
