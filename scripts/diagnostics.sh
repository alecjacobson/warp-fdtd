#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
scripts/run.sh forward --load results/ring20_optimized/best_params.npz --beta 50 \
    --time-fs 300 --objective modal --modal-window hann --poynting-plane \
    --exact-interpolation --reps 1 --output results/bend_flow20 > results/bend_flow20.log 2>&1
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m baseline.visualize_flow
JAX_PLATFORMS=cuda,cpu OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 XLA_PYTHON_CLIENT_PREALLOCATE=false \
    taskset -c 0-3 .venv/bin/python -m baseline.convergence > results/convergence.log 2>&1
