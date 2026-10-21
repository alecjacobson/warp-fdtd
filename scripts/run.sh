#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export JAX_PLATFORMS="${JAX_PLATFORMS:-cuda,cpu}"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=4
export MPLBACKEND=Agg
export XLA_PYTHON_CLIENT_PREALLOCATE=false
# Four vCPUs are available by cgroup quota. Limit oversized host thread pools.
exec taskset -c 0-3 .venv/bin/python -m baseline.run "$@"
