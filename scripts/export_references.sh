#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
cases=("$@")
if [[ ${#cases[@]} == 0 ]]; then
    cases=(forward50_gpu forward10_gpu hann_10nm gradient20_hann_exact gradient20_hann_adjoint)
fi
for run in "${cases[@]}"; do
    JAX_PLATFORMS=cuda,cpu OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 XLA_PYTHON_CLIENT_PREALLOCATE=false \
        taskset -c 0-3 .venv/bin/python -m baseline.export_case --run "results/$run" \
        --output "results/warp_reference/$run" > "results/export_$run.log" 2>&1
done
