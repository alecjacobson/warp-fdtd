#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p results
bash scripts/bootstrap.sh
JAX_PLATFORMS=cpu OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 taskset -c 0-3 .venv/bin/python -m pytest \
    upstream/fdtdx/tests/simulation/fdtd/test_reversible_recorder_gradient.py \
    upstream/fdtdx/tests/simulation/fdtd/test_segmented_reversible.py::TestSegmentedGradient \
    --junitxml=results/gradient_tests.xml > results/gradient_tests.log 2>&1
bash scripts/validate.sh
scripts/run.sh gradient --gradient exact --reps 3 --output results/gradient20_exact > results/gradient20_exact.log 2>&1
bash scripts/quality.sh
bash scripts/adjoint.sh
bash scripts/finalize_baseline.sh
