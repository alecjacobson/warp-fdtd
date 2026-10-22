# Warp solver measurements

Spatial and temporal adaptivity are authorized work in progress. The flagship
remains uniform-grid Yee. See [measured adaptivity progress](adaptivity-progress.md)
and the [literature review and acceptance tests](adaptivity-research.md).

The runtime imports Warp and NumPy, not JAX or FDTDX. NumPy reads immutable
reference bundles and prepares temporal source and detector coefficients on the
host. Field updates, CPML recurrence, source injection and detector accumulation
run in Warp kernels on CUDA. NumPy performs final scalar modal overlaps and plots.
The regression suite checks that JAX and FDTDX were not imported.

## Supported flagship

`warp_fdtd.solver.Solver` consumes the exported `inputs.npz` and `manifest.json`.
The current tested configuration is float32, positive isotropic permittivity,
relative permeability one, uniform cubic grids, six axis-aligned CPML slabs,
and a real +x mode source spanning its plane. Unsupported input variants raise
errors. This is not yet feature parity with FDTDX (dispersive/anisotropic media,
nonuniform grids, other source orientations, symmetry, and multi-GPU are absent).

The explicit Yee update uses one kernel for E and one for H. Both include their
source correction. The H correction samples the incident E profile at n+1/2.
A missed half-step caused the initial field comparison to fail; its fix has a
one-step regression test. CUDA graphs capture 64 steps, with a distinct tail graph
so no time steps are skipped or repeated. Reset/replay is tested for identical
full E/H fields. The default PML history is full-volume allocated but accessed only inside its
slabs. Optional `--cpml-storage packed` stores only active planes, reducing owned
array storage but currently slowing the update; see the measured control in
[adaptivity progress](adaptivity-progress.md). The default remains the faster path.

Modal and flow detectors spatially colocate at the E_z Yee location. H is centered
in time using (H_previous + H_current)/2. Phasors use the exported final-eight-
period Hann coefficients. Mode overlaps use relative face-area weights, whereas
flux detectors use physical area. This distinction matters for absolute modal
power, even when ratios cancel an erroneous common area factor.

## Fairness and measured scope

Same physical input bundle, float32 fields, source, CPML coefficients, time step,
number of steps, and all exported detectors. Input and reference-field SHA256
checks precede comparisons. The 192k-cell case compares every E/H component in
3D; the larger archived cases contain central xy field slices, plus detector
traces and modal powers. Every current run checks finite full-volume fields.
No unknown full-volume errors are inferred from slice comparisons.

Warm timing excludes allocation, graph capture/JIT, initial execution, reset,
field transfer, final host modal overlaps and file output. Raw synchronized
repetitions, warmup, setup, compile/capture time, source hash and device name are
saved in each result. FDTDX timing includes its scalar objective; the Warp host
scalar overlap is outside trajectory timing, an asymmetry expected to be tiny
but explicitly acknowledged. Runs execute serially on the same 46-SM NVIDIA
RTX PRO 6000 Blackwell Server Edition MIG 1g.24gb partition. CPU affinity is 0–3.
GPU isolation outside this container cannot be established. These are measured
workloads, not a peak utilization or all-hardware claim.

A forward benchmark refuses a reference bundle labeled as a gradient workload.
The exact-adjoint and fast-adjoint FDTDX baselines must be compared with gradient
work, never with a forward solve.

## Accuracy

Matched Yee calculations validate implementation agreement, not improved formal
accuracy. A second periodic vacuum propagator combines fourth-order staggered
spatial differences with classical RK4 time integration. It evolves all six
components on a 3D grid. The analytic test has fields varying along one axis;
it does not establish accuracy for discontinuous dielectric interfaces, CPML,
all propagation directions or inverse design. The fourth-order path is not used
for the flagship bend yet.

The frequency fit searches the measured trajectory independently of the
numerical dispersion prediction. The continuum ground truth is omega=c*k.
Time-only error uses the exact-in-time spatially discretized frequency. The
fully discrete formula is an additional consistency check, not the fitted answer.
At fine grids, float32 roundoff and fit accuracy limit apparent convergence.
The 10 nm bend is a trustworthy refinement reference, not exact ground truth.

## Reproduce

Install `requirements-warp.txt` into a Python 3.12 environment. No upstream
checkout is needed for Warp execution. Existing FDTDX input/reference bundles
are committed. From the repository root, with the GPU otherwise idle:

```bash
OPENBLAS_NUM_THREADS=1 taskset -c 0-3 .venv/bin/python -m unittest discover -s tests -v
OPENBLAS_NUM_THREADS=1 taskset -c 0-3 .venv/bin/python -m warp_fdtd.benchmark
OPENBLAS_NUM_THREADS=1 taskset -c 0-3 .venv/bin/python -m warp_fdtd.benchmark --case results/warp_reference/hann_200fs --output results/warp/hann20
OPENBLAS_NUM_THREADS=1 taskset -c 0-3 .venv/bin/python -m warp_fdtd.benchmark --case results/warp_reference/forward10_gpu --output results/warp/forward10_gpu
OPENBLAS_NUM_THREADS=1 taskset -c 0-3 .venv/bin/python -m warp_fdtd.benchmark --case results/warp_reference/hann_10nm --output results/warp/hann10
OPENBLAS_NUM_THREADS=1 taskset -c 0-3 .venv/bin/python -m warp_fdtd.benchmark --case results/warp_reference/bend_flow20 --output results/warp/bend_flow20
OPENBLAS_NUM_THREADS=1 taskset -c 0-3 .venv/bin/python -m warp_fdtd.convergence
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m warp_fdtd.report
```

Replace `.venv` with your environment path. The two 24M-cell benchmark suites
run multiple full trajectories and take several minutes each on this partition.
The report regenerates the concise main README from measured JSON records.
Full FDTDX methodology and its known limitations remain in
[the archived baseline documentation](fdtdx-baseline.md).

The fourth-order frequency fit also fits exponential amplitude decay rather than
mistaking damping for frequency error. At CFL 0.99, measured amplitude loss over
20 periods is 1.52% at 8 cells/wavelength, 0.0493% at 16, and 0.00153% at 32.
The observed coarse-grid frequency convergence order is 3.97; the Yee order is
2.01. These are frequency-error results, not a claim of zero dissipation. The
error-versus-runtime panel uses a separate warm FDTDX benchmark with the same
point probe; the original first-execution timings are deliberately excluded.

## Exact material adjoint

`warp_fdtd.adjoint.Adjoint` supplies an explicit transpose of the discrete Yee
and CPML recurrences, including the transpose of detector interpolation. It
records three forward curl components only in the 80×80×20 design volume.
Reverse mode through the Gaussian/subpixel material map uses Warp's tape; the
large time trajectory is not recorded on a tape. The current driver deliberately
supports the fixed 20 nm, 4×4×1.5 µm smooth bend geometry.

The 3M-cell gradient takes **12.742 s**, versus **55.385 s** for FDTDX exact and
**25.227 s** for FDTDX's accepted frequency adjoint: **4.35× / 1.98×** speedup.
The full 6,400-parameter gradient differs from FDTDX exact by **1.85e-6 relative
L2**, cosine **0.9999999999992**. Three directional finite-difference step sizes
agree within 0.074%; best error is 0.032%. A changed-design replay also passes
finite differences (0.091%) and increases coarse-grid transmission from 92.8977%
to 93.0711%. This last check is a design-update smoke test, not a comparison of
optimization quality or budgets.

The history buffer alone is **8.056 GB** (decimal), plus fields, CPML, adjoint
buffers and graphs. This is a deliberate speed/memory tradeoff: FDTDX's fast
adjoint uses substantially less memory. No large-scale gradient memory advantage
is claimed. Timing includes forward, refreshed objective cotangents, reverse,
material pullback and gradient transfer. Initial material construction and graph
compilation are excluded. Raw synchronized repetitions and finite differences
are in [the gradient record](../results/warp/gradient20/result.json).

A material-map precision discrepancy required a separate reference. The original
FDTDX default convolution uses reduced-precision products; direct float32 Warp
Gaussian filtering differed by up to 0.00106 in inverse permittivity at a few
interfaces. The new FDTDX reference uses `JAX_DEFAULT_MATMUL_PRECISION=highest`,
with this flag recorded in provenance, and agrees in inverse permittivity to
1.61e-6. The original forward results remain untouched. Both exact and fast
FDTDX gradient baselines were remeasured at the matching precision. The fast
reference itself agrees with exact to 0.0513% relative L2.

Captured graph source buffers are updated in place after every objective
measurement. Reallocating them would leave old pointers inside the captured
graph. The changed-design regression verifies graph reuse and finite differences.

```python
import numpy as np
from warp_fdtd.adjoint import Adjoint
from warp_fdtd.detectors import attach

solver = Adjoint("results/warp_reference/gradient20_highest_exact")
attach(solver)
params = solver.data["parameters/p0"].squeeze(-1)
loss, gradient = solver.differentiate(params)  # loss = -modal transmission
updated = np.clip(params - 0.2 * gradient / np.linalg.norm(gradient), 0, 1)
next_loss, next_gradient = solver.differentiate(updated)
```

Reproduce the gradient validation with an otherwise idle GPU and enough free
memory (the 24 GB partition used here suffices):

```bash
OPENBLAS_NUM_THREADS=1 taskset -c 0-3 .venv/bin/python -m warp_fdtd.gradient_benchmark --reps 3
```

The six fast GPU regression tests include the full CPML-state transpose identity,
the detector space/time transpose identity, material-map finite differences, a
periodic mode varying along all three axes and its divergence constraint, source
half-step timing, and full 3D field/flux agreement and replay. The full adjoint
and changed-design finite differences above are separate integration checks.

To regenerate the matching FDTDX references, use the pinned baseline environment:

```bash
for method in exact adjoint; do
  JAX_DEFAULT_MATMUL_PRECISION=highest scripts/run.sh gradient --gradient "$method" \
    --quality-monitors --exact-interpolation --modal-window hann --objective modal \
    --beta 50 --load results/ring20_optimized/best_params.npz \
    --output "results/gradient20_highest_$method" --reps 2
done
JAX_DEFAULT_MATMUL_PRECISION=highest bash scripts/export_references.sh gradient20_highest_exact
```


## Direct periodic accuracy per runtime

The latest cost panel includes `baseline.periodic_benchmark`: identical 4×4×N
initial states, all six components, 20 nominal optical periods, float32, periodic
boundaries, and the same E_x point probe each step. Both solvers round the target
duration to the nearest step; this fixed a one-step ceil/round difference at
64 cells/wavelength. The report checks matching shapes, counts and time steps
(up to float32 grid-coordinate rounding). Compilation and the first execution
are separate from three synchronized warm repetitions.

For an interior example in the complete plotted sweep, Warp fourth order at
32 cells/wavelength takes **13.896 ms**, versus **18.570 ms** for FDTDX Yee at 64:
**1.34× faster with 32.8× lower frequency error**. RK4 loses 0.00153% amplitude
over 20 periods in this case. This is a smooth, effectively one-dimensional
mode on a 3D grid; it does not transfer the higher-order accuracy claim to the
3D bend's discontinuous material interfaces.

Regenerate the FDTDX timing fixture with the pinned baseline environment:

```bash
JAX_PLATFORMS=cuda,cpu OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 \
  XLA_PYTHON_CLIENT_PREALLOCATE=false taskset -c 0-3 \
  .venv/bin/python -m baseline.periodic_benchmark
```

Then run `warp_fdtd.convergence` and `warp_fdtd.report` as above. Raw FDTDX
repetitions are in `results/periodic_performance/result.json` and the plotted
example is summarized in `results/warp/periodic_cost_comparison.json`.
