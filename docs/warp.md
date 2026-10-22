# Warp solver measurements

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
full E/H fields. PML memory is currently full-volume allocated but accessed only
inside its slabs; compact storage is a future capacity optimization.

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
error-versus-runtime panel compares the two Warp methods; archived FDTDX mode
execution timing included first execution and is deliberately not used there.
