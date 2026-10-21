# FDTDX baseline for a future Warp implementation

This repository measures the unmodified FDTDX solver and builds a silicon waveguide
bend. It separates solver performance, numerical correctness, optimization quality,
and reproduction of the published experiment. Results are local measurements, not
claims about all GPUs or all FDTDX workloads.

Read the [measured report](results/REPORT.md), [quality gates](results/quality_checks.json),
and [numerical comparisons](results/comparisons.json). Raw per-run JSON retains failures
and every timed repetition. Full reproduction is `bash scripts/reproduce.sh`; it
replaces the named result directories and takes roughly an hour or more on this partition.

The baseline is pinned to FDTDX commit
[`2cab8dbb2af9a43ec6443712f011f50ecf58790b`](https://github.com/ymahlau/fdtdx/tree/2cab8dbb2af9a43ec6443712f011f50ecf58790b),
version 0.6.3, with JAX 0.12.0 and Python 3.12.15. All Python distributions are
pinned in `requirements.lock`. `upstream/` is an unmodified checkout, deliberately
separate from this measurement harness. The scripts reject a dirty or wrong checkout.

## Hardware actually available

- NVIDIA RTX PRO 6000 Blackwell Server Edition **MIG 1g.24gb**, 46 SMs;
  24,192 MiB exposed by NVIDIA, about 19.0 GB allocator limit under default JAX policy.
- AMD EPYC 9B45 host, **four CPU equivalents by cgroup quota**, 16 GiB host RAM limit.
- CPU affinity is fixed to logical CPUs 0–3; BLAS uses one thread. GPU and CPU
  benchmark processes run sequentially. Host CPU/GPU sharing outside this container
  cannot be excluded; this is a MIG partition, not an exclusive full GPU.

The paper used H100 hardware, including four GPUs at its largest scale. Direct
runtime ratios against those published numbers are not hardware speedups.
Multi-GPU and billion-cell capacity require additional hardware and are unvalidated here.

## Reproduce

```bash
scripts/bootstrap.sh
scripts/validate.sh
.venv/bin/python -m baseline.compare
scripts/run.sh forward --output results/my_forward --reps 5
scripts/run.sh gradient --gradient exact --output results/my_gradient --reps 3
```

The default simulation is 4 × 4 × 1.5 µm, 20 nm spacing, 200 fs, float32,
Courant factor 0.99, six 10-cell PMLs, 1550 nm fundamental mode excitation,
400 nm square silicon guides, a 600 nm silica substrate, and a 1.6 × 1.6 µm
design region with 6,400 parameters. The grid has 3 million cells and 5,245 steps.
Material relative permittivities are air=1, silica=2.25, silicon=12.25.

Every run writes settings, exact package/hardware provenance, setup and compilation
times, first-execution warmup, every synchronized repetition, median runtime,
allocator peak, compiler memory analysis, finite-field checks, and measurements to
`results/<name>/result.json`. Outputs are synchronized using `jax.block_until_ready`.
First execution is excluded from steady-state timing. GCUPS means complete electric
and magnetic cell updates per second; for a gradient run it is only **forward-equivalent**
work per gradient evaluation, not a count of all backward/recomputation updates.

Timings include device parameter application, field/boundary/detector reset, source
injection, PMLs, detector collection and objective calculation. They exclude mesh/source
setup, compilation, validation reductions, host copies, plotting and file writes.
Optimization records gradient time and synchronized update time separately.
Allocator peaks include setup/compilation/warmup; they are not total board memory.

## Correctness and quality

`baseline.physics` calls unmodified upstream analytic test functions directly on
CUDA because upstream pytest's root conftest forces CPU. It tests wave speed and
impedance in vacuum/dielectric and Fresnel transmission. The upstream Fresnel
"power conservation" test derives R from 1−T; it is not an independent measurement
of reflected power. A separate CPU pytest run covers reversible recorder and
segmented-gradient equivalence, with results in `results/gradient_tests.xml`.

`baseline.compare` checks GPU/CPU detector traces and full three-dimensional E/H fields against
a relative L2 tolerance of 1e-3. It compares full parameter gradients with
checkpointed autodiff, including cosine similarity and a central finite-difference
step-size sweep along the gradient direction. Float32 cancellation at small steps
must not be confused with a solver gradient error. All step sizes and failures are
retained. Binary/brush transformations use straight-through estimators; ordinary
finite differences are not a validity test for those surrogate derivatives.

Three reversible recording choices are exposed:

- `exact`: uncompressed float32, segmented recording, automatic checkpoint count.
- `paper`: sample every two steps and store float16 (paper-era example setting).
- `compressed`: sample every five steps and store float8 (current example setting).

The paper-era core-flux ratio uses input periods 3–5 and the final two output periods.
Since the current source ramps up over four periods, this ratio is **not sufficient
evidence of physical transmission efficiency**. `--quality-monitors` adds separate
forward/backward input mode overlaps and output overlap over the last eight periods,
with exact field interpolation and the full transverse aperture. Report both metrics.
Use `--modal-window hann` for the quality protocol. Eight rounded periods with a
rectangular window gave a 1.785 percentage-point duration dependence on the saved
design. FDTDX's built-in Hann apodization removes this phasor-window sensitivity;
both sets of measurements are retained. The optimization used the original rectangular
window; the independent evaluator changes only the measurement window.
Use longer-time, finer-grid and PML/padding checks before certifying a design.
`baseline.quality_checks` requires straight-guide T within 1% of unity and R below
0.1%, duration/PML transmission changes below 0.5 percentage point, mesh changes
below 1 percentage point, and fine-mesh transmission of at least 92%. It also checks
that the physical material cross section and saved candidate match across evaluations.
These gates were set before evaluating the optimized candidate. `--pad-domain`
extends all outer faces when increasing PML thickness, preserving the active geometry.
The historical source used an abrupt cosine without this ramp: `--source-profile paper`
restores that waveform. The ramp concern applies to mixing the old monitor windows
with the modern source, not to a demonstrated flaw in the original paper.

## Inverse design and the published result

The target is the silicon bend in
[Mahlau et al., arXiv:2412.12360v2](https://arxiv.org/html/2412.12360v2), which reports
92% efficiency. Geometry and legacy monitor placement are adapted from the
[December 2024 example](https://github.com/ymahlau/fdtdx/blob/a0a1c98b11fb765dbd8c6d339dde79fca1bfe148/examples/optimize_ceviche_corner.py).
The current upstream example has changed geometry, duration and parameterization;
running it alone does not reproduce that paper.

The harness supports the binary pretraining and 100 nm circular-brush phase:

```bash
scripts/run.sh optimize --gradient paper --source-profile paper --parameterization binary --epochs 250 \
  --output results/paper_pretrain_seed0
scripts/run.sh optimize --gradient paper --source-profile paper --parameterization brush --epochs 20 \
  --load results/paper_pretrain_seed0/final_params.npz --output results/paper_brush_seed0
scripts/run.sh forward --source-profile paper --parameterization brush --quality-monitors --exact-interpolation \
  --load results/paper_brush_seed0/best_params.npz --output results/paper_eval_seed0
```

This is a port to the current solver, not an exact historical software reproduction.
`PaperBrush` restores the old brush's complemented forward values and unchanged
straight-through derivative, so latent orientation stays consistent with pretraining.
`PaperClosestIndex` flattens the current isotropic material lookup table to avoid
an upstream broadcasting error in inverse-permittivity quantization. These adapters
are checked by `baseline.protocol_checks`; they do not modify the measured solver.
Do not assert the 92% result from the existence of these commands. Full reproduction
also needs independent evaluation and multiple seeds (the publication compared five).
At approximately a minute per exact gradient, long runs take hours on this partition.

A separate practical baseline can optimize from a specified quarter-annulus using
`--init ring --objective modal --fixed-beta --beta 50`. This is explicitly a different
initialization and objective from the publication. It can establish a useful device
and solver baseline without representing a random-start paper reproduction.
`scripts/quality.sh` runs this optimization and its independent evaluations.
`bash scripts/evaluate_hann.sh` repeats only the independent evaluations of the saved
candidate. `bash scripts/finalize_baseline.sh` checks gradients with this window,
runs the capacity stress case, and exports the fine-grid inputs.
`scripts/adjoint.sh` compares the newer frequency-adjoint method with exact segmented
gradients on the modal objective, including a full-resolution optimized candidate.
An adjoint speedup is accepted only alongside gradient agreement; short CW runs or
unsettled resonances can invalidate the frequency approximation.

Optimization checkpoints every five steps, retains the best **evaluated** parameter
set and its metric, and saves the final unevaluated update separately. Resume with
the same command plus `--resume <run>/checkpoint.pkl`. Only load local trusted pickle
checkpoints. Parameter `.npz` files are the interchange format for later Warp work.

## Fair comparison contract for Warp

Match geometry/rasterized materials, all six field components, dtype, grid, Courant
factor, step count, boundary thickness and profiles, source waveform/mode, detector
positions, interpolation, reset behavior and output quantity. For inverse design,
also match parameter transforms, initial parameters, objective, derivative accuracy,
optimizer, seed and optimization budget. Save identical input arrays for both engines.

Compare full synchronized forward runs and end-to-end optimization separately from
kernel-only timings. Report cold compilation, warm execution, dispersion across
repetitions, memory, and attained physical quality. Do not compare a Warp vacuum
stencil with FDTDX's full PML/source/detector/gradient workload, exclude FDTDX setup
while including Warp setup, or silently lower precision/monitor accuracy. GPU speed
claims here apply to this specific MIG partition and four-core CPU allocation.

Predeclare the cross-engine gates before implementing Warp: full small-grid E/H
relative L2 error at most 1e-3, same-grid modal transmission difference at most
0.001 absolute (0.1 percentage point), and parameter-gradient relative L2 error
below 1% with cosine above 0.999 against exact segmented differentiation. Retain
the independent physical-quality gates above. Compare the faster accepted FDTDX
frequency adjoint as well as the exact reference when measuring optimization.

`baseline.export_case` writes `inputs.npz` and a checksummed manifest under
`results/warp_reference/<case>/` for the small full-field comparison, 24-million-cell
performance and quality cases, and exact/frequency-adjoint gradient cases. This includes the full inverse-permittivity
array, grid edges, source/reference-mode arrays, staggered source timing offsets,
CPML coefficients, detector sample steps, and Hann weights. The manifest points to
the pinned update conventions. Recreate it with the command in
`scripts/export_references.sh`; the exporter verifies material agreement with the
evaluated cross section. Match this quality case separately from the default faster
performance cases, which use different detector interpolation and measurement work.
Each manifest also identifies and checksums the measured fields, detector traces,
and parameter gradients where available. The small case retains full 3D fields;
large cases retain cross sections and detector measurements.

## Attribution

`baseline/scene.py` adapts the MIT-licensed FDTDX example. The upstream license is
preserved in `NOTICE`. Solver sources are not modified by this repository.
