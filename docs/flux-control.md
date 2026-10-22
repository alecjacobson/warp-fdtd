# Long-time flux arithmetic control

[Issue #27](https://github.com/alecjacobson/warp-fdtd/issues/27) exposed a long-time
float32 constraint drift in the experimental periodic rectilinear backend.
The ordinary and Kahan-compensated field updates fail the unchanged 1e-4 Gauss
gate at one million steps; float64 passes. These failing records remain public.

`FluxRectilinearPeriodic` uses the same explicit E-first Yee/leapfrog operator,
with integrated electric displacement and magnetic face fluxes as state. The
metric and material factors convert fluxes into edge impulses. Each edge
impulse is computed once in a separate kernel and shared by adjacent faces;
four signed impulses update a face. The oriented boundary-of-boundary identity
then makes divergence of the curl cancel before arithmetic rounding.

Each face has a high/low float32 accumulator. TwoSum tracks the rounding error
of each signed edge addition, followed by renormalization of the two-word
expansion. Fast math is disabled. The low word protects accumulation; the high
word drives the next explicit edge update. Metric coefficients and impulses
are float32, so this is not an all-float64 solve. The error-free addition primitive
comes from [Ogita, Rump and Oishi, *Accurate Sum and Dot Product*](https://epubs.siam.org/doi/10.1137/030601818);
that arithmetic result does not prove stability or accuracy of our Maxwell code.

This initial implementation stores 21 float32 values per cell plus axis-width
arrays: six high fluxes, six low words, six coefficients and three shared edge
impulses. It launches four kernels per step. Relative to the ordinary periodic
control, that is about 2.33× the float32 owned-array storage and 1.17× the
float64 owned-array storage. These are allocation counts, not peak device
memory. GPU timing determines whether this arithmetic tradeoff is useful.
The flagship solver and default periodic backend are not changed by this control.

## Validation

```bash
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m warp_fdtd.adaptive_validation \
  --flux --output results/warp/flux_validation_gpu
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m warp_fdtd.flux_validation \
  --output results/warp/flux_sweep_gpu
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m warp_fdtd.flux_validation \
  --cost-only --output results/warp/flux_cost
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m warp_fdtd.fresnel_validation \
  --output results/warp/fresnel_flux_gpu
```

The original analytic wave, independent tiny-operator and one-million-step
broadband tests retain their gates. A separate 12-case million-step sweep uses
CFL 0.3/0.9/0.99, uniform and unequal smoothly graded axes, and seeded sharp
diagonal permittivity contrasts up to 100. Material Gauss checks use D=epsilon E,
not E. Ten checkpoints per case measure modified-energy drift and normalized
D/B divergence; all use the existing 1e-4 threshold. The cost command measures
five synchronized warm trajectories on 64³ and 128³ grids for ordinary float32,
ordinary float64 and the flux control, with initialization and compilation
excluded. No source, CPML or block-interface costs are present in these controls.

CPU runs are available with `--device cpu` on the flux validation commands;
these are correctness controls, not GPU throughput evidence. CPU graph warmup
must copy saved arrays because Warp's CPU NumPy view aliases its storage. Tests
check exact high/low state preservation and reset/replay in addition to
independent NumPy curl comparisons on unequal axes with material contrast.

The Fresnel control has planar symmetry but evolves the actual 3D operator.
A Gaussian carrier pulse is initialized in epsilon=1 and propagates toward a
planar interface with epsilon=2.25 or 12.25. Tangential electric components
use an arithmetic average over their interface-straddling dual volume. Reflected
and transmitted packets are separated before measuring their fractions of the
proper leapfrog modified energy. The known nondispersive half-space result is
R=((1-sqrt(epsilon))/(1+sqrt(epsilon)))², T=1-R. The test reports error versus
resolution, energy drift and residual boundary energy. Its passing gate tests
second-order convergence, not a fixed accuracy at every coarse resolution.

This remains a periodic rectilinear correctness control. It does not establish
oblique scattering, frequency-resolved slab spectra, 3D sphere/Mie accuracy, coarse–fine face/edge/
corner coupling, local time stepping, source/CPML compatibility or adjoints.
Those dependencies remain in [#13–#16](https://github.com/alecjacobson/warp-fdtd/issues/13).

## Finite-slab spectral integral

`warp_fdtd.slab_validation` adds a finite slab with both interfaces. It compares
separated outgoing packet energies with the analytic Fabry–Perot transmission
integrated over the known Gaussian carrier spectrum. For n=sqrt(epsilon),
interface power reflection rho=((1-n)/(1+n))² and slab thickness d,

T(k)=(1-rho)² / (1-2 rho cos(2 n k d)+rho²).

The Poisson-kernel expansion gives an analytic Gaussian-weighted cosine series.
The cosine moment at delay tau is
exp(-tau²/(4 sigma²)) [cos(k0 tau)+exp(-sigma² k0²)] / [1+exp(-sigma² k0²)].
The series is summed through 64 terms; the reported geometric tail bound is
2 rho^65/(1+rho), excluding floating-point evaluation error. An independent
SciPy spectral quadrature test checks both narrow and broad real-pulse spectra.

The domain is 3×0.08×0.08, the slab occupies x=[1.2,1.22], and the pulse starts
at x=0.5 with sigma=0.06 and carrier wavelength 0.05. At t≈1.6 the reflected
and transmitted packets remain away from the periodic boundaries. Residual
slab energy is reported and must be below 1e-5; boundary energy below 1e-6.
The same second-order and 1e-4 energy-drift gates apply. This tests the integrated
Fabry–Perot response; a resolved frequency sweep and oblique incidence remain
separate verification tasks.

```bash
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m warp_fdtd.slab_validation \
  --output results/warp/slab_flux_gpu
```

## Measured GPU results

The original million-step control and all twelve material/CFL million-step cases
pass on the RTX PRO 6000 Blackwell MIG 1g.24gb. The sweep maxima are energy
drift **5.54e-6** and normalized Gauss residual **1.03e-6**. CPU and GPU checkpoint
metrics agree exactly in these runs. All **22 regression tests** pass.

The `--flux --flux-plain` ablation retains shared-edge metric-consistent fluxes
but removes the two-float accumulator. It preserves second-order convergence
and energy (maximum drift 1.73e-6), but its million-step Gauss residual reaches
**7.30e-4**, failing the unchanged 1e-4 gate. Shared fluxes alone are insufficient.
That failure is retained in `results/warp/flux_plain_gpu/result.json`.

| Grid | Ordinary float32 | Ordinary float64 | Two-float flux | Owned arrays: float32 / float64 / flux |
|---|---:|---:|---:|---:|
| 64³ | 0.024013 s | 0.206630 s | 0.071081 s | 9.44 / 18.88 / 22.02 MB |
| 128³ | 0.483669 s | 1.588842 s | 1.154028 s | 75.50 / 151.00 / 176.16 MB |

Five synchronized warm repetitions of 1024 steps; raw samples and code hashes
are in `results/warp/flux_cost_final/result.json`. At 128³ the flux control is
1.38× faster than the passing float64 control, but 2.39× slower than ordinary
float32 and uses more memory. The smaller grid has a different memory/cache
regime. This is a correctness/performance tradeoff, not an adaptive flagship win.
[#32](https://github.com/alecjacobson/warp-fdtd/issues/32) tracks cost reduction
and production integration. The ordinary float32 control remains a failed
long-time constraint reference, even though its short-run timing is reported.

GPU Fresnel convergence slopes are **2.002 / 2.020** for epsilon=2.25 / 12.25.
The finite-slab slopes are **2.008 / 1.982**; maximum residual slab energy is
reported per run and stays below the 1e-5 gate. These are normal-incidence
convergence checks. Oblique, sphere/Mie and coarse–fine interface tests remain
open under #13–#15.

The initial GPU results and the final rerun after adding the ablation are both
retained. All passing long-time checkpoint metrics are identical between those
versions; `*_gpu_final` and `flux_cost_final` identify the final implementation.
