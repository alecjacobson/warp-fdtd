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
oblique scattering, slab resonances, 3D sphere/Mie accuracy, coarse–fine face/edge/
corner coupling, local time stepping, source/CPML compatibility or adjoints.
Those dependencies remain in [#13–#16](https://github.com/alecjacobson/warp-fdtd/issues/13).
