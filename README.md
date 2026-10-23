# Warp FDTD

Pure NVIDIA Warp 3D electromagnetics, compared with pinned, unmodified FDTDX
on an optimized silicon 90° waveguide bend. The flagship uses explicit Yee
updates, a uniform grid, CPML, injected modes and Hann-window modal detectors.

**Measured status:** **2.96–3.24× faster** matched forward runs. Adaptive
flagship gains and continuum reference uncertainty remain unresolved.

| Same-input flagship workload | FDTDX | Warp | Speedup |
|---|---:|---:|---:|
| 20 nm · 3M cells · 200 fs | 14.683 s | 4.534 s | 3.24× |
| 10 nm · 24M cells · 300 fs | 316.138 s | 106.709 s | 2.96× |

![Flagship performance and same-input numerical agreement](results/warp/performance.png)

The new fixed-geometry convergence study reaches **93.2662% transmission at
5 nm / 192M cells**. Compared with that mesh, the 6.67 nm complex electric-field
L2 difference is **0.685%**, and Poynting-vector L2 difference **0.809%**. These are
mesh differences, not continuum error bounds; [#30](https://github.com/alecjacobson/warp-fdtd/issues/30)
tracks the remaining reference-accuracy budget.

![Latest controlled flagship mesh trends](results/physical_reference/comparison_fine/convergence.png)

![Common-scale time-averaged power flow across three 3D meshes](results/physical_reference/comparison_fine/power_flow.png)

Power-flow images show a common z=790 nm slice, normalized by incident power.
Timing fixtures retain their archived source/port positions; the separate
convergence study fixes physical geometry and observation windows. Reference
generation and volume-observer timings are excluded from speedup claims.
The archived 3M-cell material gradient is **1.98× faster**, with **1.85e-6**
relative agreement with FDTDX's fast adjoint.

Measurements use an RTX PRO 6000 Blackwell **MIG 1g.24gb**, float32 and
synchronized warm runs; throughput timings exclude compilation, setup and host
postprocessing.

[Reproduction](docs/warp.md) · [Controlled references](docs/physical-reference.md) ·
[Adaptivity progress](docs/adaptivity-progress.md) ·
[Prioritized issues](https://github.com/alecjacobson/warp-fdtd/issues)
