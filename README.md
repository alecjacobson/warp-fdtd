# Warp FDTD

Pure NVIDIA Warp 3D electromagnetics, with uniform FDTDX comparisons and a new
binary-octree waveguide prototype.

**3D octree result: 22.2× faster and 12.0× less solver GPU memory** than the best
tested uniform grid at comparable sampled accuracy. The 4.8 × 12.8 × 12.8 µm
straight-guide problem uses 25 nm cells near the guide and up to 200 nm in cladding.

| Solver | Active cells | Full warm simulation | Solver GPU memory | Field difference¹ |
|---|---:|---:|---:|---:|
| Uniform Warp, 25 nm | 50.33M | 11.774 s | 1,157 MiB | 5.07% |
| Octree Warp, 25–200 nm | 399,360 | 0.530 s | 96.6 MiB | 5.17% |

¹ Sampled transient E/H difference from a 12.5 nm, 402.7M-cell reference; not a
continuum error bound. Source and sponge updates are timed. **Prototype limits:**
sponge/interface flux consistency fails, and 33.3 s host assembly makes a fresh
one-shot solve slower. These are warm simulation and GPU-memory gains.

![Latest octree accuracy, runtime and memory trends](results/octree_guide_large/trends.png)

![3D guide fields and instantaneous power-flow comparison](results/octree_guide_large/fields.png)

[Octree mesh, raw measurements and reproduction](docs/octree-guide.md) ·
[Flux fix #42](https://github.com/alecjacobson/warp-fdtd/issues/42) ·
[Setup cost #43](https://github.com/alecjacobson/warp-fdtd/issues/43)

**Optimized 3D bend versus pinned FDTDX:** matched forward runs remain
**2.96–3.24× faster**, with uniform Yee grids, CPML and modal sources/detectors.
The fixed-geometry reference reaches 93.2662% transmission at 5 nm / 192M cells;
the 6.67 nm electric-field difference is 0.685% and power-flow difference 0.809%.

![Bend performance and numerical agreement](results/warp/performance.png)

![Latest controlled bend mesh trends](results/physical_reference/comparison_fine/convergence.png)

![Bend time-averaged power flow](results/physical_reference/comparison_fine/power_flow.png)

Measurements use an RTX PRO 6000 Blackwell **MIG 1g.24gb**, float32 and serial
synchronized GPU runs. Warm timings exclude setup, compilation and diagnostics.

[Bend reproduction](docs/warp.md) · [Reference accuracy](docs/physical-reference.md) ·
[Prioritized work](https://github.com/alecjacobson/warp-fdtd/issues)
