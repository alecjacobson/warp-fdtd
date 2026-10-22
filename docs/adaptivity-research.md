# Spatial and temporal adaptivity: research proposal

Research date: 2026-10-22. **Implementation authorized by the user.** Work follows
the dependency order below. The initial verification and measurement work is
recorded in [adaptivity progress](adaptivity-progress.md); no adaptive flagship
speedup has yet been demonstrated. Main README results are flagship-only.

## Recommendation

Start with a common verification suite and fixed, locally refined Cartesian
blocks. Establish spatial-interface correctness with one global time step before
adding local time stepping (LTS). Use the published 3:1 dissipative-interface
construction as the first 3D reference algorithm. Evaluate 2:1 SBP-SAT separately
if binary refinement or higher-order interiors justify its additional work.

For the continuously driven silicon bend, refine the waveguides, bend, material
interfaces and their evanescent-field buffers; keep sources, port planes and
CPML in deliberately resolved regions. Refining only the design box is unlikely
to control propagation error along the leads. This is our engineering inference,
not a result established by the papers.

Use GPU batches of dense blocks with precomputed interface lists. An octree can
organize these blocks later; it does not itself define a stable Maxwell update.
Avoid committing to pointer-based per-cell traversal before measuring its cost.
A 3:1 prototype is not an octree. A 2:1 balanced block hierarchy is a different
implementation choice requiring its own interface treatment.

## What adapts?

| Choice | Meaning | Main tradeoff for this repository |
| --- | --- | --- |
| Rectilinear / anisotropic grid | Different spacing along axes or across coordinate planes | Simple structured control; refinement stretches across entire planes |
| Static local h-refinement | Small cells only in selected blocks | Fewer unknowns, but a global explicit step still obeys the smallest-cell restriction |
| Local time stepping | Small steps in fine blocks, larger steps elsewhere | Requires consistent space-time interface exchange, not just repeated fine updates |
| Global variable time step | All cells change step together using an error estimate | Still CFL-limited; must resolve the optical carrier even after the envelope settles |
| p / hp adaptation | Change local polynomial order, optionally cell size | Attractive in smooth regions; interfaces and singular corners need separate treatment |
| Dynamic AMR | Refine/coarsen while the wave propagates | Transfer error, divergence, scheduling, allocation and replay become part of the algorithm |

For our linear, fixed-geometry, continuous-wave workload, there is no obvious
large temporal interval in which a global explicit step can grow beyond its
stability bound. Low field amplitude does not remove that bound. LTS is the
more compelling temporal experiment. A variable-step control is still useful
for pulsed workloads, with an integrator designed for changing steps; changing
`dt` inside the current staggered leapfrog loop is not a valid implementation.

## Primary literature and evidence limits

Full-text sections on formulation, stability and numerical experiments were
examined for [A], [B], [C], [D], [G], [H], [I], [J], [K] and [L]. For [E], [F]
and [M], access was limited to publisher/institutional abstracts and available
section previews. These are leads with explicitly limited evidence, not papers
whose full algorithms were independently reproduced. The additional [N]–[P]
leads below were examined through institutional records or indexed
author-manuscript excerpts; full retrieval was unavailable.
None of the published experiments was rerun during this research pass.

**[A] Bekmambetova, Zhang and Triverio, 2018,**
[A Dissipation Theory for Three-Dimensional FDTD with Application to Stability Analysis and Subgridding](https://arxiv.org/html/1705.02274v2).
Sections IV–VI and the appendix construct lossless inter-grid exchange, including
3D boundary junctions and material traversal. The demonstrated refinement ratios
are odd; all regions share the finest-grid time step. A cavity test runs one
million steps. This is the closest starting point to our Yee implementation.
Its reported large uniform-fine runtime is extrapolated from 30 steps, so its
headline speedup is not a GPU performance forecast for us.

**[B] Wang et al., 2022 preprint / 2023 journal,**
[SBP-SAT subgridding without modifying Yee field components](https://arxiv.org/html/2202.10770v2).
Section III gives 2:1 interfaces with paired interpolation operators satisfying
an energy-weighted transpose relation. SAT terms enforce boundary coupling.
The demonstrated system is **2D TM**, with a global fine-grid time step. This
supports a binary-refinement experiment, not an already validated 3D AMR solver.

**[C] Cheng et al., 2022 preprint,**
[Towards the Development of a Three-Dimensional SBP-SAT FDTD Method](https://arxiv.org/html/2206.00638v1).
The 3D framework is tested on cavities, a filter and a tissue model, including
CPML. The conclusion explicitly leaves the subgridding application to another
article. Combining this with [B] still requires a derived and tested 3D interface.

**[D] Wang et al., April 2026 preprint,**
[A Stable SBP-SAT FDTD Subgridding Method Without Region Split](https://arxiv.org/html/2604.14618v1).
Embedded regions avoid fragmenting the surrounding coarse domain. Tests remain
2D TM, use a global fine-grid step, and run in single-threaded MATLAB. The paper
reports first-order corner closures despite second-order interior interpolation.
Interesting topology follow-up; not evidence of a production 3D GPU octree or LTS.

**[E] Collino, Fouquet and Joly, 2006,**
[Conservative space-time mesh refinement methods for FDTD Maxwell](https://doi.org/10.1016/j.jcp.2005.03.035).
The publisher abstract describes discrete-energy conservation and an SPD solve
restricted to the coarse–fine interface at every step. This is a useful
conservative alternative if explicit coupling disappoints, but is not a wholly
explicit stencil algorithm. Full-text derivation must be obtained before a port.

**[F] Grote and Mitkova, 2010,**
[Explicit local time-stepping methods for Maxwell's equations](https://doi.org/10.1016/j.cam.2010.04.028),
with an [institutional record](https://edoc.unibas.ch/entities/publication/583bac5a-b507-48a5-a4fe-c902b863f9e9).
The described Maxwell schemes use diagonal/block-local mass structure; the
lossless source-free case conserves discrete energy. DG numerical experiments
are reported. These are method-specific results, not permission to subcycle any
spatial interface with linear interpolation.

**[G] Grote and Mitkova, 2012,**
[Explicit local time-stepping methods for time-dependent wave propagation](https://arxiv.org/html/1205.0654v2).
Sections 3–5 give leapfrog/modified-equation and damped Adams–Bashforth LTS
algorithms. The fine-region projector and the spatial operator do not commute;
interface-neighbor work matters. Tests show that overlap and integrator choice
affect the usable CFL. Damping support is relevant to CPML research, but is not
a proof for our particular auxiliary-variable CPML recurrence.

**[H] Grote, Michel and Sauter, 2021, revised preprint 2022,**
[Stabilized leapfrog-based local time stepping](https://arxiv.org/pdf/2005.13350).
The original LF-LTS can have isolated unstable step sizes even below its nominal
limit. Section 4 demonstrates deliberately excited critical modes; a Chebyshev
stabilization removes them in the analyzed wave-equation setting. We should
sweep step sizes and examine amplification matrices, not test only CFL 0.99.
Transferring the result to first-order Yee plus CPML requires further analysis.

**[I] Grote, Michel and Sauter, 2024 preprint,**
[Explicit LTS for the inhomogeneous wave equation](https://arxiv.org/html/2409.18085v1).
Source terms crossing refinement boundaries can degrade convergence; the paper
introduces a weighted transition to recover the analyzed accuracy. Its theorems
concern its FE wave formulation. For us this motivates source-location tests and
careful temporal injection, rather than reusing a theorem outside its scope.

**[J] Schnepp and Weiland, 2011 preprint,**
[Efficient large-scale electromagnetics using dynamically adapted DG meshes](https://arxiv.org/html/1109.0856v2).
Projection-based h/p changes are analyzed for stability; coarsening loses
information. The automatic error/smoothness-driven hp examples are 1D. The
large 3D radar-reflector example instead uses an energy-density indicator.
That distinction matters: field magnitude is not a certified estimator of
transmission error or of weak reflected waves.

**[K] Lilienthal, Schnepp and Weiland, 2014,**
[Non-dissipative space-time hp-DG for Maxwell](https://arxiv.org/html/1307.5310v2).
Local spatial and temporal adaptation is demonstrated, including comparison
with sphere-scattering solutions. The method is **implicit** and solves
space-time systems iteratively. Mesh coarsening can introduce dissipation.
Useful evidence for feasibility and reference-solution indicators, but a much
larger architectural change than block Yee refinement.

**[L] Farjadpour et al., 2006,**
[Improving accuracy by subpixel smoothing in FDTD](https://math.mit.edu/~stevenj/papers/FarjadpourRo06.pdf).
Proper interface-oriented tensor averaging improves dielectric-boundary
accuracy; arbitrary scalar smoothing is insufficient. Sharp corners complicate
convergence. Our current density filtering/projection is not automatically this
constitutive discretization. Geometry error should be controlled separately
before crediting mesh adaptation with an accuracy improvement.

**[M] Pederson and Raja, 2019,**
[A stable FDTD scheme for local time stepping on an adaptive mesh](https://doi.org/10.1016/j.jcp.2019.05.043).
The accessible preview reports 1D–3D tests, ratios 2 and 3, and higher-order time
stepping that damps aliasing, with a stated CFL limit of 5/6 in its convention.
This is a relevant alternate LTS lead. Obtain the complete algorithm and map its
CFL definition before using it; quantify damping and resonator Q, not only lack
of blow-up.

**[N] Van Londersele et al., 2018,**
[Systematic cell-by-cell FDTD subgridding in 3D](https://biblio.ugent.be/publication/8587734).
The institutional abstract describes finite-integration-based coupling with
variable refinement ratios and a ridged-waveguide S-parameter validation. This
is another established 3D family; SBP-SAT is not the only route to flexible
refinement. Full text was institution-restricted in the inspected record.

**[O] Martín Valverde et al., 2024,**
[Stability improvement of 3D subgridding using LECT](https://doi.org/10.1109/TAP.2023.3327776).
The indexed author manuscript describes orthogonalized integral-based
subgridding, spectral stability analysis, and local enlarged-cell changes to
increase the permitted step. It references an explicit LTS construction.
Obtain the full derivation and separate proven bounds from heuristic LTS tests
before treating it as an alternative to [A] plus a new temporal coupling.

**[P] Martín Valverde, Teixeira and Ruiz-Cabello, 2025,**
[Discretization errors in orthogonalized integral-based subgridding](https://doi.org/10.1109/TAP.2025.3617134).
The indexed manuscript reports that points per wavelength alone do not control
near-field error: object-to-refinement-boundary buffer distance also matters.
This directly motivates our buffer-distance sweep and warns against refining
only immediately adjacent to material. Full PDF retrieval failed during this
pass; the numerical tables and complete methodology remain to be checked.

## Design implications

Energy-compatible transfers constrain both directions of interpolation. For a
simple paired trace space, a representative requirement is
`W_coarse R = P^T W_fine`, where P prolongs coarse traces and R restricts fine
traces. Actual E/H staggered locations and boundary normals determine the
operators and signs. Independently chosen trilinear transfers generally do not
establish this property. An interface energy argument also does not by itself
prove preservation of `div B` or `div D`; test compatible flux transfers separately.

Distinguish a spatial (semi-discrete) energy estimate from stability of the
actual time integrator. Use its proper staggered/modified discrete energy.
Test closed lossless systems first; CPML, sources, material maps and observers
then need their own checks. Ports must accumulate with the actual physical
sampling times and quadrature weights. E/H co-location and half-step timing
remain necessary under subcycling.

Higher order and adaptivity are complementary. Use high order where fields and
coefficients are smooth; use refinement or an interface-aware discretization at
jumps. A fourth-order interior stencil does not imply fourth-order convergence
of a dielectric bend with lower-order boundaries, material averaging or PML.

The fastest plausible route is still custom Warp kernels. An exploratory DG
branch may evaluate `warp.fem` for basis/quadrature/geometry infrastructure, but
should measure generated kernels, mass inversion, face exchange and memory
before choosing it. Locally implicit or space-time methods remain fallback
research options: removing a stability restriction does not remove phase-error
requirements or the cost of linear solves.

For inverse design, initially select a hierarchy between solves and freeze it
during differentiation. The exact discrete transpose must include all transfers,
substeps, source/detector operations and auxiliary state. A derivative conditional
on a fixed mesh is not a derivative through threshold-driven remeshing.

## Success contract proposed for discussion

These are **proposed project gates**, not numerical guarantees from the papers.
Freeze tolerances, workloads and reporting rules before tuning the algorithms.
A failed experiment should publish its failure and can close with a no-go decision.

| Test | Proposed measurement / gate |
| --- | --- |
| Algebra | Tiny-grid assembled operators; interface power balance and transpose residual <=1e-11 in float64 construction, <=1e-5 in float32 execution; inspect amplification spectrum and defective neutral modes |
| Smooth waves | 3D oblique periodic modes and homogeneous pulses; mesh and time sweeps separately; measured order >=1.8 for the second-order candidate before roundoff; phase and amplitude both reported |
| Artificial-interface scattering | Both directions, TE/TM, normal/30/60-degree incidence, multiple wavelengths and interface locations; reflected power <=1e-5 (-50 dB) over the declared band with >=20 coarse cells per shortest material wavelength |
| Material physics | Analytic dielectric slab Fresnel/Fabry–Perot; 3D dielectric sphere against Mie; move material boundaries through refinement faces/edges/corners; include epsilon_r=12.25 |
| Long time | Small source-free closed 3D cases for 1e6 fine steps, seeded broadband and near-Nyquist modes; no secular growth and <=1e-4 relative drift of the appropriate invariant for a conservative scheme; quantify physical-mode attenuation for dissipative candidates |
| Constraints | Source-free composite-grid Gauss residuals, including refinement interfaces; require normalized residual <=1e-4 on compatible smooth tests, using L times divergence norm / field norm and a stated fixed physical L; inspect localized spikes separately |
| Flagship | Same continuous geometry, wavelength, physical duration, source normalization and detector windows; absolute T error <=1e-3 (0.1 percentage point), absolute R error <=1e-4; volume-weighted full 3D complex-field relative L2 <=1%; averaged Poynting-vector relative L2 <=2% on fixed physical comparison planes |
| Reference adequacy | Finest uniform result is a reference, not truth. Three spatial levels plus independent time/PML/domain refinements; estimated reference uncertainty <=1/3 of each claimed tolerance; otherwise mark the comparison unresolved |
| GPU benefit | Target >=10x faster and >=10x lower forward memory than the best accuracy-matched uniform Warp configuration; explore 30–100x on larger sparse domains. The original >=1.5x / >=2x thresholds are early prototype gates, not the project objective. Report measured peak device memory separately from owned-array accounting and publish losses too. |
| Incremental LTS benefit | >=1.25x faster than the SAME refined mesh with global fine stepping at the SAME error budget, including exchange and scheduling; publish fine-volume/patch-size break-even curves |
| Adjoint readiness | Fixed-hierarchy transpose tests; at least five independent random directional finite differences over three step sizes, best resolved errors <1e-3; full-gradient comparison where affordable; changed-design replay |

Reflection isolation needs a reference propagation run and incident/reflected
mode separation, with a measured numerical noise floor below the threshold.
Sphere and slab tests need adequately converged analytic evaluation and boundary
truncation. Do not infer scattering solely from `1-T`. For field comparisons,
map staggered phasors to common physical locations with a validated reconstruction,
and include phase error; no fitted phase shift to conceal it. Report both global
and interface-region errors. Small R needs an absolute rather than relative gate.
The present 10 nm bend and stored 2D slices do not yet meet this full reference
contract; generate the required full-volume observables before claiming a win.

Performance must count source/detector work, halo exchange, interpolation,
substep scheduling, and CPML; dynamic runs also count marking, remapping,
compaction and graph rebuilds. Report warm trajectory and end-to-end time
separately, plus peak allocated memory, including covered parent cells, halos,
metadata and temporary buffers. Use the same GPU partition, dtype and complete
physical trajectory; at least five synchronized serial repetitions with raw
samples and dispersion. Repeated cells-per-second is only a diagnostic.

Our own idealized cost estimate illustrates why fine-volume fraction matters.
For coarse/fine ratio r and a fine physical-volume fraction f, relative storage
is approximately `f + (1-f)/r^3`; relative update work with ideal LTS is
`f + (1-f)/r^4`, compared with uniform fine stepping. At r=2 and f=0.1 this
suggests ceilings of 4.71x memory reduction and 6.4x work reduction. These ignore
interfaces, coarse coverage, differing stable CFLs and GPU overhead, and are
not expected speedups. Measure f after geometry and error buffers are included.

## Proposed issue sequence

The linked issues carry dependencies, experiment boundaries, citations and
pass/no-go tests. All are authorized; implementation proceeds in dependency order.

| Issue | Proposed experiment |
| --- | --- |
| [harness](https://github.com/alecjacobson/warp-fdtd/issues/13) | [P0] Define adaptive Maxwell verification and accuracy-per-cost gates |
| [controls](https://github.com/alecjacobson/warp-fdtd/issues/14) | [P1] Compare rectilinear grids and material-interface accuracy as simpler controls |
| [static](https://github.com/alecjacobson/warp-fdtd/issues/15) | [P1] Test energy-stable 3D block subgridding with a global fine time step |
| [lts](https://github.com/alecjacobson/warp-fdtd/issues/16) | [P1] Evaluate stable local time stepping on a fixed refined mesh |
| [sbp](https://github.com/alecjacobson/warp-fdtd/issues/17) | [P2] Evaluate 2:1 SBP-SAT interfaces and higher-order block interiors |
| [selection](https://github.com/alecjacobson/warp-fdtd/issues/18) | [P2] Select refined regions from error indicators between solves |
| [adjoint](https://github.com/alecjacobson/warp-fdtd/issues/19) | [P2] Validate the discrete adjoint of a frozen adaptive hierarchy |
| [dg](https://github.com/alecjacobson/warp-fdtd/issues/20) | [P3] Benchmark a small hp-DG Maxwell alternative before a solver rewrite |
| [dynamic](https://github.com/alecjacobson/warp-fdtd/issues/21) | [P3] Test dynamic block AMR on moving 3D pulses with conservative transfers |

Recommended first tranche: verification/reference contract, the simpler
rectilinear/material controls, and the fixed 3D subgrid experiment. Then decide
whether its memory/accuracy benefit justifies LTS. Keep 2:1/high-order coupling,
hp-DG and dynamic remeshing as explicitly evaluated alternatives, not implied
commitments to implement every approach.
