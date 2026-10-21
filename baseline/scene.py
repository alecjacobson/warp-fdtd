"""Silicon bend adapted from FDTDX's MIT-licensed ceviche corner example.

Paper geometry: 4 x 4 x 1.5 um, 400 nm square guides, 1.6 um design square,
600 nm silica substrate, 1550 nm CW mode source. Source/monitor placement
and two-period flux windows follow the December 2024 example, commit a0a1c98.
Current solver API and optional current smooth parameterization are used.
"""

from dataclasses import dataclass
import math

import fdtdx as f
import jax
import jax.numpy as jnp
from fdtdx.core.jax.pytrees import autoinit
from fdtdx.core.jax.ste import straight_through_estimator
from fdtdx.materials import compute_allowed_permittivities


@autoinit
class PaperCWProfile(f.TemporalProfile):
    """Abrupt cosine used by the December 2024 source, without the modern ramp."""

    def get_amplitude(self, time, period, phase_shift=0.0):
        return jnp.cos(2 * jnp.pi * time / period + phase_shift)


@autoinit
class PaperBrush(f.BrushConstraint2D):
    """Preserve the 2024 brush's latent orientation and straight-through derivative.

    The old brush returned 1-generator(arr); the modern brush returns generator(arr)
    for an air background. Binary pretraining maps increasing latents toward air.
    Complement only the forward brush values to retain that historical convention.
    """

    def __call__(self, params, **kwargs):
        mapped = super().__call__(params, **kwargs)
        return jax.tree.map(lambda x: straight_through_estimator(x, 1 - x), mapped)


@autoinit
class PaperClosestIndex(f.ClosestIndex):
    """Historical isotropic quantizer with the current material table flattened.

    Upstream's inverse-permittivity option broadcasts (materials,1) against a
    scalar field, incorrectly adding a material axis to the output. This local
    adapter preserves the old flat lookup and straight-through estimator.
    """

    def __call__(self, params, **kwargs):
        allowed = 1 / jnp.asarray(
            compute_allowed_permittivities(self._materials, isotropic=True)
        ).reshape(-1)
        return jax.tree.map(
            lambda x: straight_through_estimator(
                x, jnp.argmin(jnp.abs(x[..., None] - allowed), axis=-1)
            ),
            params,
        )


@dataclass
class Settings:
    spacing_nm: float = 20
    time_fs: float = 200
    seed: int = 0
    backend: str = "gpu"
    gradient: str = "none"
    parameterization: str = "smooth"
    exact_interpolation: bool = False
    pml_cells: int = 10
    domain_um: float = 4
    voxel_nm: float = 20
    quality_monitors: bool = False
    precision: str = "float32"
    scene: str = "bend"
    source_profile: str = "current"
    pad_domain: bool = False
    modal_window: str = "rectangular"
    poynting_plane: bool = False


def modal_apodization(s, config, steps, period):
    if len(steps) < 8 * period:
        raise ValueError(
            "Modal quality evaluation requires at least eight optical periods"
        )
    if s.modal_window == "rectangular":
        return None
    return f.TukeyWindow(
        start_time=steps[-8 * period] * config.time_step_duration,
        end_time=steps[-1] * config.time_step_duration,
        alpha=1.0,
    )


def build(s: Settings):
    if s.scene == "straight":
        return build_straight(s)
    for extent_nm in (s.domain_um * 1000, 1500, 600, 400, 1600):
        if not math.isclose(
            extent_nm / s.spacing_nm, round(extent_nm / s.spacing_nm), abs_tol=1e-8
        ):
            raise ValueError(
                "Use a grid spacing that divides all geometry extents (e.g. 50, 25, 20, 10, 5 nm)"
            )
    config = f.SimulationConfig(
        time=s.time_fs * 1e-15,
        grid=f.UniformGrid(spacing=s.spacing_nm * 1e-9),
        dtype=getattr(jnp, s.precision),
        courant_factor=0.99,
        backend=s.backend,
    )
    if s.gradient != "none":
        if s.gradient == "compressed":
            gc = f.GradientConfig(
                recorder=f.Recorder(
                    modules=[
                        f.LinearReconstructEveryK(k=5),
                        f.DtypeConversion(dtype=jnp.float8_e4m3fnuz),
                    ]
                )
            )
        elif s.gradient == "paper":
            gc = f.GradientConfig(
                recorder=f.Recorder(
                    modules=[
                        f.LinearReconstructEveryK(k=2),
                        f.DtypeConversion(dtype=jnp.float16),
                    ]
                )
            )
        elif s.gradient == "exact":
            gc = f.GradientConfig(
                recorder=f.Recorder(modules=[]),
                recording_mode="segmented",
                num_checkpoints_reversible="auto",
            )
        elif s.gradient == "checkpointed":
            gc = f.GradientConfig(method="checkpointed", num_checkpoints=8)
        elif s.gradient == "adjoint":
            period = round(
                f.constants.wavelength_to_period(1.55e-6) / config.time_step_duration
            )
            gc = f.GradientConfig(
                method="adjoint",
                adjoint_mode="frequency",
                adjoint_window=f.OnOffSwitch(
                    fixed_on_time_steps=list(range(config.time_steps_total))[
                        -8 * period :
                    ]
                ),
            )
        else:
            raise ValueError(s.gradient)
        config = config.aset("gradient_config", gc)
    period = round(
        f.constants.wavelength_to_period(1.55e-6) / config.time_step_duration
    )
    steps = list(range(config.time_steps_total))
    if len(steps) <= 5 * period:
        raise ValueError("Simulation must cover input measurement window (>5 periods)")
    constraints = []
    # A PML convergence study grows the outside of all six faces, preserving
    # the physical active domain and translating every structure equally.
    extra = (
        max(0.0, s.pml_cells * s.spacing_nm * 1e-9 - 0.2e-6) if s.pad_domain else 0.0
    )
    volume = f.SimulationVolume(
        name="volume",
        partial_real_shape=(
            s.domain_um * 1e-6 + 2 * extra,
            s.domain_um * 1e-6 + 2 * extra,
            1.5e-6 + 2 * extra,
        ),
    )
    bounds, bc = f.boundary_objects_from_config(
        f.BoundaryConfig.from_uniform_bound(thickness=s.pml_cells), volume
    )
    objects = [volume, *bounds.values()]
    constraints.extend(bc)
    substrate = f.UniformMaterialObject(
        name="substrate",
        partial_real_shape=(None, None, 0.6e-6 + extra),
        material=f.Material(permittivity=2.25),
    )
    constraints.append(
        substrate.place_relative_to(
            volume, axes=2, own_positions=-1, other_positions=-1
        )
    )
    objects.append(substrate)
    materials = {
        "Air": f.Material(permittivity=1.0),
        "Silicon": f.Material(permittivity=12.25),
    }
    if s.parameterization == "smooth":
        transforms = [
            f.GaussianSmoothing2D(std_discrete=3),
            f.SubpixelSmoothedProjection(),
        ]
    elif s.parameterization == "binary":
        transforms = [f.StandardToInversePermittivityRange(), PaperClosestIndex()]
    elif s.parameterization == "brush":
        from fdtdx.objects.device.parameters.discretization import circular_brush

        transforms = [
            f.StandardToPlusOneMinusOneRange(),
            PaperBrush(brush=circular_brush(round(100 / s.voxel_nm)), axis=2),
        ]
    else:
        raise ValueError(s.parameterization)
    device = f.Device(
        name="Device",
        partial_real_shape=(1.6e-6, 1.6e-6, 0.4e-6),
        materials=materials,
        param_transforms=transforms,
        partial_voxel_real_shape=(s.voxel_nm * 1e-9, s.voxel_nm * 1e-9, 0.4e-6),
    )
    constraints.append(
        device.place_relative_to(
            substrate,
            axes=(0, 1, 2),
            own_positions=(1, -1, -1),
            other_positions=(1, -1, 1),
            grid_margins=(-s.pml_cells, s.pml_cells, 0),
            margins=(-0.2e-6, 0.2e-6, 0),
        )
    )
    objects.append(device)
    wi = f.UniformMaterialObject(
        name="input_guide",
        partial_real_shape=(None, 0.4e-6, 0.4e-6),
        material=materials["Silicon"],
    )
    wo = f.UniformMaterialObject(
        name="output_guide",
        partial_real_shape=(0.4e-6, None, 0.4e-6),
        material=materials["Silicon"],
    )
    constraints.extend(
        [
            wi.place_at_center(device, axes=1),
            wi.extend_to(device, axis=0, direction="+"),
            wi.place_above(substrate),
            wo.place_at_center(device, axes=0),
            wo.extend_to(device, axis=1, direction="-"),
            wo.place_above(substrate),
        ]
    )
    objects.extend([wi, wo])
    source = f.ModePlaneSource(
        name="source",
        partial_grid_shape=(1, None, None),
        wave_character=f.WaveCharacter(wavelength=1.55e-6),
        direction="+",
        temporal_profile=PaperCWProfile()
        if s.source_profile == "paper"
        else f.SingleFrequencyProfile(),
    )
    constraints.append(
        source.place_relative_to(
            wi,
            axes=(0,),
            other_positions=(-1,),
            own_positions=(1,),
            grid_margins=(s.pml_cells + 4,),
        )
    )
    objects.append(source)
    di = f.PoyntingFluxDetector(
        name="input",
        partial_grid_shape=(1, None, None),
        direction="+",
        exact_interpolation=s.exact_interpolation,
        switch=f.OnOffSwitch(fixed_on_time_steps=steps[3 * period : 5 * period]),
    )
    do = f.PoyntingFluxDetector(
        name="output",
        partial_grid_shape=(None, 1, None),
        direction="+",
        exact_interpolation=s.exact_interpolation,
        switch=f.OnOffSwitch(fixed_on_time_steps=steps[-2 * period :]),
    )
    constraints.extend(
        [
            di.place_relative_to(
                wi,
                axes=(0, 1, 2),
                own_positions=(-1, 0, 0),
                other_positions=(-1, 0, 0),
                grid_margins=(s.pml_cells, 0, 0),
                margins=(0.2e-6, 0, 0),
            ),
            di.size_relative_to(wi, axes=(1, 2)),
            do.place_relative_to(
                wo,
                axes=(0, 1, 2),
                own_positions=(0, 1, 0),
                other_positions=(0, 1, 0),
                grid_margins=(0, -s.pml_cells - 5, 0),
            ),
            do.size_relative_to(wo, axes=(0, 2)),
        ]
    )
    objects.extend([di, do])
    if s.quality_monitors:
        for name, guide, axis, direction, anchor, margin in (
            ("mode_in", wi, 0, "+", -1, s.pml_cells + round(200 / s.spacing_nm)),
            ("mode_reflected", wi, 0, "-", -1, s.pml_cells + round(200 / s.spacing_nm)),
            ("mode_out", wo, 1, "+", 1, -s.pml_cells - 5),
        ):
            shape = [None, None, None]
            shape[axis] = 1
            det = f.ModeOverlapDetector(
                name=name,
                partial_grid_shape=tuple(shape),
                direction=direction,
                wave_characters=(f.WaveCharacter(wavelength=1.55e-6),),
                mode_index=0,
                exact_interpolation=True,
                apodization=modal_apodization(s, config, steps, period),
                switch=f.OnOffSwitch(fixed_on_time_steps=steps[-8 * period :]),
            )
            constraints.append(
                det.place_relative_to(
                    guide,
                    axes=axis,
                    own_positions=anchor,
                    other_positions=anchor,
                    grid_margins=margin,
                )
            )
            transverse = tuple(a for a in range(3) if a != axis)
            constraints.extend(
                [
                    det.same_size(volume, axes=transverse),
                    det.place_at_center(volume, axes=transverse),
                ]
            )
            objects.append(det)
    if s.poynting_plane:
        flow = f.PhasorDetector(
            name="flow",
            partial_grid_shape=(None, None, 1),
            wave_characters=(f.WaveCharacter(wavelength=1.55e-6),),
            exact_interpolation=True,
            apodization=modal_apodization(s, config, steps, period),
            switch=f.OnOffSwitch(fixed_on_time_steps=steps[-8 * period :]),
        )
        objects.append(flow)
        constraints.extend(
            [flow.same_size(volume, axes=(0, 1)), flow.place_at_center(volume)]
        )
    return f.place_objects(objects, config, constraints, jax.random.PRNGKey(s.seed))


def build_straight(s):
    """Independent normalization check: a uniform waveguide should have T≈1, R≈0."""
    if s.gradient != "none":
        raise ValueError("The straight calibration has no design parameters")
    config = f.SimulationConfig(
        time=s.time_fs * 1e-15,
        grid=f.UniformGrid(spacing=s.spacing_nm * 1e-9),
        dtype=getattr(jnp, s.precision),
        courant_factor=0.99,
        backend=s.backend,
    )
    volume = f.SimulationVolume(
        name="volume",
        partial_real_shape=(s.domain_um * 1e-6, s.domain_um * 1e-6, 1.5e-6),
    )
    boundaries, constraints = f.boundary_objects_from_config(
        f.BoundaryConfig.from_uniform_bound(thickness=s.pml_cells), volume
    )
    substrate = f.UniformMaterialObject(
        name="substrate",
        partial_real_shape=(None, None, 0.6e-6),
        material=f.Material(permittivity=2.25),
    )
    guide = f.UniformMaterialObject(
        name="guide",
        partial_real_shape=(None, 0.4e-6, 0.4e-6),
        material=f.Material(permittivity=12.25),
    )
    constraints.extend(
        [
            substrate.place_relative_to(
                volume, axes=2, own_positions=-1, other_positions=-1
            ),
            guide.place_above(substrate),
            guide.place_at_center(volume, axes=1),
        ]
    )
    source = f.ModePlaneSource(
        name="source",
        partial_grid_shape=(1, None, None),
        wave_character=f.WaveCharacter(wavelength=1.55e-6),
        direction="+",
        temporal_profile=PaperCWProfile()
        if s.source_profile == "paper"
        else f.SingleFrequencyProfile(),
    )
    constraints.append(
        source.set_grid_coordinates(
            axes=(0,), sides=("-",), coordinates=(s.pml_cells + 3,)
        )
    )
    objects = [volume, *boundaries.values(), substrate, guide, source]
    period = round(
        f.constants.wavelength_to_period(1.55e-6) / config.time_step_duration
    )
    steps = list(range(config.time_steps_total))
    input_x = s.pml_cells + round(200 / s.spacing_nm)
    output_x = round(s.domain_um * 1000 / s.spacing_nm) - s.pml_cells - 6
    for name, x, window in (
        ("input", input_x, steps[3 * period : 5 * period]),
        ("output", output_x, steps[-2 * period :]),
    ):
        d = f.PoyntingFluxDetector(
            name=name,
            partial_grid_shape=(1, None, None),
            direction="+",
            exact_interpolation=s.exact_interpolation,
            switch=f.OnOffSwitch(fixed_on_time_steps=window),
        )
        constraints.extend(
            [
                d.set_grid_coordinates(axes=(0,), sides=("-",), coordinates=(x,)),
                d.place_at_center(guide, axes=(1, 2)),
                d.same_size(guide, axes=(1, 2)),
            ]
        )
        objects.append(d)
    for name, x, direction in (
        ("mode_in", input_x, "+"),
        ("mode_reflected", input_x, "-"),
        ("mode_out", output_x, "+"),
    ):
        d = f.ModeOverlapDetector(
            name=name,
            partial_grid_shape=(1, None, None),
            direction=direction,
            wave_characters=(f.WaveCharacter(wavelength=1.55e-6),),
            exact_interpolation=True,
            apodization=modal_apodization(s, config, steps, period),
            switch=f.OnOffSwitch(fixed_on_time_steps=steps[-8 * period :]),
        )
        constraints.append(
            d.set_grid_coordinates(axes=(0,), sides=("-",), coordinates=(x,))
        )
        objects.append(d)
    return f.place_objects(objects, config, constraints, jax.random.PRNGKey(s.seed))


def fluxes(arrays, objects=None):
    pin = arrays.detector_states["input"]["poynting_flux"].mean()
    pout = arrays.detector_states["output"]["poynting_flux"].mean()
    result = {"input_flux": pin, "output_flux": pout, "transmission": pout / pin}
    if "mode_in" in arrays.detector_states:
        powers = {
            name: jnp.abs(
                objects[name].compute_overlap(arrays.detector_states[name])
            ).squeeze()
            ** 2
            for name in ("mode_in", "mode_reflected", "mode_out")
        }
        result.update(powers)
        result["modal_transmission"] = powers["mode_out"] / powers["mode_in"]
        result["modal_reflection"] = powers["mode_reflected"] / powers["mode_in"]
    return result
