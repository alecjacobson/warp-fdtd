"""Physical contract for new bend convergence studies; legacy fixtures stay intact.

Coordinates are Yee-grid anchors in a common design frame, in nanometers.
Staggered component positions remain part of each discretization. The source's
phase origin is its grid anchor; the TFSF cut is staggered within its cell.
"""

import math

PROTOCOL = "fixed_physical_v1"


def grid_index(position_nm, spacing_nm):
    ratio = position_nm / spacing_nm
    if not math.isfinite(ratio) or not math.isclose(
        ratio, round(ratio), rel_tol=0, abs_tol=1e-7
    ):
        raise ValueError(
            f"{position_nm:g} nm is not representable on a {spacing_nm:g} nm grid"
        )
    return round(ratio)


def layout(settings):
    s = settings
    if s.reference_protocol not in ("legacy", PROTOCOL):
        raise ValueError("Unknown reference protocol")
    if not math.isfinite(s.courant_factor) or not 0 < s.courant_factor <= 0.99:
        raise ValueError("Require 0 < courant_factor <= 0.99")
    if s.reference_protocol == "legacy":
        return None
    if s.scene != "bend" or s.gradient != "none":
        raise ValueError(
            "fixed_physical_v1 currently supports forward bend references only"
        )
    if s.spacing_nm <= 0 or not math.isfinite(s.spacing_nm):
        raise ValueError("Positive finite spacing required")
    if s.domain_um < 4 or not math.isfinite(s.domain_um):
        raise ValueError("Fixed design requires domain_um >= 4")
    if s.parameterization != "smooth" or s.voxel_nm != 20:
        raise ValueError(
            "fixed_physical_v1 requires the saved smooth 20 nm latent design grid"
        )
    grid_index(s.voxel_nm, s.spacing_nm)
    pml_nm = s.pml_cells * s.spacing_nm
    if pml_nm < 200 or (not s.pad_domain and not math.isclose(pml_nm, 200)):
        raise ValueError(
            "Use 200 nm PML, or --pad-domain when increasing its thickness"
        )
    if not math.isfinite(s.modal_window_fs) or not 0 < s.modal_window_fs < s.time_fs:
        raise ValueError(
            "Physical modal window must fit strictly inside the trajectory"
        )
    extra = pml_nm - 200 if s.pad_domain else 0
    xy_extra = 0.5 * (s.domain_um * 1000 - 4000)
    origin = (extra + xy_extra, extra + xy_extra, extra)
    base = {
        "source": (0, 300.0),
        "input": (0, 400.0),
        "output": (1, 3600.0),
        "mode_in": (0, 400.0),
        "mode_reflected": (0, 400.0),
        "mode_out": (1, 3600.0),
    }
    planes = {
        name: {
            "axis": axis,
            "design_anchor_nm": value,
            "grid_index": grid_index(value + origin[axis], s.spacing_nm),
        }
        for name, (axis, value) in base.items()
    }
    device_start = tuple(
        grid_index(v + o, s.spacing_nm) for v, o in zip((2000, 400, 600), origin)
    )
    for value in (
        *origin,
        1600,
        400,
        600 + extra,
        1500 + 2 * extra,
        s.domain_um * 1000 + 2 * extra,
    ):
        grid_index(value, s.spacing_nm)
    return {
        "name": PROTOCOL,
        "origin_nm": origin,
        "planes": planes,
        "device_start": device_start,
        "device_design_bounds_nm": [[2000, 3600], [400, 2000], [600, 1000]],
        "pml_thickness_nm": pml_nm,
        "pml_padding_nm": extra,
        "modal_window_fs": [s.time_fs - s.modal_window_fs, s.time_fs],
        "source_phase_origin": "Yee grid anchor; TFSF cut is staggered within the source cell",
        "material_policy": "Fixed latent voxel grid and filter; upstream projection is reevaluated at each field resolution",
    }


def modal_steps(settings, dt, total_steps):
    if settings.reference_protocol == "legacy":
        period = round((1550e-9 / 299792458.0) / dt)
        return list(range(total_steps))[-8 * period :]
    lo = (settings.time_fs - settings.modal_window_fs) * 1e-15
    hi = settings.time_fs * 1e-15
    return [i for i in range(total_steps) if lo <= i * dt <= hi]


def audit_placed(settings, objects, config):
    contract = layout(settings)
    if contract is None:
        return None
    dx = settings.spacing_nm
    geometry = {}
    for obj in objects.objects:
        if obj.name in ("Device", "input_guide", "output_guide", "substrate"):
            bounds = [
                [sl.start * dx - o, sl.stop * dx - o]
                for sl, o in zip(obj.grid_slice, contract["origin_nm"])
            ]
            geometry[obj.name] = bounds
    if geometry.get("Device") is None or any(
        abs(a - b) > 1e-6
        for actual, expected in zip(
            geometry["Device"], contract["device_design_bounds_nm"]
        )
        for a, b in zip(actual, expected)
    ):
        raise AssertionError(
            f"Device placement differs from physical contract: {geometry}"
        )
    resolved = {}
    for name, expected in contract["planes"].items():
        if name not in [o.name for o in objects.objects]:
            continue
        obj = objects[name]
        axis = expected["axis"]
        sl = obj.grid_slice[axis]
        if sl.start != expected["grid_index"] or sl.stop != sl.start + 1:
            raise AssertionError(
                f"Plane {name} placement differs from physical contract"
            )
        resolved[name] = {
            "axis": axis,
            "design_anchor_nm": sl.start * dx - contract["origin_nm"][axis],
        }
    samples = modal_steps(settings, config.time_step_duration, config.time_steps_total)
    return {
        **contract,
        "geometry_design_bounds_nm": geometry,
        "resolved_planes": resolved,
        "actual_duration_fs": config.time_steps_total
        * config.time_step_duration
        * 1e15,
        "actual_modal_sample_range_fs": [
            samples[0] * config.time_step_duration * 1e15,
            samples[-1] * config.time_step_duration * 1e15,
        ],
    }
