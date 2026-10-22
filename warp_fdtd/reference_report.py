"""Controlled flagship mesh trends and nuisance controls, without a certified bound.

Read the passive Warp volume observations, verify their inputs and checksums,
then compare incident-normalized phasors on common physical points. The modal
measurements are the independently evaluated FDTDX reference measurements.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

ROI_NM = ((200, 3800), (200, 3800), (200, 1300))


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def common_indices(axes_m, origin_nm, target_nm, grid_origin_m=(0.0, 0.0, 0.0)):
    """Locate exact shared physical samples; never interpolate between meshes."""
    indices = []
    for axis, origin, target, grid_origin in zip(
        axes_m, origin_nm, target_nm, grid_origin_m
    ):
        physical = (np.asarray(axis) - grid_origin) * 1e9 - origin
        target = np.asarray(target)
        idx = np.searchsorted(physical, target - 1e-5)
        if np.any(idx >= len(physical)) or not np.allclose(
            physical[idx], target, rtol=0, atol=1e-4
        ):
            raise ValueError(
                "Volume does not contain the declared common physical points"
            )
        indices.append(idx)
    return indices


def load_volume(root, name, targets):
    directory = root / "volumes" / name
    path = directory / "result.json"
    r = json.loads(path.read_text())
    bundle = root / "bundles" / name
    m = json.loads((bundle / "manifest.json").read_text())
    if digest(root / name / "result.json") != m["reference_result_sha256"]:
        raise ValueError(f"Stale reference export {bundle}")
    if r["inputs_sha256"] != m["inputs_sha256"]:
        raise ValueError(f"Stale volume {directory}")
    if digest(bundle / "inputs.npz") != m["inputs_sha256"]:
        raise ValueError(f"Input checksum mismatch: {bundle}")
    if r["physical_contract"]["name"] != "fixed_physical_v1":
        raise ValueError("Fixed physical reference protocol required")
    with np.load(bundle / "inputs.npz") as z:
        grid_origin = [float(z[c + "_edges_m"][0]) for c in "xyz"]
    idx = common_indices(
        r["axes_m"], r["physical_contract"]["origin_nm"], targets, grid_origin
    )
    result = np.empty((6, *(len(i) for i in idx)), np.complex64)
    covered = np.zeros(len(idx[2]), bool)
    for chunk in r["files"]:
        file = directory / chunk["file"]
        if digest(file) != chunk["sha256"]:
            raise ValueError(f"Volume checksum mismatch: {file}")
        selected = np.flatnonzero(
            (idx[2] >= chunk["z_start"]) & (idx[2] < chunk["z_stop"])
        )
        if not selected.size:
            continue
        if np.any(covered[selected]):
            raise ValueError("Overlapping volume chunks")
        with np.load(file) as z:
            result[..., selected] = z["phasors"][
                np.ix_(
                    np.arange(6), idx[0], idx[1], idx[2][selected] - chunk["z_start"]
                )
            ]
        covered[selected] = True
    if not covered.all():
        raise ValueError("Incomplete volume chunks")
    result /= complex(*r["incident_normalization"]["field_divisor"])
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite normalized fields")
    return r, m, result


def poynting(fields):
    return 0.5 * np.cross(fields[:3], fields[3:].conj(), axis=0).real


def relative_l2(delta, reference, weight=1.0):
    numerator = np.sum(weight * np.abs(delta.astype(np.complex128)) ** 2)
    denominator = np.sum(weight * np.abs(reference.astype(np.complex128)) ** 2)
    if denominator <= 0:
        raise ValueError("Reference norm is zero")
    return float(np.sqrt(numerator / denominator))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path("results/physical_reference"))
    p.add_argument("--mesh-cases", nargs=3, default=["mesh20", "mesh10", "mesh6p67"])
    p.add_argument(
        "--controls",
        nargs="*",
        default=["time20_450", "cfl20_half", "pml20_400", "domain20_4p4"],
    )
    args = p.parse_args()
    targets = [
        np.arange(a + (10 if c == 2 else 0), b + (0 if c == 2 else 1), 20.0)
        for c, (a, b) in enumerate(ROI_NM)
    ]
    if len(set(args.mesh_cases)) != 3:
        p.error("Three distinct mesh cases required")
    spacings = [
        json.loads((args.root / "bundles" / n / "manifest.json").read_text())[
            "settings"
        ]["spacing_nm"]
        for n in args.mesh_cases
    ]
    if not all(a > b for a, b in zip(spacings, spacings[1:])):
        p.error("Mesh cases must be ordered from coarse to fine")
    finest = args.mesh_cases[-1]
    ref, manifest, reference = load_volume(args.root, finest, targets)
    origin = np.asarray(ref["physical_contract"]["origin_nm"])
    dx = manifest["settings"]["spacing_nm"]
    # Fixed fine-grid scalar material weights at the cell containing each point.
    # These are norm weights, not an extra interpolated constitutive model.
    cell_indices = [
        np.floor((a + o) / dx + 1e-8).astype(int) for a, o in zip(targets, origin)
    ]
    with np.load(args.root / "bundles" / finest / "inputs.npz") as z:
        eps = 1.0 / z["inv_permittivity"][np.ix_([0], *cell_indices)]
    flow_ref = poynting(reference)
    rows = []
    slices = []
    slice_z = int(np.argmin(np.abs(targets[2] - 800)))
    coarse = None
    for name in [*args.mesh_cases, *args.controls]:
        r, m, fields = (
            (ref, manifest, reference)
            if name == finest
            else load_volume(args.root, name, targets)
        )
        if (m["parameter_sha256"], m["beta"]) != (
            manifest["parameter_sha256"],
            manifest["beta"],
        ):
            raise ValueError("Candidate mismatch")
        if (
            r["incident_normalization"]["canonical_mode_component"]
            != ref["incident_normalization"]["canonical_mode_component"]
        ):
            raise ValueError("Dominant input polarization changed across meshes")
        if name == args.mesh_cases[0]:
            coarse = fields
        comparison = reference if name in args.mesh_cases else coarse
        flow_comparison = flow_ref if name in args.mesh_cases else poynting(coarse)
        baseline_name = finest if name in args.mesh_cases else args.mesh_cases[0]
        baseline_manifest = json.loads(
            (args.root / "bundles" / baseline_name / "manifest.json").read_text()
        )
        transmission = m["reference_measurements"]["modal_transmission"]
        if name in args.mesh_cases:
            slices.append(poynting(fields)[..., slice_z].copy())
        rows.append(
            dict(
                case=name,
                compared_with=baseline_name,
                spacing_nm=m["settings"]["spacing_nm"],
                cells=int(np.prod(r["solver_shape"])),
                steps=r["time_steps"],
                dt_s=r["dt_s"],
                modal_transmission=transmission,
                modal_reflection=m["reference_measurements"]["modal_reflection"],
                modal_reflection_absolute_difference=abs(
                    m["reference_measurements"]["modal_reflection"]
                    - baseline_manifest["reference_measurements"]["modal_reflection"]
                ),
                modal_transmission_absolute_difference=abs(
                    transmission
                    - baseline_manifest["reference_measurements"]["modal_transmission"]
                ),
                electric_energy_weighted_relative_l2=relative_l2(
                    fields[:3] - comparison[:3], comparison[:3], eps
                ),
                magnetic_relative_l2=relative_l2(
                    fields[3:] - comparison[3:], comparison[3:]
                ),
                poynting_relative_l2=relative_l2(
                    poynting(fields) - flow_comparison, flow_comparison
                ),
                volume_record_sha256=digest(
                    args.root / "volumes" / name / "result.json"
                ),
                bundle_manifest_sha256=digest(
                    args.root / "bundles" / name / "manifest.json"
                ),
            )
        )
    out = args.root / "comparison"
    out.mkdir(exist_ok=True)
    record = dict(
        status="complete",
        reference=finest,
        certified_reference=False,
        uncertainty_status="Mesh differences and coarse-grid nuisance controls only. These are not a continuum error bound; fine-grid controls and asymptotic verification remain required.",
        parameter_sha256=manifest["parameter_sha256"],
        beta=manifest["beta"],
        region_nm=ROI_NM,
        sample_nm=20,
        sample_shape=[len(a) for a in targets],
        normalization="Unit area-weighted incident mode power and input eigenmode phase; no fit to propagated fields.",
        electric_norm="Fine-grid scalar epsilon, piecewise constant in the cell containing each common physical sample.",
        timing_scope="Reference generation and passive volume diagnostics are excluded from performance claims.",
        code_sha256={
            n: digest(Path("warp_fdtd", n))
            for n in ("reference_report.py", "reference_fields.py")
        },
        rows=rows,
    )
    (out / "result.json").write_text(
        json.dumps(record, indent=2, allow_nan=False) + "\n"
    )
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), layout="constrained")
    mesh = rows[:3]
    x = [r["spacing_nm"] for r in mesh]
    axes[0].plot(x, [100 * r["modal_transmission"] for r in mesh], "o-")
    axes[0].set(
        xlabel="Mesh spacing (nm)",
        ylabel="Modal transmission (%)",
        title="Fixed 3D bend and 40 fs window",
    )
    for key, label in [
        ("electric_energy_weighted_relative_l2", "Electric field"),
        ("poynting_relative_l2", "Poynting vector"),
    ]:
        axes[1].loglog(x[:-1], [r[key] for r in mesh[:-1]], "o-", label=label)
    axes[1].set(
        xlabel="Mesh spacing (nm)",
        ylabel="Relative L2 difference",
        title=f"Against {dx:.3g} nm reference",
    )
    axes[1].legend()
    controls = rows[3:]
    axes[2].bar(
        np.arange(len(controls)),
        [100 * r["modal_transmission_absolute_difference"] for r in controls],
    )
    axes[2].set_xticks(
        np.arange(len(controls)),
        [
            {
                "time20_450": "450 fs",
                "cfl20_half": "Half dt",
                "pml20_400": "400 nm PML",
                "domain20_4p4": "4.4 µm domain",
            }.get(r["case"], r["case"])
            for r in controls
        ],
        rotation=20,
    )
    axes[2].set(
        ylabel="Transmission change (percentage points)",
        title="20 nm controls; not fine-grid bounds",
    )
    for ax in axes:
        ax.grid(alpha=0.2)
    fig.savefig(out / "convergence.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.4), layout="constrained")
    peak = max(float(np.linalg.norm(slices[-1], axis=0).max()), 1e-30)
    grid_x, grid_y = np.meshgrid(targets[0] / 1000, targets[1] / 1000, indexing="ij")
    stride = (slice(None, None, 12), slice(None, None, 12))
    for ax, flow, row in zip(axes, slices, mesh):
        magnitude = np.linalg.norm(flow, axis=0)
        plotted = ax.pcolormesh(
            grid_x,
            grid_y,
            np.log10(np.maximum(magnitude / peak, 1e-6)),
            shading="auto",
            vmin=-5,
            vmax=0,
            cmap="inferno",
        )
        planar = np.maximum(np.linalg.norm(flow[:2], axis=0), peak * 1e-8)
        mask = magnitude[stride] / peak > 1e-3
        ax.quiver(
            grid_x[stride][mask],
            grid_y[stride][mask],
            (flow[0] / planar)[stride][mask],
            (flow[1] / planar)[stride][mask],
            color="white",
            alpha=0.65,
            scale=23,
            width=0.004,
        )
        ax.set(
            aspect="equal",
            xlabel="Design x (µm)",
            ylabel="Design y (µm)",
            title=f"{row['spacing_nm']:.3g} nm · T={100 * row['modal_transmission']:.4f}%",
        )
    fig.colorbar(
        plotted,
        ax=axes,
        label="log10(|time-averaged S| / fine-slice peak)",
        shrink=0.75,
    )
    fig.suptitle(
        f"Controlled 3D bend: common z={targets[2][slice_z]:.0f} nm slice, unit incident power"
    )
    fig.savefig(out / "power_flow.png", dpi=180)
    plt.close(fig)
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
