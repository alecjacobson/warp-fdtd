"""Passive full-volume phasors at fixed physical points, independent of mesh size.

Field storage remains Yee-staggered. Sample all six components at the same
physical point and center H in time before Hann-window accumulation. Sampling
is separate from solver benchmarking: its extra cost is always reported.
"""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import warp as wp

from .detectors import attach, measurements
from .solver import Solver, owned_array_bytes

wp.set_module_options({"enable_backward": False, "fast_math": False})


@wp.func
def trilinear(
    f: wp.array2d(dtype=float), component: int, p: wp.vec3, nx: int, ny: int, nz: int
):
    ix, iy, iz = int(wp.floor(p[0])), int(wp.floor(p[1])), int(wp.floor(p[2]))
    tx, ty, tz = p[0] - float(ix), p[1] - float(iy), p[2] - float(iz)
    value = float(0.0)
    for a in range(2):
        for b in range(2):
            for c in range(2):
                x, y, z = ix + a, iy + b, iz + c
                if x >= 0 and y >= 0 and z >= 0 and x < nx and y < ny and z < nz:
                    w = (
                        (tx if a else 1.0 - tx)
                        * (ty if b else 1.0 - ty)
                        * (tz if c else 1.0 - tz)
                    )
                    value += w * f[component, (x * ny + y) * nz + z]
    return value


@wp.func
def offset(component: int):
    if component == 0:
        return wp.vec3(0.5, 0.0, 0.0)
    if component == 1:
        return wp.vec3(0.0, 0.5, 0.0)
    if component == 2:
        return wp.vec3(0.0, 0.0, 0.5)
    if component == 3:
        return wp.vec3(0.0, 0.5, 0.5)
    if component == 4:
        return wp.vec3(0.5, 0.0, 0.5)
    return wp.vec3(0.5, 0.5, 0.0)


@wp.kernel
def record_volume(
    e: wp.array2d(dtype=float),
    h: wp.array2d(dtype=float),
    prev: wp.array2d(dtype=float),
    positions: wp.array(dtype=wp.vec3),
    weights: wp.array(dtype=wp.vec2),
    step: wp.array(dtype=int),
    phasor: wp.array2d(dtype=wp.vec2),
    nx: int,
    ny: int,
    nz: int,
    first: int,
):
    i = wp.tid()
    t = step[0]
    if t >= first:
        w = weights[t]
        for component in range(6):
            p = positions[i] - offset(component)
            value = float(0.0)
            if component < 3:
                value = trilinear(e, component, p, nx, ny, nz)
            else:
                value = 0.5 * (
                    trilinear(h, component - 3, p, nx, ny, nz)
                    + trilinear(prev, component - 3, p, nx, ny, nz)
                )
            phasor[component, i] += w * value


class VolumeMonitor:
    def __init__(self, solver, axes_m, window_monitor):
        self.axes_m = tuple(np.asarray(a, dtype=np.float64) for a in axes_m)
        if len(self.axes_m) != 3 or any(
            a.ndim != 1
            or not a.size
            or not np.isfinite(a).all()
            or np.any(np.diff(a) <= 0)
            for a in self.axes_m
        ):
            raise ValueError("Three finite increasing sampling axes required")
        self.shape = tuple(len(a) for a in self.axes_m)
        self.n = int(np.prod(self.shape))
        self.obj = {"class": "VolumeDiagnostic", "name": "volume"}
        self.name = "volume"
        self.first = window_monitor.first
        self.weights = window_monitor.weights
        self.exact = True
        # Nominal spacing avoids accumulated errors from float32 edge subtraction.
        dx = solver.meta["settings"]["spacing_nm"] * 1e-9
        axes = [
            (a - float(solver.data[c + "_edges_m"][0])) / dx
            for a, c in zip(self.axes_m, "xyz")
        ]
        if any(np.any(a < 0) or np.any(a > n) for a, n in zip(axes, solver.shape)):
            raise ValueError("Sampling points must lie within the physical grid")
        points = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
        self.positions = wp.array(
            points.astype("f"), dtype=wp.vec3, device=solver.device
        )
        self.phasor = wp.zeros((6, self.n), dtype=wp.vec2, device=solver.device)
        solver.monitors.append(self)
        solver.save_h_from = min(solver.save_h_from, self.first)

    def launch(self, solver):
        wp.launch(
            record_volume,
            self.n,
            inputs=[
                solver.E,
                solver.H,
                solver.hprev,
                self.positions,
                self.weights,
                solver.step,
                self.phasor,
                *solver.shape,
                self.first,
            ],
            device=solver.device,
        )

    def reset(self):
        self.phasor.zero_()

    def phasors(self):
        raw = self.phasor.numpy()
        return (raw[..., 0] + 1j * raw[..., 1]).reshape(6, *self.shape)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--case", default="results/warp_reference/bend_flow20")
    p.add_argument("--output", default="results/warp/volume20")
    p.add_argument("--sample-nm", type=float, default=20)
    args = p.parse_args()
    if not np.isfinite(args.sample_nm) or args.sample_nm <= 0:
        p.error("--sample-nm must be positive and finite")
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    s = Solver(args.case, cpml_storage="packed")
    attach(s)
    source_window = next((m for m in s.monitors if m.name == "mode_out"), None)
    if source_window is None:
        raise ValueError("A settled Hann mode_out detector is required")
    if s.meta["settings"]["modal_window"] != "hann":
        raise ValueError("A Hann window is required")
    extent = [
        float(s.data[c + "_edges_m"][-1] - s.data[c + "_edges_m"][0]) for c in "xyz"
    ]
    counts = [round(v / (args.sample_nm * 1e-9)) for v in extent]
    if min(counts) < 2 or not np.allclose(
        np.array(counts) * args.sample_nm * 1e-9, extent, rtol=1e-6
    ):
        raise ValueError("Sampling spacing must divide each physical domain extent")
    # Common physical points at the reference mesh E_z locations.
    axes = [
        (np.arange(n, dtype=float) + (0.5 if c == 2 else 0)) * args.sample_nm * 1e-9
        + float(s.data["xyz"[c] + "_edges_m"][0])
        for c, n in enumerate(counts)
    ]
    volume = VolumeMonitor(s, axes, source_window)
    compiled = s.compile()
    started = time.perf_counter()
    s.run()
    elapsed = time.perf_counter() - started
    phasors = volume.phasors()
    if not np.isfinite(phasors).all():
        raise FloatingPointError("Nonfinite volume phasors")
    measured = measurements(s)
    delta_t = abs(
        measured["modal_transmission"]
        - s.meta["reference_measurements"]["modal_transmission"]
    )
    if delta_t > 1e-5:
        raise AssertionError(f"Passive observer changed modal transmission: {delta_t}")
    checks = {"modal_transmission_absolute_difference": delta_t}
    flow = next((m for m in s.monitors if m.name == "flow"), None)
    if flow is not None and tuple(counts) == s.shape:
        z = flow.obj["grid_slice"][2][0]
        ref = flow.phasors().squeeze(-1)
        err = float(np.linalg.norm(phasors[..., z] - ref) / np.linalg.norm(ref))
        checks["plane_vs_existing_colocation_relative_l2"] = err
        if err > 2e-5:
            raise AssertionError(checks)
    cache = out / "volume-cache"
    cache.mkdir(exist_ok=True)
    files = []
    for z in range(0, counts[2], 8):
        path = cache / f"phasors-z{z:04d}.npz"
        np.savez_compressed(path, phasors=phasors[..., z : z + 8])
        files.append(
            {
                "file": str(path.relative_to(out)),
                "z_start": z,
                "z_stop": min(z + 8, counts[2]),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    record = dict(
        status="complete",
        diagnostic_only=True,
        case=args.case,
        inputs_sha256=s.meta["inputs_sha256"],
        solver_shape=s.shape,
        sample_shape=counts,
        sample_nm=args.sample_nm,
        axes_m=[a.tolist() for a in axes],
        time_steps=s.steps,
        dt_s=s.meta["dt_s"],
        measurements=measured,
        checks=checks,
        device=s.device.name,
        warp_version=wp.__version__,
        compiled_seconds=compiled,
        diagnostic_trajectory_seconds=elapsed,
        timing_scope="One synchronized trajectory including volume sampling; not a throughput benchmark",
        owned_device_array_bytes=owned_array_bytes(s),
        window_first_step=volume.first,
        window_weights_sha256=hashlib.sha256(
            volume.weights.numpy().tobytes()
        ).hexdigest(),
        code_sha256={
            name: hashlib.sha256(Path("warp_fdtd", name).read_bytes()).hexdigest()
            for name in ("solver.py", "kernels.py", "detectors.py", "volume.py")
        },
        files=files,
    )
    (out / "result.json").write_text(
        json.dumps(record, indent=2, allow_nan=False) + "\n"
    )
    print(
        json.dumps(
            {k: v for k, v in record.items() if k not in ("files", "axes_m")}, indent=2
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
