"""Colocated field/flux detectors; all time accumulation runs in Warp."""

import numpy as np
import warp as wp
from .kernels import sample

wp.set_module_options({"enable_backward": False, "fast_math": False})


@wp.func
def ht(
    h: wp.array2d(dtype=float),
    prev: wp.array2d(dtype=float),
    x: int,
    y: int,
    z: int,
    nx: int,
    ny: int,
    nz: int,
):
    return 0.5 * (
        sample(h, x, y, z, nx, ny, nz, 0) + sample(prev, x, y, z, nx, ny, nz, 0)
    )


@wp.func
def colocate(
    e: wp.array2d(dtype=float),
    h: wp.array2d(dtype=float),
    prev: wp.array2d(dtype=float),
    x: int,
    y: int,
    z: int,
    nx: int,
    ny: int,
    nz: int,
):
    e0 = sample(e, x, y, z, nx, ny, nz, 0)
    ex = sample(e, x - 1, y, z, nx, ny, nz, 0)
    ey = sample(e, x, y - 1, z, nx, ny, nz, 0)
    ez = sample(e, x, y, z + 1, nx, ny, nz, 0)
    exz = sample(e, x - 1, y, z + 1, nx, ny, nz, 0)
    eyz = sample(e, x, y - 1, z + 1, nx, ny, nz, 0)
    ec = wp.vec3(
        0.25 * (e0[0] + ex[0] + ez[0] + exz[0]),
        0.25 * (e0[1] + ey[1] + ez[1] + eyz[1]),
        e0[2],
    )
    h0 = ht(h, prev, x, y, z, nx, ny, nz)
    hx = ht(h, prev, x - 1, y, z, nx, ny, nz)
    hy = ht(h, prev, x, y - 1, z, nx, ny, nz)
    hxy = ht(h, prev, x - 1, y - 1, z, nx, ny, nz)
    hz = ht(h, prev, x, y, z + 1, nx, ny, nz)
    hxz = ht(h, prev, x - 1, y, z + 1, nx, ny, nz)
    hyz = ht(h, prev, x, y - 1, z + 1, nx, ny, nz)
    hxyz = ht(h, prev, x - 1, y - 1, z + 1, nx, ny, nz)
    hc = wp.vec3(
        0.5 * (h0[0] + hy[0]),
        0.5 * (h0[1] + hx[1]),
        0.125 * (h0[2] + hx[2] + hy[2] + hxy[2] + hz[2] + hxz[2] + hyz[2] + hxyz[2]),
    )
    return ec, hc


@wp.kernel
def record(
    e: wp.array2d(dtype=float),
    h: wp.array2d(dtype=float),
    prev: wp.array2d(dtype=float),
    step: wp.array(dtype=int),
    weights: wp.array(dtype=wp.vec2),
    indices: wp.array(dtype=int),
    phasor: wp.array2d(dtype=wp.vec2),
    flux: wp.array(dtype=float),
    area: wp.array(dtype=float),
    nx: int,
    ny: int,
    nz: int,
    x0: int,
    y0: int,
    z0: int,
    sy: int,
    sz: int,
    axis: int,
    exact: int,
    is_phasor: int,
):
    i = wp.tid()
    t = step[0]
    index = indices[t]
    if index >= 0:
        x = x0 + i // (sy * sz)
        y = y0 + (i // sz) % sy
        z = z0 + i % sz
        ee = sample(e, x, y, z, nx, ny, nz, 0)
        hh = sample(h, x, y, z, nx, ny, nz, 0)
        if exact != 0:
            ee, hh = colocate(e, h, prev, x, y, z, nx, ny, nz)
        if is_phasor != 0:
            w = weights[t]
            for c in range(3):
                phasor[c, i] += w * ee[c]
                phasor[c + 3, i] += w * hh[c]
        else:
            power = wp.cross(ee, hh)[axis] * area[i]
            wp.atomic_add(flux, index, power)


class Monitor:
    def __init__(self, solver, obj):
        self.obj = obj
        self.name = obj["name"]
        self.is_phasor = obj["class"] in ("ModeOverlapDetector", "PhasorDetector")
        if obj["class"] not in (
            "ModeOverlapDetector",
            "PhasorDetector",
            "PoyntingFluxDetector",
        ):
            raise ValueError(f"Unsupported detector {obj['class']}")
        self.shape = tuple(hi - lo for lo, hi in obj["grid_slice"])
        self.n = int(np.prod(self.shape))
        self.exact = self.is_phasor or solver.meta["settings"].get(
            "exact_interpolation", False
        )
        indices = solver.data[self.name + "/_time_step_to_arr_idx"]
        self.indices = wp.array(indices, dtype=int, device=solver.device)
        self.first = int(np.flatnonzero(indices >= 0)[0])
        weights = np.zeros(solver.steps, np.complex64)
        if self.is_phasor:
            src = next(
                o for o in solver.meta["objects"] if o["class"] == "ModePlaneSource"
            )
            t = np.arange(solver.steps, dtype=np.float32) * np.float32(
                solver.meta["dt_s"]
            )
            phase = np.float32(2 * np.pi * src["frequency_hz"]) * t
            weights = (
                np.exp(1j * phase).astype(np.complex64)
                * np.float32(2 / obj["window_sum"])
                * solver.data[self.name + "/_window_at_time_step_arr"]
            )
        self.weights = wp.array(
            np.column_stack((weights.real, weights.imag)),
            dtype=wp.vec2,
            device=solver.device,
        )
        self.phasor = wp.zeros(
            (6, self.n if self.is_phasor else 1), dtype=wp.vec2, device=solver.device
        )
        self.flux = wp.zeros(int(indices.max()) + 1, dtype=float, device=solver.device)
        self.axis = obj.get("propagation_axis", 2)
        widths = []
        for a, axis in enumerate("xyz"):
            lo, hi = obj["grid_slice"][a]
            v = (
                np.diff(solver.data[axis + "_edges_m"])[lo:hi]
                if a != self.axis
                else np.ones(hi - lo, np.float32)
            )
            sh = [1, 1, 1]
            sh[a] = len(v)
            widths.append(v.reshape(sh))
        self.area_numpy = np.broadcast_to(
            widths[0] * widths[1] * widths[2], self.shape
        ).copy()
        if obj["class"] == "ModeOverlapDetector":
            self.area_numpy /= self.area_numpy.mean()
        self.area = wp.array(self.area_numpy.ravel(), dtype=float, device=solver.device)

    def reset(self):
        self.phasor.zero_()
        self.flux.zero_()

    def launch(self, s):
        lo = [a for a, b in self.obj["grid_slice"]]
        wp.launch(
            record,
            self.n,
            inputs=[
                s.E,
                s.H,
                s.hprev,
                s.step,
                self.weights,
                self.indices,
                self.phasor,
                self.flux,
                self.area,
                *s.shape,
                *lo,
                *self.shape[1:],
                self.axis,
                int(self.exact),
                int(self.is_phasor),
            ],
            device=s.device,
        )

    def phasors(self):
        raw = self.phasor.numpy()
        return (raw[..., 0] + 1j * raw[..., 1]).reshape(6, *self.shape)

    def power(self, s):
        ph = self.phasors()
        em = s.data[self.name + "/_mode_E"][0]
        hm = s.data[self.name + "/_mode_H"][0]
        integrand = (
            np.cross(em, ph[3:].conj(), axis=0) + np.cross(ph[:3].conj(), hm, axis=0)
        )[self.axis]
        alpha = np.sum(integrand * self.area_numpy) / 4
        return float(abs(alpha) ** 2)


def attach(solver):
    for obj in solver.meta["objects"]:
        if obj["class"].endswith("Detector"):
            m = Monitor(solver, obj)
            solver.monitors.append(m)
            if m.exact:
                solver.save_h_from = min(solver.save_h_from, m.first)


def measurements(solver):
    powers = {
        m.name: m.power(solver)
        for m in solver.monitors
        if m.obj["class"] == "ModeOverlapDetector"
    }
    if powers:
        powers["modal_transmission"] = powers["mode_out"] / powers["mode_in"]
        powers["modal_reflection"] = powers["mode_reflected"] / powers["mode_in"]
    return powers
