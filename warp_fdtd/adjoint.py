"""Exact discrete material adjoint, storing only curls in the design volume.

The transpose includes the CPML state and detector spatial/time interpolation.
Storage is O(steps * design cells), not O(steps * simulation cells). Only the
fixed 20 nm smooth bend parameterization is currently exposed by the driver.
"""

import numpy as np
import warp as wp
from .solver import Solver
from . import kernels as k

wp.set_module_options({"enable_backward": False, "fast_math": False})
vec6 = wp.types.vector(length=6, dtype=wp.float32)


@wp.kernel
def record_curl(
    h: wp.array2d(dtype=float),
    hist: wp.array3d(dtype=float),
    step: wp.array(dtype=int),
    nx: int,
    ny: int,
    nz: int,
    x0: int,
    y0: int,
    z0: int,
    sx: int,
    sy: int,
    sz: int,
    courant: float,
):
    i = wp.tid()
    x = x0 + i // (sy * sz)
    y = y0 + (i // sz) % sy
    z = z0 + i % sz
    v = k.sample(h, x, y, z, nx, ny, nz, 0)
    dx = v - k.sample(h, x - 1, y, z, nx, ny, nz, 0)
    dy = v - k.sample(h, x, y - 1, z, nx, ny, nz, 0)
    dz = v - k.sample(h, x, y, z - 1, nx, ny, nz, 0)
    value = courant * wp.vec3(dy[2] - dz[1], dz[0] - dx[2], dx[1] - dy[0])
    for c in range(3):
        hist[step[0], c, i] = value[c]


@wp.kernel
def prepare(
    q: wp.array2d(dtype=float),
    eps: wp.array(dtype=float),
    psi: wp.array2d(dtype=float),
    coeff: wp.array2d(dtype=wp.vec3),
    d: wp.array2d(dtype=float),
    nx: int,
    ny: int,
    nz: int,
    courant: float,
    electric: int,
):
    i = wp.tid()
    x = i // (ny * nz)
    y = (i // nz) % ny
    z = i % nz
    factor = -courant
    if electric != 0:
        factor = courant * eps[i]
    v = factor * wp.vec3(q[0, i], q[1, i], q[2, i])
    for axis in range(3):
        pos = x
        if axis == 1:
            pos = y
        elif axis == 2:
            pos = z
        co = coeff[axis, pos]
        a = (axis + 1) % 3
        b = (axis + 2) % 3
        q1 = -v[a]
        q2 = v[b]
        d1 = co[2] * q1
        d2 = co[2] * q2
        if co[0] != 0.0:
            p1 = psi[2 * axis, i] + q1
            p2 = psi[2 * axis + 1, i] + q2
            d1 += co[0] * p1
            d2 += co[0] * p2
            psi[2 * axis, i] = co[1] * p1
            psi[2 * axis + 1, i] = co[1] * p2
        d[2 * axis, i] = d1
        d[2 * axis + 1, i] = d2


@wp.func
def read(
    d: wp.array2d(dtype=float),
    slot: int,
    x: int,
    y: int,
    z: int,
    nx: int,
    ny: int,
    nz: int,
):
    if x < 0 or y < 0 or z < 0 or x >= nx or y >= ny or z >= nz:
        return float(0.0)
    return d[slot, (x * ny + y) * nz + z]


@wp.kernel
def gather(
    q: wp.array2d(dtype=float),
    d: wp.array2d(dtype=float),
    hist: wp.array3d(dtype=float),
    grad: wp.array(dtype=float),
    step: wp.array(dtype=int),
    nx: int,
    ny: int,
    nz: int,
    x0: int,
    y0: int,
    z0: int,
    sx: int,
    sy: int,
    sz: int,
    from_h: int,
):
    i = wp.tid()
    x = i // (ny * nz)
    y = (i // nz) % ny
    z = i % nz
    delta = int(1)
    if from_h != 0:
        delta = -1
    v = wp.vec3(0.0)
    # D_forward^T = shift(-1)-I, D_backward^T = I-shift(+1).
    v[0] = (read(d, 2, x, y + delta, z, nx, ny, nz) - d[2, i]) + (
        read(d, 5, x, y, z + delta, nx, ny, nz) - d[5, i]
    )
    v[1] = (read(d, 1, x + delta, y, z, nx, ny, nz) - d[1, i]) + (
        read(d, 4, x, y, z + delta, nx, ny, nz) - d[4, i]
    )
    v[2] = (read(d, 0, x + delta, y, z, nx, ny, nz) - d[0, i]) + (
        read(d, 3, x, y + delta, z, nx, ny, nz) - d[3, i]
    )
    if from_h == 0:
        v = -v
    value = wp.vec3(q[0, i], q[1, i], q[2, i]) + v
    for c in range(3):
        q[c, i] = value[c]
    if (
        from_h != 0
        and x >= x0
        and x < x0 + sx
        and y >= y0
        and y < y0 + sy
        and z >= z0
        and z < z0 + sz
    ):
        j = ((x - x0) * sy + y - y0) * sz + z - z0
        grad[j] += (
            hist[step[0], 0, j] * value[0]
            + hist[step[0], 1, j] * value[1]
            + hist[step[0], 2, j] * value[2]
        )


@wp.func
def add(
    q: wp.array2d(dtype=float),
    c: int,
    x: int,
    y: int,
    z: int,
    nx: int,
    ny: int,
    nz: int,
    value: float,
):
    if x >= 0 and y >= 0 and z >= 0 and x < nx and y < ny and z < nz:
        wp.atomic_add(q, c, (x * ny + y) * nz + z, value)


@wp.kernel
def inject(
    qe: wp.array2d(dtype=float),
    qh: wp.array2d(dtype=float),
    source: wp.array2d(dtype=wp.vec2),
    weights: wp.array(dtype=wp.vec2),
    step: wp.array(dtype=int),
    nx: int,
    ny: int,
    nz: int,
    x0: int,
    y0: int,
    z0: int,
    sy: int,
    sz: int,
    old_h_only: int,
):
    i = wp.tid()
    t = step[0]
    w = weights[t]
    if wp.abs(w[0]) + wp.abs(w[1]) > 0.0:
        x = x0 + i // (sy * sz)
        y = y0 + (i // sz) % sy
        z = z0 + i % sz
        v = vec6(0.0)
        for c in range(6):
            a = source[c, i]
            v[c] = a[0] * w[0] - a[1] * w[1]
        if old_h_only == 0:
            for dz in range(2):
                for dx in range(2):
                    add(qe, 0, x - dx, y, z + dz, nx, ny, nz, 0.25 * v[0])
                for dy in range(2):
                    add(qe, 1, x, y - dy, z + dz, nx, ny, nz, 0.25 * v[1])
            add(qe, 2, x, y, z, nx, ny, nz, v[2])
        for dy in range(2):
            add(qh, 0, x, y - dy, z, nx, ny, nz, 0.25 * v[3])
        for dx in range(2):
            add(qh, 1, x - dx, y, z, nx, ny, nz, 0.25 * v[4])
        for dx in range(2):
            for dy in range(2):
                for dz in range(2):
                    add(qh, 2, x - dx, y - dy, z + dz, nx, ny, nz, 0.0625 * v[5])


@wp.kernel
def retreat(step: wp.array(dtype=int)):
    step[0] -= 1


class Adjoint(Solver):
    def __init__(self, case):
        super().__init__(case)
        if (
            self.shape != (200, 200, 75)
            or self.meta["settings"]["spacing_nm"] != 20
            or self.meta["settings"]["parameterization"] != "smooth"
        ):
            raise ValueError(
                "Adjoint driver currently supports the 20 nm smooth bend only"
            )
        settings = self.meta["settings"]
        required = {
            "time_fs": 200,
            "pml_cells": 10,
            "domain_um": 4,
            "voxel_nm": 20,
            "scene": "bend",
        }
        if any(
            settings.get(key) != value for key, value in required.items()
        ) or settings.get("pad_domain", False):
            raise ValueError(
                "Adjoint driver requires the unpadded 200 fs reference geometry"
            )
        self.box = (100, 20, 30, 80, 80, 20)
        self.design_n = 80 * 80 * 20
        self.history = wp.empty(
            (self.steps, 3, self.design_n), dtype=float, device=self.device
        )
        self.qe = wp.zeros_like(self.E)
        self.qh = wp.zeros_like(self.H)
        self.psie = wp.zeros_like(self.psi_e)
        self.psih = wp.zeros_like(self.psi_h)
        self.d = wp.empty_like(self.psi_e)
        self.grad = wp.zeros(self.design_n, dtype=float, device=self.device)
        self.reverse_step = wp.zeros(1, dtype=int, device=self.device)
        self.adjoint_sources = []
        self.reverse_graphs = {}

    def launch_step(self):
        common = [
            self.hprev,
            self.save_h_from,
            *self.shape,
            self.source_x,
            self.courant,
            0,
        ]
        wp.launch(
            k.update_e,
            self.n,
            inputs=[
                self.E,
                self.H,
                self.eps,
                self.psi_e,
                self.coeffs["E"],
                self.sources["H"],
                self.waves["H"],
                self.step,
                *common,
            ],
            device=self.device,
        )
        wp.launch(
            record_curl,
            self.design_n,
            inputs=[
                self.H,
                self.history,
                self.step,
                *self.shape,
                *self.box,
                self.courant,
            ],
            device=self.device,
        )
        wp.launch(
            k.update_h,
            self.n,
            inputs=[
                self.H,
                self.E,
                self.eps,
                self.psi_h,
                self.coeffs["H"],
                self.sources["E"],
                self.waves["E"],
                self.step,
                *common,
            ],
            device=self.device,
        )
        for m in self.monitors:
            m.launch(self)
        wp.launch(k.advance, 1, inputs=[self.step], device=self.device)

    def prepare_sources(self):
        previous = self.adjoint_sources
        monitors = {m.name: m for m in self.monitors}
        pin = monitors["mode_in"].power(self)
        pout = monitors["mode_out"].power(self)
        if not np.isfinite(pin) or pin <= 0 or not np.isfinite(pout):
            raise ValueError("Finite modal powers and positive incident power required")
        self.adjoint_sources = []
        for name, factor in [("mode_in", pout / pin**2), ("mode_out", -1 / pin)]:
            m = monitors[name]
            ph = m.phasors()
            em = self.data[name + "/_mode_E"][0]
            hm = self.data[name + "/_mode_H"][0]
            axis = m.axis
            a = (axis + 1) % 3
            b = (axis + 2) % 3
            qe = np.zeros_like(em)
            qh = np.zeros_like(hm)
            qe[a] = hm[b].conj() / 4
            qe[b] = -hm[a].conj() / 4
            qh[b] = em[a].conj() / 4
            qh[a] = -em[b].conj() / 4
            coeff = np.concatenate((qe, qh)) * m.area_numpy[None]
            beta = np.sum(coeff * ph)
            source = (
                (2 * factor * beta.conjugate() * coeff)
                .reshape(6, -1)
                .astype(np.complex64)
            )
            host = np.stack((source.real, source.imag), axis=-1)
            if previous:
                old_monitor, v = previous[len(self.adjoint_sources)]
                if old_monitor.name != m.name:
                    raise ValueError("Changing detector layout requires new graphs")
                v.assign(host)
            else:
                v = wp.array(host, dtype=wp.vec2, device=self.device)
            self.adjoint_sources.append((m, v))

    def inject(self, old):
        for m, v in self.adjoint_sources:
            lo = [a for a, b in m.obj["grid_slice"]]
            wp.launch(
                inject,
                m.n,
                inputs=[
                    self.qe,
                    self.qh,
                    v,
                    m.weights,
                    self.reverse_step,
                    *self.shape,
                    *lo,
                    *m.shape[1:],
                    old,
                ],
                device=self.device,
            )

    def reverse_one(self):
        self.inject(0)
        common = [*self.shape, self.courant]
        wp.launch(
            prepare,
            self.n,
            inputs=[self.qh, self.eps, self.psih, self.coeffs["H"], self.d, *common, 0],
            device=self.device,
        )
        args = [
            self.d,
            self.history,
            self.grad,
            self.reverse_step,
            *self.shape,
            *self.box,
        ]
        wp.launch(gather, self.n, inputs=[self.qe, *args, 1], device=self.device)
        wp.launch(
            prepare,
            self.n,
            inputs=[self.qe, self.eps, self.psie, self.coeffs["E"], self.d, *common, 1],
            device=self.device,
        )
        wp.launch(gather, self.n, inputs=[self.qh, *args, 0], device=self.device)
        self.inject(1)
        wp.launch(retreat, 1, inputs=[self.reverse_step], device=self.device)

    def reset_reverse(self):
        for a in (self.qe, self.qh, self.psie, self.psih, self.grad):
            a.zero_()
        self.reverse_step.assign(np.array([self.steps - 1], np.int32))
        wp.synchronize_device(self.device)

    def compile_reverse(self):
        self.reset_reverse()
        self.reverse_one()
        wp.synchronize_device(self.device)
        self.reset_reverse()
        for n in {self.graph_steps, self.steps % self.graph_steps} - {0}:
            with wp.ScopedCapture(device=self.device) as c:
                for _ in range(n):
                    self.reverse_one()
            self.reverse_graphs[n] = c.graph
        self.reset_reverse()

    def backward(self):
        for _ in range(self.steps // self.graph_steps):
            wp.capture_launch(self.reverse_graphs[self.graph_steps])
        if self.steps % self.graph_steps:
            wp.capture_launch(self.reverse_graphs[self.steps % self.graph_steps])
        wp.synchronize_device(self.device)

    def set_design(self, parameters):
        from .material import Material

        material = Material(np.asarray(parameters).reshape(80, 80), self.meta["beta"])
        wp.launch(
            write_design,
            self.design_n,
            inputs=[material.inv_eps, self.eps, *self.shape, *self.box],
            device=self.device,
        )
        return material

    def differentiate(self, parameters):
        """Evaluate -transmission and its parameter gradient for a new design."""
        from .detectors import measurements

        material = self.set_design(parameters)
        if not self.graphs:
            self.compile()
        self.reset()
        self.run()
        self.prepare_sources()
        if not self.reverse_graphs:
            self.compile_reverse()
        self.reset_reverse()
        self.backward()
        voxel = self.grad.numpy().reshape(80, 80, 20).sum(axis=2)
        gradient = material.pullback(voxel)
        return -measurements(self)["modal_transmission"], gradient


@wp.kernel
def write_design(
    plane: wp.array2d(dtype=float),
    eps: wp.array(dtype=float),
    nx: int,
    ny: int,
    nz: int,
    x0: int,
    y0: int,
    z0: int,
    sx: int,
    sy: int,
    sz: int,
):
    i = wp.tid()
    x = i // (sy * sz)
    y = (i // sz) % sy
    z = i % sz
    eps[((x0 + x) * ny + y0 + y) * nz + z0 + z] = plane[x, y]
