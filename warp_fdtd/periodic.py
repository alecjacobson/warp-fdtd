"""Periodic vacuum propagators: Yee leapfrog and fourth-order space/RK4 time.

The fourth-order path is deliberately restricted to smooth homogeneous periodic
media. It is not used across the bend's dielectric interfaces or its CPML.
"""

import time
import numpy as np
import warp as wp
from .kernels import update_e, update_h

wp.set_module_options({"enable_backward": False, "fast_math": False})


@wp.func
def sample6(
    a: wp.array2d(dtype=float),
    c: int,
    x: int,
    y: int,
    z: int,
    nx: int,
    ny: int,
    nz: int,
):
    i = (((x + 2 * nx) % nx) * ny + (y + 2 * ny) % ny) * nz + (z + 2 * nz) % nz
    return a[c, i]


@wp.func
def derivative4(
    a: wp.array2d(dtype=float),
    c: int,
    x: int,
    y: int,
    z: int,
    axis: int,
    forward: int,
    nx: int,
    ny: int,
    nz: int,
):
    dx = int(axis == 0)
    dy = int(axis == 1)
    dz = int(axis == 2)
    # Forward H curl: +1/0 and +2/-1. Backward E curl: 0/-1 and +1/-2.
    b = forward
    d1 = sample6(a, c, x + b * dx, y + b * dy, z + b * dz, nx, ny, nz) - sample6(
        a, c, x + (b - 1) * dx, y + (b - 1) * dy, z + (b - 1) * dz, nx, ny, nz
    )
    d3 = sample6(
        a, c, x + (b + 1) * dx, y + (b + 1) * dy, z + (b + 1) * dz, nx, ny, nz
    ) - sample6(a, c, x + (b - 2) * dx, y + (b - 2) * dy, z + (b - 2) * dz, nx, ny, nz)
    return 1.125 * d1 - d3 / 24.0


@wp.kernel
def rk_stage(
    base: wp.array2d(dtype=float),
    current: wp.array2d(dtype=float),
    intermediate: wp.array2d(dtype=float),
    acc: wp.array2d(dtype=float),
    nx: int,
    ny: int,
    nz: int,
    courant: float,
    stage: int,
):
    c, i = wp.tid()
    x = i // (ny * nz)
    y = (i // nz) % ny
    z = i % nz
    component = c % 3
    ca = (component + 1) % 3
    cb = (component + 2) % 3
    forward = int(c >= 3)
    offset = 3 * (1 - forward)
    rhs = derivative4(
        current, offset + cb, x, y, z, ca, forward, nx, ny, nz
    ) - derivative4(current, offset + ca, x, y, z, cb, forward, nx, ny, nz)
    rhs *= courant * float(1 - 2 * forward)
    if stage == 1:
        acc[c, i] = base[c, i] + rhs / 6.0
        intermediate[c, i] = base[c, i] + 0.5 * rhs
    elif stage == 2:
        acc[c, i] += rhs / 3.0
        intermediate[c, i] = base[c, i] + 0.5 * rhs
    elif stage == 3:
        acc[c, i] += rhs / 3.0
        intermediate[c, i] = base[c, i] + rhs
    else:
        base[c, i] = acc[c, i] + rhs / 6.0


@wp.kernel
def record_mode(
    e: wp.array2d(dtype=float), signal: wp.array(dtype=float), step: wp.array(dtype=int)
):
    signal[step[0]] = e[0, 0]
    step[0] += 1


class Periodic:
    def __init__(self, n, cfl, method, shape=None):
        if method not in ("yee", "rk4") or n < 4 or not 0 < cfl <= 0.99:
            raise ValueError("Require method yee/rk4, >=4 cells and 0<CFL<=0.99")
        self.shape = shape or (4, 4, n)
        self.ncells = int(np.prod(self.shape))
        self.n = n
        self.cfl = cfl
        self.method = method
        self.courant = cfl / np.sqrt(3)
        self.dt = 1550e-9 / n * self.courant / 299792458.0
        self.steps = int(np.ceil(20 * (1550e-9 / 299792458.0) / self.dt))
        self.device = wp.get_device("cuda:0")
        initial = np.zeros((6, *self.shape), np.float32)
        initial[0] = np.cos(2 * np.pi * np.arange(self.shape[2], dtype=np.float32) / n)
        self.initial = initial.reshape(6, -1)
        self.state = wp.array(self.initial, dtype=float, device=self.device)
        self.E = self.state[:3]
        self.H = self.state[3:]
        self.step = wp.zeros(1, dtype=int, device=self.device)
        self.signal = wp.zeros(self.steps, dtype=float, device=self.device)
        self.aux = (
            [wp.zeros_like(self.state) for _ in range(3)] if method == "rk4" else []
        )
        if method == "yee":
            self.eps = wp.ones(self.ncells, dtype=float, device=self.device)
            self.psi = wp.zeros((1, 1), dtype=float, device=self.device)
            coeff = np.zeros((3, max(self.shape), 3), np.float32)
            coeff[:, :, 1:] = 1
            self.coeff = wp.array(coeff, dtype=wp.vec3, device=self.device)
            self.dummy = wp.zeros((3, 1), dtype=float, device=self.device)
        self.graphs = {}

    def launch_step(self):
        if self.method == "yee":
            args = [
                self.eps,
                self.psi,
                self.coeff,
                self.dummy,
                self.dummy,
                self.step,
                self.dummy,
                self.steps,
                *self.shape,
                -1,
                self.courant,
                1,
            ]
            wp.launch(
                update_e,
                self.ncells,
                inputs=[self.E, self.H, *args],
                device=self.device,
            )
            wp.launch(
                update_h,
                self.ncells,
                inputs=[self.H, self.E, *args],
                device=self.device,
            )
        else:
            a, b, acc = self.aux
            for stage, cur, out in (
                (1, self.state, a),
                (2, a, b),
                (3, b, a),
                (4, a, b),
            ):
                wp.launch(
                    rk_stage,
                    (6, self.ncells),
                    inputs=[
                        self.state,
                        cur,
                        out,
                        acc,
                        *self.shape,
                        self.courant,
                        stage,
                    ],
                    device=self.device,
                )
        wp.launch(
            record_mode, 1, inputs=[self.E, self.signal, self.step], device=self.device
        )

    def reset(self):
        self.state.assign(self.initial)
        self.step.zero_()
        self.signal.zero_()
        wp.synchronize_device(self.device)

    def compile(self):
        start = time.perf_counter()
        self.launch_step()
        wp.synchronize_device(self.device)
        self.reset()
        for n in {64, self.steps % 64} - {0}:
            with wp.ScopedCapture(device=self.device) as c:
                for _ in range(n):
                    self.launch_step()
            self.graphs[n] = c.graph
        self.reset()
        return time.perf_counter() - start

    def run(self):
        for _ in range(self.steps // 64):
            wp.capture_launch(self.graphs[64])
        if self.steps % 64:
            wp.capture_launch(self.graphs[self.steps % 64])
        wp.synchronize_device(self.device)
