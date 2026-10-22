"""Experimental explicit 3D Yee solver on periodic rectilinear grids.

A verification/control backend: no CPML, source, port or flagship support yet.
Normalized c=mu=1; E lies on primal edges and H on dual edges. Backward curl
uses dual widths; forward curl uses primal widths. This is static tensor-product
spacing, not local block refinement or an octree.
"""

import numpy as np
import warp as wp

wp.set_module_options({"enable_backward": False, "fast_math": False})


@wp.func
def get(f: wp.array2d(dtype=float), x: int, y: int, z: int, nx: int, ny: int, nz: int):
    i = (((x + nx) % nx) * ny + (y + ny) % ny) * nz + (z + nz) % nz
    return wp.vec3(f[0, i], f[1, i], f[2, i])


@wp.func
def get_double(
    f: wp.array2d(dtype=wp.float64), x: int, y: int, z: int, nx: int, ny: int, nz: int
):
    i = (((x + nx) % nx) * ny + (y + ny) % ny) * nz + (z + nz) % nz
    return wp.vec3d(f[0, i], f[1, i], f[2, i])


def make_step(electric, double=False, compensated=False):
    dtype = wp.float64 if double else wp.float32
    vector = wp.vec3d if double else wp.vec3
    sample = get_double if double else get

    @wp.kernel
    def step(
        f: wp.array2d(dtype=dtype),
        g: wp.array2d(dtype=dtype),
        correction: wp.array2d(dtype=dtype),
        eps: wp.array2d(dtype=dtype),
        widths: wp.array2d(dtype=dtype),
        nx: int,
        ny: int,
        nz: int,
        dt: dtype,
    ):
        i = wp.tid()
        x = i // (ny * nz)
        y = (i // nz) % ny
        z = i % nz
        v = sample(g, x, y, z, nx, ny, nz)
        if wp.static(electric):
            dx = (v - sample(g, x - 1, y, z, nx, ny, nz)) / widths[0, x]
            dy = (v - sample(g, x, y - 1, z, nx, ny, nz)) / widths[1, y]
            dz = (v - sample(g, x, y, z - 1, nx, ny, nz)) / widths[2, z]
        else:
            dx = (sample(g, x + 1, y, z, nx, ny, nz) - v) / widths[0, x]
            dy = (sample(g, x, y + 1, z, nx, ny, nz) - v) / widths[1, y]
            dz = (sample(g, x, y, z + 1, nx, ny, nz) - v) / widths[2, z]
        curl = vector(dy[2] - dz[1], dz[0] - dx[2], dx[1] - dy[0])
        for c in range(3):
            if wp.static(compensated):
                increment = dtype(0.0)
                if wp.static(electric):
                    increment = dt * curl[c] / eps[c, i]
                else:
                    increment = -dt * curl[c]
                adjusted = increment - correction[c, i]
                old = f[c, i]
                updated = old + adjusted
                correction[c, i] = (updated - old) - adjusted
                f[c, i] = updated
            else:
                if wp.static(electric):
                    f[c, i] += dt * curl[c] / eps[c, i]
                else:
                    f[c, i] -= dt * curl[c]

    return step


step_e = make_step(True)
step_h = make_step(False)
step_e_double = make_step(True, double=True)
step_h_double = make_step(False, double=True)
step_e_compensated = make_step(True, compensated=True)
step_h_compensated = make_step(False, compensated=True)


def curl_numpy(field, widths, forward):
    field = np.asarray(field, dtype=float)
    derivatives = []
    for axis, w in enumerate(widths):
        shape = [1, 1, 1, 1]
        shape[axis + 1] = len(w)
        derivative = (
            np.roll(field, -1, axis=axis + 1) - field
            if forward
            else field - np.roll(field, 1, axis=axis + 1)
        )
        derivatives.append(derivative / np.asarray(w).reshape(shape))
    dx, dy, dz = derivatives
    return np.stack((dy[2] - dz[1], dz[0] - dx[2], dx[1] - dy[0]))


def divergence_numpy(field, widths, forward):
    result = np.zeros(field.shape[1:], float)
    for axis, w in enumerate(widths):
        shape = [1, 1, 1]
        shape[axis] = len(w)
        d = (
            np.roll(field[axis], -1, axis=axis) - field[axis]
            if forward
            else field[axis] - np.roll(field[axis], 1, axis=axis)
        )
        result += d / np.asarray(w).reshape(shape)
    return result


class RectilinearPeriodic:
    def __init__(
        self,
        edges,
        permittivity=1.0,
        cfl=0.5,
        device="cuda:0",
        precision="float32",
        compensated=False,
    ):
        if precision not in ("float32", "float64"):
            raise ValueError("precision must be float32 or float64")
        if compensated and precision != "float32":
            raise ValueError("Compensation control currently requires float32")
        self.compensated = compensated
        self.precision = precision
        self.dtype = wp.float64 if precision == "float64" else wp.float32
        self.update_e = step_e_double if precision == "float64" else step_e
        self.update_h = step_h_double if precision == "float64" else step_h
        if compensated:
            self.update_e, self.update_h = step_e_compensated, step_h_compensated
        self.edges = tuple(np.asarray(x, dtype=float) for x in edges)
        if len(self.edges) != 3 or any(
            x.ndim != 1
            or len(x) < 4
            or not np.isfinite(x).all()
            or np.any(np.diff(x) <= 0)
            for x in self.edges
        ):
            raise ValueError(
                "Three increasing finite edge arrays, at least three cells per axis, required"
            )
        if not np.isfinite(cfl) or not 0 < cfl <= 0.99:
            raise ValueError("0<CFL<=0.99 required")
        self.shape = tuple(len(x) - 1 for x in self.edges)
        self.n = int(np.prod(self.shape))
        self.primal = tuple(np.diff(x) for x in self.edges)
        self.dual = tuple((w + np.roll(w, 1)) * 0.5 for w in self.primal)
        self.epsilon = np.broadcast_to(
            np.asarray(permittivity, float), (3, *self.shape)
        ).copy()
        if not np.isfinite(self.epsilon).all() or np.any(self.epsilon <= 0):
            raise ValueError("Finite positive componentwise permittivity required")
        # Conservative bound using each axis's smallest primal/dual width.
        minimum = [min(w.min(), d.min()) for w, d in zip(self.primal, self.dual)]
        self.dt = float(
            cfl * np.sqrt(self.epsilon.min()) / np.sqrt(sum(1 / v**2 for v in minimum))
        )
        self.device = wp.get_device(device)
        self.E = wp.zeros((3, self.n), dtype=self.dtype, device=self.device)
        self.H = wp.zeros_like(self.E)
        correction_shape = (3, self.n) if compensated else (1, 1)
        self.correction_e = wp.zeros(
            correction_shape, dtype=self.dtype, device=self.device
        )
        self.correction_h = wp.zeros_like(self.correction_e)
        self.eps = wp.array(
            self.epsilon.astype(self.precision).reshape(3, -1), device=self.device
        )
        self.width_arrays = []
        for widths in (self.dual, self.primal):
            a = np.ones((3, max(self.shape)), self.precision)
            for axis, w in enumerate(widths):
                a[axis, : len(w)] = w
            self.width_arrays.append(wp.array(a, device=self.device))
        self.mass_e = self.mass_h = None
        weights = []
        for electric in (True, False):
            vector = []
            for c in range(3):
                volume = np.ones(self.shape)
                for axis in range(3):
                    w = (
                        self.primal[axis]
                        if ((axis == c) == electric)
                        else self.dual[axis]
                    )
                    sh = [1, 1, 1]
                    sh[axis] = len(w)
                    volume *= w.reshape(sh)
                vector.append(volume)
            weights.append(np.array(vector))
        self.mass_e = weights[0] * self.epsilon
        self.mass_h = weights[1]
        self.graphs = {}

    def set_fields(self, e, h):
        self.correction_e.zero_()
        self.correction_h.zero_()
        for name, a in [("E", e), ("H", h)]:
            a = np.asarray(a)
            if a.shape != (3, *self.shape) or not np.isfinite(a).all():
                raise ValueError("Finite three-component fields matching grid required")
            getattr(self, name).assign(a.astype(self.precision).reshape(3, -1))

    def fields(self):
        return tuple(a.numpy().reshape(3, *self.shape) for a in (self.E, self.H))

    def step(self):
        wp.launch(
            self.update_e,
            self.n,
            inputs=[
                self.E,
                self.H,
                self.correction_e,
                self.eps,
                self.width_arrays[0],
                *self.shape,
                self.dt,
            ],
            device=self.device,
        )
        wp.launch(
            self.update_h,
            self.n,
            inputs=[
                self.H,
                self.E,
                self.correction_h,
                self.eps,
                self.width_arrays[1],
                *self.shape,
                self.dt,
            ],
            device=self.device,
        )

    def compile(self, lengths=(256,)):
        # Preserve caller's initial fields across JIT warmup and graph capture.
        initial = self.fields()
        corrections = (self.correction_e.numpy(), self.correction_h.numpy())
        self.step()
        wp.synchronize_device(self.device)
        self.set_fields(*initial)
        self.correction_e.assign(corrections[0])
        self.correction_h.assign(corrections[1])
        for length in set(lengths):
            if length < 1:
                continue
            with wp.ScopedCapture(device=self.device) as capture:
                for _ in range(length):
                    self.step()
            self.graphs[length] = capture.graph
        self.set_fields(*initial)
        self.correction_e.assign(corrections[0])
        self.correction_h.assign(corrections[1])
        wp.synchronize_device(self.device)

    def run(self, steps):
        if not isinstance(steps, int) or steps < 1:
            raise ValueError("Positive integer step count required")
        for length in {min(steps, 256), steps % 256} - {0}:
            if length not in self.graphs:
                self.compile((length,))
        for _ in range(steps // 256):
            wp.capture_launch(self.graphs[256])
        if steps % 256:
            wp.capture_launch(self.graphs[steps % 256])
        wp.synchronize_device(self.device)

    def energy(self, e, h):
        # Invariant for E^n,H^(n+1/2) with E-first symplectic Euler/leapfrog.
        cross = e * curl_numpy(h, self.dual, False) * (self.mass_e / self.epsilon)
        return float(
            0.5
            * (
                np.sum(self.mass_e * e.astype(float) ** 2)
                + np.sum(self.mass_h * h.astype(float) ** 2)
                + self.dt * np.sum(cross)
            )
        )

    def coordinates(self, component):
        centers = [(x[:-1] + x[1:]) * 0.5 for x in self.edges]
        nodes = [
            centers[a]
            if ((a == component % 3) == (component < 3))
            else self.edges[a][:-1]
            for a in range(3)
        ]
        return np.meshgrid(*nodes, indexing="ij")
