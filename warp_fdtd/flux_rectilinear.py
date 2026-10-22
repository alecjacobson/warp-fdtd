"""Experimental float32 flux control for periodic rectilinear Maxwell updates.

D and B face fluxes use two-float accumulators. Each signed edge impulse is
computed once and shared by its adjacent faces, preserving the topological
curl/divergence cancellation before rounding. Four kernels per timestep.
No source, CPML, block interface, local stepping, or adjoint support.
"""

import numpy as np
import warp as wp

from .rectilinear import RectilinearPeriodic

wp.set_module_options({"enable_backward": False, "fast_math": False})


@wp.func
def accumulate(high: float, low: float, term: float):
    # TwoSum, twice. Do not reassociate these expressions or enable fast math.
    summed = high + term
    part = summed - high
    error = (high - (summed - part)) + (term - part)
    residual = low + error
    updated = summed + residual
    part = updated - summed
    correction = (summed - (updated - part)) + (residual - part)
    return wp.vec2(updated, correction)


@wp.kernel
def edge_impulses(
    flux: wp.array2d(dtype=float),
    coefficient: wp.array2d(dtype=float),
    impulse: wp.array2d(dtype=float),
):
    c, i = wp.tid()
    impulse[c, i] = coefficient[c, i] * flux[c, i]


@wp.func
def neighbor(i: int, axis: int, shift: int, nx: int, ny: int, nz: int):
    x, y, z = i // (ny * nz), (i // nz) % ny, i % nz
    if axis == 0:
        x = (x + shift + nx) % nx
    elif axis == 1:
        y = (y + shift + ny) % ny
    else:
        z = (z + shift + nz) % nz
    return (x * ny + y) * nz + z


def make_face_update(forward):
    @wp.kernel
    def update(
        flux: wp.array2d(dtype=float),
        correction: wp.array2d(dtype=float),
        impulse: wp.array2d(dtype=float),
        nx: int,
        ny: int,
        nz: int,
    ):
        c, i = wp.tid()
        b, a = (c + 1) % 3, (c + 2) % 3
        terms = wp.vec4()
        if wp.static(forward):
            terms = wp.vec4(
                -impulse[a, neighbor(i, b, 1, nx, ny, nz)],
                impulse[a, i],
                impulse[b, neighbor(i, a, 1, nx, ny, nz)],
                -impulse[b, i],
            )
        else:
            terms = wp.vec4(
                impulse[a, i],
                -impulse[a, neighbor(i, b, -1, nx, ny, nz)],
                -impulse[b, i],
                impulse[b, neighbor(i, a, -1, nx, ny, nz)],
            )
        state = wp.vec2(flux[c, i], correction[c, i])
        for k in range(4):
            state = accumulate(state[0], state[1], terms[k])
        flux[c, i] = state[0]
        correction[c, i] = state[1]

    return update


update_d = make_face_update(False)
update_b = make_face_update(True)


class FluxRectilinearPeriodic(RectilinearPeriodic):
    """Same physical field API as RectilinearPeriodic, with integrated flux storage.

    E/H attributes internally store high parts of D/B face fluxes. The diagnostic
    fields() method decodes physical E/H. Device storage is 21 float32 values per
    cell plus axis widths: 6 high, 6 low, 6 metric coefficients, 3 edge impulses.
    """

    def __init__(self, edges, permittivity=1.0, cfl=0.5, device="cuda:0"):
        super().__init__(edges, permittivity=permittivity, cfl=cfl, device=device)
        self.correction_e = wp.zeros_like(self.E)
        self.correction_h = wp.zeros_like(self.H)
        self.impulse = wp.zeros_like(self.E)
        self.face_scales = []
        self.coefficients = []
        for electric in (True, False):
            areas, coefficients = [], []
            for c in range(3):
                widths = self.dual if electric else self.primal
                area = np.ones(self.shape)
                for axis in range(3):
                    if axis != c:
                        sh = [1, 1, 1]
                        sh[axis] = self.shape[axis]
                        area *= widths[axis].reshape(sh)
                if electric:
                    area *= self.epsilon[c]
                sh = [1, 1, 1]
                sh[c] = self.shape[c]
                length = (self.primal if electric else self.dual)[c].reshape(sh)
                areas.append(area)
                coefficients.append(self.dt * length / area)
            self.face_scales.append(np.asarray(areas))
            self.coefficients.append(
                wp.array(
                    np.asarray(coefficients, dtype="f").reshape(3, -1),
                    device=self.device,
                )
            )
        # Material is incorporated into the electric impulse coefficients.
        self.eps = None

    def set_fields(self, e, h):
        for field, scale, high, low in zip(
            (e, h),
            self.face_scales,
            (self.E, self.H),
            (self.correction_e, self.correction_h),
        ):
            a = np.asarray(field, dtype=float)
            if a.shape != (3, *self.shape) or not np.isfinite(a).all():
                raise ValueError("Finite three-component fields matching grid required")
            flux = a * scale
            rounded = flux.astype("f")
            high.assign(rounded.reshape(3, -1))
            low.assign((flux - rounded).astype("f").reshape(3, -1))

    def fields(self):
        # Low words protect accumulation; high words drive the explicit update.
        # Decode in float64, then expose float32 as in the ordinary control.
        return tuple(
            (a.numpy().reshape(3, *self.shape).astype(float) / scale).astype("f")
            for a, scale in zip((self.E, self.H), self.face_scales)
        )

    def step(self):
        for source, coefficient, destination, correction, kernel in (
            (self.H, self.coefficients[1], self.E, self.correction_e, update_d),
            (self.E, self.coefficients[0], self.H, self.correction_h, update_b),
        ):
            wp.launch(
                edge_impulses,
                (3, self.n),
                inputs=[source, coefficient, self.impulse],
                device=self.device,
            )
            wp.launch(
                kernel,
                (3, self.n),
                inputs=[destination, correction, self.impulse, *self.shape],
                device=self.device,
            )

    def compile(self, lengths=(256,)):
        # Preserve the exact flux expansion across warmup and capture.
        arrays = (self.E, self.H, self.correction_e, self.correction_h)
        initial = [a.numpy().copy() for a in arrays]
        self.step()
        wp.synchronize_device(self.device)
        for a, value in zip(arrays, initial):
            a.assign(value)
        if self.device.is_cuda:
            for length in set(lengths):
                if length > 0:
                    with wp.ScopedCapture(device=self.device) as capture:
                        for _ in range(length):
                            self.step()
                    self.graphs[length] = capture.graph
        for a, value in zip(arrays, initial):
            a.assign(value)
        wp.synchronize_device(self.device)

    def run(self, steps):
        if not isinstance(steps, int) or steps < 1:
            raise ValueError("Positive integer step count required")
        if self.device.is_cuda:
            super().run(steps)
        else:
            for _ in range(steps):
                self.step()
            wp.synchronize_device(self.device)
