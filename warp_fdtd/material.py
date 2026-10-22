"""Differentiable Gaussian filtering and subpixel material projection in Warp.

Implements the mathematical material map used by the pinned baseline (19x19
Gaussian, sigma=3 cells, edge extension; circular subpixel projection). Reverse
mode is restricted to this small material map, not an unrolled FDTD trajectory.
"""

import numpy as np
import warp as wp


@wp.kernel
def smooth(
    p: wp.array2d(dtype=float),
    weights: wp.array2d(dtype=float),
    out: wp.array2d(dtype=float),
):
    x, y = wp.tid()
    total = float(0.0)
    for dx in range(-9, 10):
        for dy in range(-9, 10):
            ix = wp.clamp(x + dx, 0, p.shape[0] - 1)
            iy = wp.clamp(y + dy, 0, p.shape[1] - 1)
            total += p[ix, iy] * weights[dx + 9, dy + 9]
    out[x, y] = total


@wp.func
def project(r: float, beta: float):
    return (wp.tanh(0.5 * beta) + wp.tanh(beta * (r - 0.5))) / (
        2.0 * wp.tanh(0.5 * beta)
    )


@wp.kernel
def subpixel(rho: wp.array2d(dtype=float), out: wp.array2d(dtype=float), beta: float):
    x, y = wp.tid()
    xm = wp.max(x - 1, 0)
    xp = wp.min(x + 1, rho.shape[0] - 1)
    ym = wp.max(y - 1, 0)
    yp = wp.min(y + 1, rho.shape[1] - 1)
    gx = (rho[xp, y] - rho[xm, y]) / float(xp - xm)
    gy = (rho[x, yp] - rho[x, ym]) / float(yp - ym)
    norm2 = gx * gx + gy * gy
    r = rho[x, y]
    value = project(r, beta)
    if norm2 > 0.0:
        norm = wp.sqrt(norm2)
        d = (0.5 - r) / (0.55 * norm)
        if wp.abs(d) < 1.0:
            fill = 0.5 - 0.9375 * d + 0.625 * d * d * d - 0.1875 * d * d * d * d * d
            lo = project(r - 0.55 * norm * fill, beta)
            hi = project(r + 0.55 * norm * (1.0 - fill), beta)
            value = (1.0 - fill) * lo + fill * hi
    out[x, y] = 1.0 / (1.0 + 11.25 * value)


class Material:
    def __init__(self, params, beta):
        if (
            params.ndim != 2
            or min(params.shape) < 2
            or not np.isfinite(params).all()
            or not np.isfinite(beta)
            or beta <= 0
        ):
            raise ValueError("Require finite 2D parameters and positive finite beta")
        self.p = wp.array(
            params.astype(np.float32), dtype=float, device="cuda:0", requires_grad=True
        )
        self.filtered = wp.zeros_like(self.p, requires_grad=True)
        self.inv_eps = wp.zeros_like(self.p, requires_grad=True)
        xy = np.arange(-9, 10, dtype=np.float32)
        weights = np.exp(-(xy[:, None] ** 2 + xy[None, :] ** 2) / 18).astype(np.float32)
        weights /= weights.sum()
        self.weights = wp.array(weights, dtype=float, device="cuda:0")
        self.beta = float(beta)
        self.tape = wp.Tape()
        with self.tape:
            wp.launch(
                smooth,
                self.p.shape,
                inputs=[self.p, self.weights, self.filtered],
                device="cuda:0",
            )
            wp.launch(
                subpixel,
                self.p.shape,
                inputs=[self.filtered, self.inv_eps, self.beta],
                device="cuda:0",
            )

    def pullback(self, gradient):
        self.tape.zero()
        self.tape.backward(
            grads={
                self.inv_eps: wp.array(
                    np.asarray(gradient, np.float32), dtype=float, device="cuda:0"
                )
            }
        )
        return self.p.grad.numpy()
