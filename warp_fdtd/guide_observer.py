"""Common-physical-point Yee sampling with explicit H time centering."""

from itertools import product
import numpy as np
import warp as wp
from .driven_uniform import DrivenUniform


@wp.kernel
def gather(
    indices: wp.array2d(dtype=int),
    weights: wp.array2d(dtype=float),
    field: wp.array(dtype=float),
    old: wp.array(dtype=float),
    center: int,
    out: wp.array(dtype=float),
):
    row = wp.tid()
    value = float(0.0)
    for j in range(8):
        v = field[indices[row, j]]
        if center == 1:
            v = 0.5 * (v + old[indices[row, j]])
        value += weights[row, j] * v
    out[row] = value


def points(problem):
    axes = (
        np.arange(0.45, problem.length - 0.44, 0.1),
        np.arange(problem.width / 2 - 0.775, problem.width / 2 + 0.776, 0.1),
        np.arange(problem.height / 2 - 0.775, problem.height / 2 + 0.776, 0.1),
    )
    return axes, np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)


def stencil(xyz, origin, h, shape, component, electric, offset):
    staggering = np.zeros(3) if electric else np.full(3, 0.5)
    staggering[component] = 0.5 if electric else 0
    index = (xyz - origin) / h - staggering
    base = np.clip(np.floor(index).astype(int), 0, np.array(shape) - 2)
    fraction = index - base
    indices = []
    weights = []
    for corner in product((0, 1), repeat=3):
        p = base + corner
        indices.append(offset + np.ravel_multi_index(tuple(p), shape))
        weights.append(np.prod(np.where(corner, fraction, 1 - fraction)))
    return indices, weights


class GuideObserver:
    def __init__(self, solver, problem):
        self.solver = solver
        self.axes, self.xyz = points(problem)
        n = len(self.xyz)
        indices = [np.empty((3 * n, 8), np.int32) for _ in range(2)]
        weights = [np.empty((3 * n, 8), np.float32) for _ in range(2)]
        if isinstance(solver, DrivenUniform):
            for electric, shapes in ((True, solver.eshapes), (False, solver.hshapes)):
                offsets = np.cumsum([0] + [int(np.prod(s)) for s in shapes])
                for c in range(3):
                    for i, xyz in enumerate(self.xyz):
                        ix, w = stencil(
                            xyz,
                            np.zeros(3),
                            solver.hspacing,
                            shapes[c],
                            c,
                            electric,
                            int(offsets[c]),
                        )
                        indices[not electric][c * n + i] = ix
                        weights[not electric][c * n + i] = w
        else:
            op = solver.operator
            mesh = op.mesh
            blocks = sorted(mesh.leaves)
            ids = {b: i for i, b in enumerate(blocks)}
            origins = np.array([mesh.bounds(b)[0] for b in blocks])
            eoffs = np.cumsum([0] + [len(g.me) for g in op.regions])
            hoffs = np.cumsum([0] + [len(g.mh) for g in op.regions])
            unit = mesh.root_width / mesh.scale
            for i, xyz in enumerate(self.xyz):
                block = mesh.locate((xyz - np.array(mesh.origin)) / unit)
                b = ids[block]
                g = op.regions[b]
                for electric, fields, off in (
                    (True, g.eids, eoffs[b]),
                    (False, g.hids, hoffs[b]),
                ):
                    for c in range(3):
                        offset = int(off + fields[c].ravel()[0])
                        shape = fields[c].shape
                        ix, w = stencil(
                            xyz, origins[b], g.h, shape, c, electric, offset
                        )
                        indices[not electric][c * n + i] = ix
                        weights[not electric][c * n + i] = w
        self.indices = [wp.array(v, device=solver.device) for v in indices]
        self.weights = [wp.array(v, device=solver.device) for v in weights]
        self.out = [wp.zeros(3 * n, device=solver.device) for _ in range(2)]

    def sample(self):
        s = self.solver
        old = wp.clone(s.h)
        if isinstance(s, DrivenUniform):
            ef = s.e
            s.magnetic_step()
        else:
            expand, magnetic, _, _ = s.kernels
            wp.launch(
                expand, len(s.local_e), inputs=[*s.p, s.e, s.local_e], device=s.device
            )
            ef = s.local_e
            wp.launch(
                magnetic,
                len(s.h),
                inputs=[s.local_e, s.h, s.widths, s.sigma_h, s.n, s.dt],
                device=s.device,
            )
        for c, field in enumerate((ef, s.h)):
            wp.launch(
                gather,
                len(self.out[c]),
                inputs=[self.indices[c], self.weights[c], field, old, c, self.out[c]],
                device=s.device,
            )
        wp.copy(s.h, old)
        result = np.concatenate([v.numpy().reshape(3, -1) for v in self.out])
        return result
