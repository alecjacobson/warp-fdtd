"""Experimental 3D 3:1 planar coupling, periodic x/z and PEC outer y faces.

Constraint assembly of the lossless tangential trace rule in Bekmambetova,
Zhang & Triverio, https://arxiv.org/abs/1705.02274, equations (38)-(54).
The optional area trace is an experimental Galerkin constraint: coarse E is
the average of unconstrained fine E over its trace tile. It is NOT the cited
published rule and has no established composite Gauss or scattering guarantee.
The circulation variant averages only along each coincident coarse edge,
so restriction respects the discrete Stokes circulation. It remains experimental.
No octree edge/corner closure, CPML, source or waveguide integration yet.
SciPy constructs a reference operator; all time stepping is in pure Warp.
The sparse kernels are a correctness prototype, not the intended dense-block
GPU implementation. Units c=mu=1. Local interface mass blocks are pre-inverted;
no global solve is performed per timestep.
"""

from dataclasses import dataclass
from itertools import product

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
import warp as wp

wp.set_module_options({"enable_backward": False, "fast_math": False})


@dataclass
class YeeRegion:
    shape: tuple
    h: float
    y0: float
    eids: list
    hids: list
    ecoords: np.ndarray
    hcoords: np.ndarray
    me: np.ndarray
    mh: np.ndarray
    coupling: sp.csr_matrix


def region(shape, h, y0, epsilon=1.0):
    nx, ny, nz = shape
    es = [(nx, ny + 1, nz), (nx, ny, nz), (nx, ny + 1, nz)]
    hs = [(nx, ny, nz), (nx, ny + 1, nz), (nx, ny, nz)]

    def ids(shapes):
        result, count = [], 0
        for s in shapes:
            n = int(np.prod(s))
            result.append(np.arange(count, count + n).reshape(s))
            count += n
        return result

    ei, hi = ids(es), ids(hs)

    def coordinates(shapes, electric):
        result = []
        for c, s in enumerate(shapes):
            xyz = np.indices(s).reshape(3, -1).T.astype(float)
            offset = np.zeros(3) if electric else np.full(3, 0.5)
            offset[c] = 0.5 if electric else 0.0
            result.append((xyz + offset) * h + [0.0, y0, 0.0])
        return np.concatenate(result)

    ec, hc = coordinates(es, True), coordinates(hs, False)
    ev = np.full(len(ec), h**3)
    for c in (0, 2):
        ev[ei[c][:, 0, :].ravel()] *= 0.5
        ev[ei[c][:, -1, :].ravel()] *= 0.5
    hv = np.full(len(hc), h**3)
    for j in (0, ny):
        hv[hi[1][:, j, :].ravel()] *= 0.5
    eps = np.broadcast_to(epsilon(ec) if callable(epsilon) else epsilon, ev.shape)
    if not np.isfinite(eps).all() or np.any(eps <= 0):
        raise ValueError("Finite positive permittivity required")
    rows, cols, values = [], [], []
    # K = V_H curl. E equations use its exact transpose before constraints.
    for c, s in enumerate(hs):
        a, b = (c + 1) % 3, (c + 2) % 3
        for index in product(*(range(n) for n in s)):
            row = int(hi[c][index])
            for derivative, component, sign in ((a, b, 1.0), (b, a, -1.0)):
                for shift, direction in ((1, 1.0), (0, -1.0)):
                    target = list(index)
                    target[derivative] += shift
                    target[0] %= nx
                    target[2] %= nz
                    rows.append(row)
                    cols.append(int(ei[component][tuple(target)]))
                    values.append(sign * direction * hv[row] / h)
    k = sp.coo_matrix((values, (rows, cols)), shape=(len(hc), len(ec))).tocsr()
    return YeeRegion(shape, h, y0, ei, hi, ec, hc, ev * eps, hv, k)


class PlanarOperator:
    def __init__(
        self,
        nx=4,
        ny_coarse=2,
        nz=4,
        h=0.25,
        ratio=3,
        epsilon_coarse=1.0,
        epsilon_fine=1.0,
        trace="published",
    ):
        if ratio not in (1, 3, 5) or min(nx, ny_coarse, nz) < 1 or h <= 0:
            raise ValueError("Positive dimensions and ratio 1,3,5 required")
        if trace not in ("published", "area", "circulation"):
            raise ValueError("trace must be published, area or circulation")
        self.trace = trace
        self.ratio = r = ratio
        self.coarse = c = region((nx, ny_coarse, nz), h, 0.0, epsilon_coarse)
        self.fine = f = region(
            (nx * r, ny_coarse * r, nz * r), h / r, ny_coarse * h, epsilon_fine
        )
        self.me_full = np.concatenate((c.me, f.me))
        self.mh = np.concatenate((c.mh, f.mh))
        self.k_full = sp.block_diag((c.coupling, f.coupling), format="csr")
        self.ecoords = np.concatenate((c.ecoords, f.ecoords))
        self.hcoords = np.concatenate((c.hcoords, f.hcoords))
        ne = len(self.me_full)
        constraints = {}
        # PEC exterior tangential E is eliminated. Interior rows have unique DOFs.
        fixed = set()
        for g, offset, j in ((c, 0, 0), (f, len(c.me), f.shape[1])):
            for component in (0, 2):
                fixed.update((g.eids[component][:, j, :].ravel() + offset).tolist())
        interface = set()
        for g, offset, j in ((c, 0, c.shape[1]), (f, len(c.me), 0)):
            for component in (0, 2):
                interface.update((g.eids[component][:, j, :].ravel() + offset).tolist())
        count = 0
        for row in range(ne):
            if row not in fixed and row not in interface:
                constraints[row] = [(count, 1.0)]
                count += 1
        self.interface_first = count
        groups = []
        for component in (0, 2):
            for i, k in product(range(nx), range(nz)):
                group_size = r if trace == "published" else r * r
                group = list(range(count, count + group_size))
                count += group_size
                groups.append(group)
                crow = int(c.eids[component][i, ny_coarse, k])
                if trace == "circulation":
                    constraints[crow] = [
                        (group[(r // 2) * r + j], 1.0 / r) for j in range(r)
                    ]
                else:
                    constraints[crow] = [(col, 1.0 / group_size) for col in group]
                # Tangential E constant along its own edge direction; varying
                # across the odd-ratio dual-width group around coarse vertex.
                for q in range(r):
                    for along in range(r):
                        col = group[q] if trace == "published" else group[q * r + along]
                        fi = (
                            (i * r + along)
                            if component == 0
                            else (i * r + q - r // 2) % (nx * r)
                        )
                        fk = (
                            (k * r + q - r // 2) % (nz * r)
                            if component == 0
                            else k * r + along
                        )
                        row = int(f.eids[component][fi, 0, fk]) + len(c.me)
                        if row in constraints:
                            raise RuntimeError("Duplicate interface owner")
                        constraints[row] = [(col, 1.0)]
        if set(constraints) != set(range(ne)) - fixed:
            raise RuntimeError("Missing electric trace constraint")
        rows, cols, vals = [], [], []
        for row, terms in sorted(constraints.items()):
            for col, val in terms:
                rows.append(row)
                cols.append(col)
                vals.append(val)
        self.p = sp.coo_matrix((vals, (rows, cols)), shape=(ne, count)).tocsr()
        self.me = (self.p.T @ sp.diags(self.me_full) @ self.p).tocsr()
        self.k = (self.k_full @ self.p).tocsr()
        # Me is scalar for interior DOFs and r-by-r (published) or r²-by-r² (area) for each independent trace.
        inv = sp.diags(1 / self.me.diagonal(), format="lil")
        for group in groups:
            block = self.me[group, :][:, group].toarray()
            inverse = np.linalg.inv(block)
            for i, a in enumerate(group):
                for j, b in enumerate(group):
                    inv[a, b] = inverse[i, j]
        self.me_inverse = inv.tocsr()
        self.groups = np.array(groups, dtype=int)
        self.h_update = (-sp.diags(1 / self.mh) @ self.k).tocsr()
        self.e_update = (self.me_inverse @ self.k.T).tocsr()
        self.fine_h = h / r
        self.lengths = (nx * h, 2 * ny_coarse * h, nz * h)
        self.e_components = np.concatenate(
            [
                np.full(a.size, c)
                for g in (self.coarse, self.fine)
                for c, a in enumerate(g.eids)
            ]
        )
        self.h_components = np.concatenate(
            [
                np.full(a.size, c)
                for g in (self.coarse, self.fine)
                for c, a in enumerate(g.hids)
            ]
        )
        self.min_epsilon = min(
            np.min(c.me / (self._e_volumes(c))), np.min(f.me / self._e_volumes(f))
        )

    @staticmethod
    def _e_volumes(g):
        v = np.full(len(g.me), g.h**3)
        for component in (0, 2):
            for j in (0, g.shape[1]):
                v[g.eids[component][:, j, :].ravel()] *= 0.5
        return v

    def project_e(self, physical):
        return self.me_inverse @ (self.p.T @ (self.me_full * np.asarray(physical)))

    def energy(self, e, h, dt):
        return float(
            0.5
            * (e @ (self.me @ e) + np.dot(h * self.mh, h) - dt * np.dot(h, self.k @ e))
        )

    def audit(self):
        residual = self.me @ self.e_update + self.h_update.T @ sp.diags(self.mh)
        inverse = self.me @ self.me_inverse - sp.eye(self.me.shape[0])

        def maxabs(matrix):
            return float(np.max(np.abs(matrix.data), initial=0.0))

        _, labels = connected_components(self.me, directed=False)
        return {
            "weighted_transpose_maxabs": maxabs(residual),
            "mass_inverse_maxabs": maxabs(inverse),
            "largest_electric_mass_block": int(np.bincount(labels).max()),
            "electric_dofs": self.me.shape[0],
            "magnetic_dofs": len(self.mh),
            "interface_groups": len(self.groups),
        }

    def normal_magnetic_jump(self, h):
        """Physical coarse-face B_n minus mean fine-face B_n at the interface.

        This is a diagnostic, not a derived composite Gauss operator. Local
        div B invariance alone does not certify this normal-flux continuity.
        """
        c, f, r = self.coarse, self.fine, self.ratio
        coarse = h[c.hids[1][:, -1, :]]
        fine = h[len(c.mh) + f.hids[1][:, 0, :]]
        averaged = fine.reshape(c.shape[0], r, c.shape[2], r).mean(axis=(1, 3))
        return coarse - averaged

    def magnetic_divergence(self, h):
        result = []
        offset = 0
        for g in (self.coarse, self.fine):
            fields = [h[offset + ids] for ids in g.hids]
            div = (np.roll(fields[0], -1, axis=0) - fields[0]) / g.h
            div += (fields[1][:, 1:, :] - fields[1][:, :-1, :]) / g.h
            div += (np.roll(fields[2], -1, axis=2) - fields[2]) / g.h
            result.append(div.ravel())
            offset += len(g.mh)
        return np.concatenate(result)


def make_sparse_step(double):
    dtype = wp.float64 if double else wp.float32

    @wp.kernel
    def step(
        offset: wp.array(dtype=int),
        index: wp.array(dtype=int),
        value: wp.array(dtype=dtype),
        source: wp.array(dtype=dtype),
        target: wp.array(dtype=dtype),
        dt: dtype,
    ):
        row = wp.tid()
        total = dtype(0.0)
        for j in range(offset[row], offset[row + 1]):
            total += value[j] * source[index[j]]
        target[row] += dt * total

    return step


step_float = make_sparse_step(False)
step_double = make_sparse_step(True)


class PlanarStepper:
    def __init__(self, operator, cfl=0.7, device="cpu", precision="float64"):
        if precision not in ("float32", "float64") or not 0 < cfl < 1:
            raise ValueError("precision float32/float64 and 0<CFL<1 required")
        self.operator = operator
        self.dt = cfl * operator.fine_h * np.sqrt(operator.min_epsilon) / np.sqrt(3)
        self.device = wp.get_device(device)
        self.dtype = wp.float64 if precision == "float64" else wp.float32
        self.kernel = step_double if precision == "float64" else step_float
        self.e = wp.zeros(operator.me.shape[0], dtype=self.dtype, device=self.device)
        self.h = wp.zeros(len(operator.mh), dtype=self.dtype, device=self.device)
        self.operators = []
        for matrix in (operator.h_update, operator.e_update):
            self.operators.append(
                [
                    wp.array(matrix.indptr.astype(np.int32), device=self.device),
                    wp.array(matrix.indices.astype(np.int32), device=self.device),
                    wp.array(matrix.data.astype(precision), device=self.device),
                ]
            )

    def set_fields(self, e, h):
        self.e.assign(np.asarray(e, dtype=self.e.numpy().dtype))
        self.h.assign(np.asarray(h, dtype=self.h.numpy().dtype))

    def step(self):
        wp.launch(
            self.kernel,
            len(self.h),
            inputs=[*self.operators[0], self.e, self.h, self.dt],
            device=self.device,
        )
        wp.launch(
            self.kernel,
            len(self.e),
            inputs=[*self.operators[1], self.h, self.e, self.dt],
            device=self.device,
        )

    def run(self, steps):
        for _ in range(steps):
            self.step()
        wp.synchronize_device(self.device)

    def fields(self):
        return self.e.numpy().copy(), self.h.numpy().copy()
