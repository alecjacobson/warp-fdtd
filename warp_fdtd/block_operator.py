"""Closed 3D refined-block Yee correctness operator with circulation constraints.

All six exterior faces are PEC. Same-level edges share one electric unknown;
coarse edges restrict the integrated circulation of collinear fine segments.
The mesh is static. Host sparse assembly is for validation; time stepping uses
PlanarStepper's pure-Warp kernels. No sources, ports, CPML or adjoint yet.
"""

from dataclasses import dataclass
from itertools import product

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components


@dataclass
class ClosedRegion:
    shape: tuple
    h: float
    eids: list
    hids: list
    ecoords: np.ndarray
    hcoords: np.ndarray
    ekeys: list
    hkeys: list
    me: np.ndarray
    mh: np.ndarray
    coupling: sp.csr_matrix


def closed_region(block, mesh, epsilon):
    n = mesh.cells_per_block
    step = mesh.ratio ** (mesh.max_level - block.level)
    lo = np.array(mesh.integer_bounds(block)[0]) * n
    unit = mesh.root_width / (n * mesh.scale)
    h = step * unit
    es = [tuple(n if a == c else n + 1 for a in range(3)) for c in range(3)]
    hs = [tuple(n + 1 if a == c else n for a in range(3)) for c in range(3)]

    def layouts(shapes, electric):
        ids, coords, keys = [], [], []
        count = 0
        for c, shape in enumerate(shapes):
            count_next = count + int(np.prod(shape))
            ids.append(np.arange(count, count_next).reshape(shape))
            count = count_next
            for index in product(*(range(v) for v in shape)):
                corner = lo + np.array(index) * step
                offset = np.zeros(3) if electric else np.full(3, 0.5)
                offset[c] = 0.5 if electric else 0
                coords.append(np.array(mesh.origin) + (corner + offset * step) * unit)
                keys.append((c, *corner.tolist(), step))
        return ids, np.array(coords), keys

    ei, ec, ek = layouts(es, True)
    hi, hc, hk = layouts(hs, False)
    ev = np.full(len(ec), h**3)
    hv = np.full(len(hc), h**3)
    for c in range(3):
        for a in range(3):
            if a != c:
                for boundary in (0, n):
                    select = [slice(None)] * 3
                    select[a] = boundary
                    ev[ei[c][tuple(select)].ravel()] *= 0.5
        for boundary in (0, n):
            select = [slice(None)] * 3
            select[c] = boundary
            hv[hi[c][tuple(select)].ravel()] *= 0.5
    eps = np.broadcast_to(epsilon(ec) if callable(epsilon) else epsilon, ev.shape)
    if not np.isfinite(eps).all() or np.any(eps <= 0):
        raise ValueError("Positive finite epsilon required")
    rows, cols, vals = [], [], []
    for c, shape in enumerate(hs):
        a, b = (c + 1) % 3, (c + 2) % 3
        for index in product(*(range(v) for v in shape)):
            row = int(hi[c][index])
            for derivative, component, sign in ((a, b, 1), (b, a, -1)):
                for shift, direction in ((1, 1), (0, -1)):
                    target = list(index)
                    target[derivative] += shift
                    rows.append(row)
                    cols.append(int(ei[component][tuple(target)]))
                    vals.append(sign * direction * hv[row] / h)
    coupling = sp.coo_matrix((vals, (rows, cols)), shape=(len(hc), len(ec))).tocsr()
    return ClosedRegion(
        (n, n, n), h, ei, hi, ec, hc, ek, hk, ev * eps, hv, coupling
    ), float(eps.min())


class BlockOperator:
    def __init__(self, mesh, epsilon=1.0, trace="circulation"):
        if trace not in ("circulation", "linear"):
            raise ValueError("trace must be circulation or linear")
        self.trace = trace
        mesh.audit()
        if mesh.cells_per_block % mesh.ratio:
            raise ValueError(
                "First field prototype requires cells_per_block divisible by refinement ratio"
            )
        self.mesh = mesh
        pairs = [closed_region(b, mesh, epsilon) for b in sorted(mesh.leaves)]
        self.regions = [p[0] for p in pairs]
        self.min_epsilon = min(p[1] for p in pairs)
        self.fine_h = min(g.h for g in self.regions)
        self.lengths = tuple(v * mesh.root_width for v in mesh.root_shape)
        self.ecoords = np.concatenate([g.ecoords for g in self.regions])
        self.hcoords = np.concatenate([g.hcoords for g in self.regions])
        self.e_components = np.concatenate(
            [np.full(a.size, c) for g in self.regions for c, a in enumerate(g.eids)]
        )
        self.h_components = np.concatenate(
            [np.full(a.size, c) for g in self.regions for c, a in enumerate(g.hids)]
        )
        all_keys = [key for g in self.regions for key in g.ekeys]
        unique = set(all_keys)
        extent = np.array(mesh.integer_extent) * mesh.cells_per_block

        def pec(key):
            c, *rest = key
            return any(a != c and rest[a] in (0, extent[a]) for a in range(3))

        maps = {}
        if trace == "linear":
            # Compatible lowest-order edge interpolation: constant along the
            # edge and linear in the other tangential coordinate. Every fine
            # trace is coupled, including those between coarse edge lines.
            relations = {}
            blocks = sorted(mesh.leaves)
            for ia, ib, axis, _ in mesh.face_connections():
                if blocks[ia].level == blocks[ib].level:
                    continue
                ci, fi = (ia, ib) if blocks[ia].level < blocks[ib].level else (ib, ia)
                coarse, fine = blocks[ci], blocks[fi]
                coarse_lo = (
                    np.array(mesh.integer_bounds(coarse)[0]) * mesh.cells_per_block
                )
                coarse_hi = (
                    np.array(mesh.integer_bounds(coarse)[1]) * mesh.cells_per_block
                )
                fine_lo = np.array(mesh.integer_bounds(fine)[0]) * mesh.cells_per_block
                plane = (
                    coarse_hi[axis]
                    if coarse_hi[axis] == fine_lo[axis]
                    else coarse_lo[axis]
                )
                step = mesh.ratio ** (mesh.max_level - coarse.level)
                for key in self.regions[fi].ekeys:
                    component, x, y, z, _ = key
                    pos = np.array([x, y, z])
                    if component == axis or pos[axis] != plane:
                        continue
                    tangent = next(a for a in range(3) if a not in (component, axis))
                    base = coarse_lo + ((pos - coarse_lo) // step) * step
                    base[axis] = plane
                    fraction = (pos[tangent] - base[tangent]) / step
                    terms = {}
                    for shift, weight in ((0, 1 - fraction), (1, fraction)):
                        if weight == 0:
                            continue
                        target = base.copy()
                        target[tangent] += shift * step
                        target_key = (component, *target.tolist(), step)
                        if target_key not in unique:
                            raise ValueError("Missing coarse tangential edge")
                        terms[target_key] = weight
                    if key in relations and relations[key] != terms:
                        raise ValueError("Inconsistent face/edge/corner interpolation")
                    relations[key] = terms
            # Long edges first; fine-to-coarse constraints recurse upward.
            for key in sorted(unique, key=lambda k: (-k[-1], k)):
                if pec(key):
                    maps[key] = {}
                elif key in relations:
                    terms = {}
                    for parent, weight in relations[key].items():
                        for target, value in maps[parent].items():
                            terms[target] = terms.get(target, 0) + weight * value
                    maps[key] = terms
                else:
                    maps[key] = {key: 1.0}
        else:
            # Short segments first, so any coarse edge resolves recursively.
            for key in sorted(unique, key=lambda k: (k[-1], k)):
                if pec(key):
                    maps[key] = {}
                    continue
                c, x, y, z, size = key
                children = []
                if size % mesh.ratio == 0:
                    for j in range(mesh.ratio):
                        pos = [x, y, z]
                        pos[c] += j * (size // mesh.ratio)
                        children.append((c, *pos, size // mesh.ratio))
                found = [child in unique for child in children]
                if any(found) and not all(found):
                    raise ValueError("Partially subdivided coarse edge")
                if children and all(found):
                    terms = {}
                    for child in children:
                        for target, value in maps[child].items():
                            terms[target] = terms.get(target, 0) + value / mesh.ratio
                    maps[key] = terms
                else:
                    maps[key] = {key: 1.0}
        masters = sorted({target for terms in maps.values() for target in terms})
        master_ids = {key: i for i, key in enumerate(masters)}
        rows, cols, vals = [], [], []
        for i, key in enumerate(all_keys):
            for target, value in maps[key].items():
                rows.append(i)
                cols.append(master_ids[target])
                vals.append(value)
        self.p = sp.coo_matrix(
            (vals, (rows, cols)), shape=(len(all_keys), len(masters))
        ).tocsr()
        self.me_full = np.concatenate([g.me for g in self.regions])
        self.mh = np.concatenate([g.mh for g in self.regions])
        self.k_full = sp.block_diag([g.coupling for g in self.regions], format="csr")
        self.me = (self.p.T @ sp.diags(self.me_full) @ self.p).tocsr()
        self.k = (self.k_full @ self.p).tocsr()
        _, labels = connected_components(self.me, directed=False)
        sizes = np.bincount(labels)
        self.mass_block_sizes = sizes
        inverse = sp.diags(1 / self.me.diagonal(), format="lil")
        for label in np.flatnonzero(sizes > 1):
            group = np.flatnonzero(labels == label)
            if len(group) > (mesh.ratio**2 if trace == "circulation" else 256):
                raise ValueError("Unexpected nonlocal electric mass coupling")
            inv = np.linalg.inv(self.me[group, :][:, group].toarray())
            for i, a in enumerate(group):
                for j, b in enumerate(group):
                    inverse[a, b] = inv[i, j]
        self.me_inverse = inverse.tocsr()
        self.h_update = (-sp.diags(1 / self.mh) @ self.k).tocsr()
        self.e_update = (self.me_inverse @ self.k.T).tocsr()
        self.normal_flux = self._normal_flux_constraints()

    def _normal_flux_constraints(self):
        faces = {}
        offset = 0
        for g in self.regions:
            for local, key in enumerate(g.hkeys):
                faces.setdefault(key, []).append(offset + local)
            offset += len(g.mh)
        rows, cols, vals = [], [], []
        row = 0
        unit = self.mesh.root_width / (self.mesh.cells_per_block * self.mesh.scale)
        for key, ids in sorted(faces.items()):
            c, x, y, z, size = key
            area = (size * unit) ** 2
            for other in ids[1:]:
                rows.extend((row, row))
                cols.extend((ids[0], other))
                vals.extend((area, -area))
                row += 1
            if size % self.mesh.ratio:
                continue
            tangents = [a for a in range(3) if a != c]
            sub = size // self.mesh.ratio
            children = []
            for a, b in product(range(self.mesh.ratio), repeat=2):
                pos = [x, y, z]
                pos[tangents[0]] += a * sub
                pos[tangents[1]] += b * sub
                children.append((c, *pos, sub))
            if all(child in faces for child in children):
                rows.append(row)
                cols.append(ids[0])
                vals.append(area)
                for child in children:
                    rows.append(row)
                    cols.append(faces[child][0])
                    vals.append(-((sub * unit) ** 2))
                row += 1
        return sp.coo_matrix((vals, (rows, cols)), shape=(row, len(self.mh))).tocsr()

    def project_e(self, physical):
        return self.me_inverse @ (self.p.T @ (self.me_full * np.asarray(physical)))

    def energy(self, e, h, dt):
        return float(
            0.5
            * (e @ (self.me @ e) + np.dot(h * self.mh, h) - dt * np.dot(h, self.k @ e))
        )

    def audit(self):
        transpose = self.me @ self.e_update + self.h_update.T @ sp.diags(self.mh)
        flux = self.normal_flux @ self.h_update
        return {
            "electric_dofs": self.me.shape[0],
            "magnetic_dofs": len(self.mh),
            "largest_mass_block": int(self.mass_block_sizes.max()),
            "shared_or_refined_normal_faces": self.normal_flux.shape[0],
            "weighted_transpose_maxabs": float(
                np.max(np.abs(transpose.data), initial=0)
            ),
            "normal_flux_rate_maxabs": float(np.max(np.abs(flux.data), initial=0)),
        }
