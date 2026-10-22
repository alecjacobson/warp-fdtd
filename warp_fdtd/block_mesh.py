"""Static geometry-driven Cartesian block forests (topology only, no field update).

Integer coordinates are in finest-block units. Each leaf owns B**3 cells;
covered parents have no field storage. Ratio 2 is an octree, ratio 3 a 27-tree.
All touching leaves (including edges/corners) differ by at most one level.
"""

from dataclasses import asdict, dataclass
from itertools import product
import json
from pathlib import Path

import numpy as np


@dataclass(frozen=True, order=True)
class Block:
    level: int
    index: tuple[int, int, int]


@dataclass(frozen=True)
class Box:
    lower: tuple[float, float, float]
    upper: tuple[float, float, float]

    def __post_init__(self):
        a, b = np.asarray(self.lower), np.asarray(self.upper)
        if (
            a.shape != (3,)
            or b.shape != (3,)
            or not np.isfinite([a, b]).all()
            or np.any(a >= b)
        ):
            raise ValueError("Box requires three finite increasing bounds")

    def distances(self, lower, upper):
        """Minimum distances from a block to this solid and its boundary.

        Exact for a single axis-aligned box. For a union we conservatively retain
        internal primitive surfaces too; no hidden assumption of a union SDF.
        """
        a, b = np.asarray(self.lower), np.asarray(self.upper)
        gap = np.maximum(np.maximum(a - upper, lower - b), 0)
        solid = float(np.linalg.norm(gap))
        inside = np.all(lower >= a) and np.all(upper <= b)
        surface = float(np.min(np.minimum(lower - a, b - upper))) if inside else solid
        return solid, surface


@dataclass(frozen=True)
class GuidePolicy:
    boxes: tuple[Box, ...]
    boundary_cell: float
    core_cell: float
    far_cell: float
    boundary_band: float = 0.0
    guide_buffer: float = 0.0
    growth: float = 0.5

    def __post_init__(self):
        values = (
            self.boundary_cell,
            self.core_cell,
            self.far_cell,
            self.boundary_band,
            self.guide_buffer,
            self.growth,
        )
        if not self.boxes or not np.isfinite(values).all():
            raise ValueError("Finite policy values and at least one box required")
        if not 0 < self.boundary_cell <= self.core_cell <= self.far_cell:
            raise ValueError("Require 0 < boundary_cell <= core_cell <= far_cell")
        if min(self.boundary_band, self.guide_buffer) < 0 or self.growth <= 0:
            raise ValueError("Nonnegative buffers and positive growth required")

    def target(self, lower, upper):
        target = self.far_cell
        for box in self.boxes:
            solid, surface = box.distances(lower, upper)
            target = min(
                target,
                self.core_cell + self.growth * max(0.0, solid - self.guide_buffer),
                self.boundary_cell
                + self.growth * max(0.0, surface - self.boundary_band),
            )
        return target


class BlockMesh:
    def __init__(
        self,
        root_shape,
        root_width,
        cells_per_block=8,
        ratio=2,
        max_level=3,
        origin=(0.0, 0.0, 0.0),
    ):
        if ratio not in (2, 3):
            raise ValueError("ratio must be 2 or 3")
        if isinstance(max_level, bool) or int(max_level) != max_level or max_level < 0:
            raise ValueError("Nonnegative integer max_level required")
        if (
            isinstance(cells_per_block, bool)
            or int(cells_per_block) != cells_per_block
            or cells_per_block < 2
        ):
            raise ValueError("Integer cells_per_block >=2 required")
        if len(root_shape) != 3 or any(int(v) != v or v < 1 for v in root_shape):
            raise ValueError("Three positive integer root counts required")
        if not np.isfinite(root_width) or root_width <= 0:
            raise ValueError("Finite positive root width required")
        if np.shape(origin) != (3,) or not np.isfinite(origin).all():
            raise ValueError("Three finite origin coordinates required")
        self.root_shape = tuple(int(v) for v in root_shape)
        self.root_width = float(root_width)
        self.cells_per_block = int(cells_per_block)
        self.ratio, self.max_level = int(ratio), int(max_level)
        self.origin = tuple(float(v) for v in origin)
        self.leaves = {
            Block(0, idx) for idx in product(*(range(v) for v in self.root_shape))
        }
        self.geometry_leaf_count = len(self.leaves)
        self.balance_splits = 0

    @property
    def scale(self):
        return self.ratio**self.max_level

    @property
    def integer_extent(self):
        return tuple(v * self.scale for v in self.root_shape)

    def integer_bounds(self, block):
        size = self.ratio ** (self.max_level - block.level)
        lower = tuple(v * size for v in block.index)
        return lower, tuple(v + size for v in lower)

    def bounds(self, block):
        lower, upper = self.integer_bounds(block)
        unit = self.root_width / self.scale
        return np.asarray(self.origin) + np.array(lower) * unit, np.asarray(
            self.origin
        ) + np.array(upper) * unit

    def cell_width(self, block):
        return self.root_width / (self.cells_per_block * self.ratio**block.level)

    def split(self, block):
        if block.level >= self.max_level:
            raise ValueError("Cannot split finest block")
        self.leaves.remove(block)
        children = [
            Block(
                block.level + 1,
                tuple(self.ratio * i + j for i, j in zip(block.index, offset)),
            )
            for offset in product(range(self.ratio), repeat=3)
        ]
        self.leaves.update(children)
        return children

    def generate(self, policy):
        finest = self.root_width / (self.cells_per_block * self.scale)
        if policy.boundary_cell < finest * (1 - 1e-12):
            raise ValueError(
                "Requested boundary cells are finer than the configured hierarchy"
            )
        pending = sorted(self.leaves, reverse=True)
        while pending:
            block = pending.pop()
            if block.level < self.max_level and self.cell_width(block) > policy.target(
                *self.bounds(block)
            ) * (1 + 1e-12):
                pending.extend(self.split(block))
        self.geometry_leaf_count = len(self.leaves)
        self.balance()
        return self

    def locate(self, point):
        """Find a leaf from integer point coordinates, without a dense owner map."""
        if any(p < 0 or p >= n for p, n in zip(point, self.integer_extent)):
            return None
        for level in range(self.max_level + 1):
            size = self.ratio ** (self.max_level - level)
            block = Block(level, tuple(int(p // size) for p in point))
            if block in self.leaves:
                return block
        raise RuntimeError("Mesh has a coverage hole")

    def touching_samples(self, block):
        """Samples sufficient to detect every coarser face/edge/corner neighbor.

        From a fine leaf, each coarser neighbor spans its full tangential extent.
        Fine neighbors are checked from their own side in the all-leaf traversal.
        Half-integer samples avoid ambiguous shared boundaries.
        """
        lo, hi = self.integer_bounds(block)
        for offset in product((-1, 0, 1), repeat=3):
            if offset == (0, 0, 0):
                continue
            point = tuple(
                lo[a] - 0.5
                if offset[a] < 0
                else hi[a] + 0.5
                if offset[a] > 0
                else (lo[a] + hi[a]) / 2
                for a in range(3)
            )
            neighbor = self.locate(point)
            if neighbor is not None:
                yield neighbor

    def balance(self):
        while True:
            split = set()
            for block in sorted(self.leaves):
                for other in self.touching_samples(block):
                    if other.level < block.level - 1:
                        split.add(other)
            if not split:
                return
            for block in sorted(split):
                self.split(block)
            self.balance_splits += len(split)

    def face_connections(self):
        """Unique (minus leaf, plus leaf, axis, overlap area in integer units).

        Fine-side face centers discover all coarse/fine tiles after balancing;
        deduplication removes the coarse-side discovery of one of these tiles.
        No fine-grid-sized lookup structure or pairwise leaf search is allocated.
        """
        ordered = sorted(self.leaves)
        ids = {block: i for i, block in enumerate(ordered)}
        connections = {}
        for block in ordered:
            lo, hi = self.integer_bounds(block)
            for axis, side in product(range(3), (-1, 1)):
                point = [(lo[a] + hi[a]) / 2 for a in range(3)]
                point[axis] = lo[axis] - 0.5 if side == -1 else hi[axis] + 0.5
                other = self.locate(point)
                if other is None:
                    continue
                a, b = (other, block) if side == -1 else (block, other)
                al, ah = self.integer_bounds(a)
                bl, bh = self.integer_bounds(b)
                area = 1
                for tangent in range(3):
                    if tangent != axis:
                        area *= min(ah[tangent], bh[tangent]) - max(
                            al[tangent], bl[tangent]
                        )
                connections[ids[a], ids[b], axis] = area
        return np.array(
            [(*key, area) for key, area in sorted(connections.items())], dtype=np.int64
        ).reshape(-1, 4)

    def audit(self):
        """Independent hierarchy/face-area checks; no dense cell allocation."""
        ordered = sorted(self.leaves)
        volume = 0
        for block in ordered:
            if not 0 <= block.level <= self.max_level or any(
                v < 0 or v >= n * self.ratio**block.level
                for v, n in zip(block.index, self.root_shape)
            ):
                raise ValueError("Invalid leaf index")
            for level in range(block.level):
                parent = Block(
                    level,
                    tuple(
                        v // self.ratio ** (block.level - level) for v in block.index
                    ),
                )
                if parent in self.leaves:
                    raise ValueError("Overlapping parent and child")
            volume += (self.ratio ** (self.max_level - block.level)) ** 3
            if any(
                abs(block.level - other.level) > 1
                for other in self.touching_samples(block)
            ):
                raise ValueError("Unbalanced touching leaves")
        if volume != int(np.prod(self.integer_extent)):
            raise ValueError("Incorrect covered volume")
        connections = self.face_connections()
        covered = np.zeros((len(ordered), 3, 2), dtype=np.int64)
        for a, b, axis, area in connections:
            if area <= 0:
                raise ValueError("Nonpositive face overlap")
            covered[a, axis, 1] += area
            covered[b, axis, 0] += area
        for i, block in enumerate(ordered):
            lo, hi = self.integer_bounds(block)
            for axis in range(3):
                expected = (hi[axis] - lo[axis]) ** 2
                for side in (0, 1):
                    boundary = (
                        lo[axis] == 0
                        if side == 0
                        else hi[axis] == self.integer_extent[axis]
                    )
                    if covered[i, axis, side] != (0 if boundary else expected):
                        raise ValueError(
                            "Face adjacency does not partition leaf boundary"
                        )
        return {
            "coverage": True,
            "nonoverlap": True,
            "strong_balance": True,
            "face_partition": True,
            "internal_face_tiles": len(connections),
        }

    def accounting(self, halo=1):
        if int(halo) != halo or halo < 0:
            raise ValueError("Nonnegative integer halo required")
        b = self.cells_per_block
        uniform = int(np.prod(self.root_shape)) * (b * self.scale) ** 3
        active = len(self.leaves) * b**3
        padded = len(self.leaves) * (b + 2 * halo) ** 3
        counts = {
            str(level): sum(block.level == level for block in self.leaves)
            for level in range(self.max_level + 1)
        }
        by_volume = {
            str(level): counts[str(level)]
            * self.ratio ** (3 * (self.max_level - level))
            / np.prod(self.integer_extent)
            for level in range(self.max_level + 1)
        }
        return {
            "leaf_blocks": len(self.leaves),
            "blocks_by_level": counts,
            "volume_fraction_by_level": by_volume,
            "active_cells": active,
            "uniform_finest_cells": uniform,
            "ideal_cell_ratio": uniform / active,
            "halo_width_cells": int(halo),
            "padded_cell_slots": padded,
            "padded_cell_ratio": uniform / padded,
            "geometry_leaf_count": self.geometry_leaf_count,
            "balance_split_count": self.balance_splits,
            "balance_added_leaf_count": len(self.leaves) - self.geometry_leaf_count,
            "covered_parent_field_cells": 0,
            "scope": "Topology cell counts; no solver allocation, runtime, accuracy, or peak-memory claim",
        }

    def payload(self):
        return {
            "schema": 1,
            "root_shape": self.root_shape,
            "root_width": self.root_width,
            "cells_per_block": self.cells_per_block,
            "ratio": self.ratio,
            "max_level": self.max_level,
            "origin": self.origin,
            "geometry_leaf_count": self.geometry_leaf_count,
            "balance_splits": self.balance_splits,
            "leaves": [asdict(block) for block in sorted(self.leaves)],
        }

    def save(self, path):
        Path(path).write_text(json.dumps(self.payload(), indent=2) + "\n")

    @classmethod
    def load(cls, path):
        data = json.loads(Path(path).read_text())
        if data.pop("schema") != 1:
            raise ValueError("Unsupported mesh schema")
        leaves = data.pop("leaves")
        count, splits = data.pop("geometry_leaf_count"), data.pop("balance_splits")
        mesh = cls(**data)
        parsed = [Block(row["level"], tuple(row["index"])) for row in leaves]
        if len(set(parsed)) != len(parsed):
            raise ValueError("Duplicate leaf")
        mesh.leaves = set(parsed)
        mesh.geometry_leaf_count, mesh.balance_splits = count, splits
        mesh.audit()
        return mesh
