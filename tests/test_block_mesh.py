"""Independent small-lattice coverage and geometric refinement checks."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from warp_fdtd.block_mesh import Block, BlockMesh, Box, GuidePolicy


class BlockMeshTests(unittest.TestCase):
    def test_coverage_balance_and_adjacency_against_dense_oracle(self):
        for ratio in (2, 3):
            mesh = BlockMesh((2, 1, 1), 1.0, ratio=ratio, max_level=3)
            mesh.split(Block(0, (0, 0, 0)))
            child = Block(1, (ratio - 1, ratio - 1, ratio - 1))
            mesh.split(child)
            child = Block(2, tuple(ratio * v + ratio - 1 for v in child.index))
            mesh.split(child)
            mesh.balance()
            mesh.audit()
            owners = np.full(mesh.integer_extent, -1, dtype=int)
            ordered = sorted(mesh.leaves)
            for i, block in enumerate(ordered):
                lo, hi = mesh.integer_bounds(block)
                sl = tuple(slice(a, b) for a, b in zip(lo, hi))
                self.assertTrue(np.all(owners[sl] == -1))
                owners[sl] = i
            self.assertTrue(np.all(owners >= 0))
            expected = {}
            for axis in range(3):
                left = np.take(owners, range(owners.shape[axis] - 1), axis=axis).ravel()
                right = np.take(owners, range(1, owners.shape[axis]), axis=axis).ravel()
                for a, b in zip(left, right):
                    if a != b:
                        key = (int(a), int(b), axis)
                        expected[key] = expected.get(key, 0) + 1
            self.assertEqual(
                expected,
                {
                    (int(a), int(b), int(axis)): int(area)
                    for a, b, axis, area in mesh.face_connections()
                },
            )
            # Independent dense check includes diagonal edge/corner touching.
            levels = np.array([b.level for b in ordered])[owners]
            from itertools import product

            for offset in product((-1, 0, 1), repeat=3):
                a = tuple(
                    slice(max(0, -d), min(n, n - d))
                    for d, n in zip(offset, levels.shape)
                )
                b = tuple(
                    slice(max(0, d), min(n, n + d))
                    for d, n in zip(offset, levels.shape)
                )
                self.assertLessEqual(int(np.max(np.abs(levels[a] - levels[b]))), 1)

    def test_thin_feature_detected_without_center_inside(self):
        guide = Box((0.49, 0.0, 0.49), (0.51, 2.0, 0.51))
        mesh = BlockMesh((2, 2, 2), 1.0, cells_per_block=4, max_level=3)
        policy = GuidePolicy((guide,), 0.03125, 0.0625, 0.25, growth=1.0)
        mesh.generate(policy)
        mesh.audit()
        for block in mesh.leaves:
            lo, hi = mesh.bounds(block)
            solid, surface = guide.distances(lo, hi)
            if surface == 0:
                self.assertEqual(block.level, 3)
            self.assertLessEqual(
                mesh.cell_width(block), policy.target(lo, hi) * (1 + 1e-12)
            )
        self.assertGreater(mesh.accounting()["ideal_cell_ratio"], 1.0)

    def test_roundtrip_determinism_and_invalid_mesh(self):
        mesh = BlockMesh((1, 1, 1), 1.0, max_level=2)
        mesh.split(Block(0, (0, 0, 0)))
        mesh.balance()
        with tempfile.TemporaryDirectory() as folder:
            p, q = Path(folder) / "a.json", Path(folder) / "b.json"
            mesh.save(p)
            BlockMesh.load(p).save(q)
            self.assertEqual(p.read_bytes(), q.read_bytes())
        mesh.leaves.add(Block(0, (0, 0, 0)))
        with self.assertRaisesRegex(ValueError, "Overlapping"):
            mesh.audit()
        with self.assertRaises(ValueError):
            BlockMesh((1, 1, 1), 1.0).generate(
                GuidePolicy((Box((0, 0, 0), (1, 1, 1)),), 0.0001, 0.1, 1.0)
            )

    def test_exact_distances(self):
        box = Box((0, 0, 0), (1, 1, 1))
        self.assertEqual(
            box.distances(np.array([0.2] * 3), np.array([0.8] * 3)),
            (0.0, 0.19999999999999996),
        )
        self.assertEqual(
            box.distances(np.array([0.5] * 3), np.array([1.5] * 3)), (0.0, 0.0)
        )
        np.testing.assert_allclose(
            box.distances(np.array([2.0] * 3), np.array([3.0] * 3)), [np.sqrt(3)] * 2
        )


if __name__ == "__main__":
    unittest.main()
