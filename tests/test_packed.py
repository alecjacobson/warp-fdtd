"""Packed CPML is a storage change, including asymmetric and sparse active planes."""

import unittest
import numpy as np
import warp as wp
from warp_fdtd import kernels as k
from warp_fdtd.solver import Solver, pack_coefficients, owned_array_bytes
from warp_fdtd.detectors import attach


class PackedTests(unittest.TestCase):
    def test_asymmetric_full_state(self):
        shape = (7, 8, 9)
        n = int(np.prod(shape))
        rng = np.random.default_rng(8)
        fields = rng.normal(size=(2, 3, n)).astype("f")
        eps = wp.array(rng.uniform(0.1, 1, n).astype("f"))
        states = []
        for packed in (False, True):
            e, h = [wp.array(v) for v in fields]
            histories, coeffs = [], []
            for field in range(2):
                c = np.zeros((3, max(shape), 3), "f")
                c[..., 1:] = 1
                for axis, length in enumerate(shape):
                    active = [0, length - 1] if field == 0 else [1, 2, length - 2]
                    if axis == 1:
                        active = active[:1]
                    c[axis, active] = [-0.12, 0.8, 0.9]
                pc, size = pack_coefficients(c, shape)
                coeffs.append(
                    wp.array(pc if packed else c, dtype=wp.vec4 if packed else wp.vec3)
                )
                histories.append(wp.zeros((6, size if packed else n)))
            states.append((e, h, histories, coeffs))
        dummy = wp.zeros((3, 1))
        step = wp.zeros(1, dtype=int)
        for _ in range(20):
            for packed, (e, h, histories, coeffs) in enumerate(states):
                for field, kernel in enumerate(
                    (k.update_e_packed, k.update_h_packed)
                    if packed
                    else (k.update_e, k.update_h)
                ):
                    wp.launch(
                        kernel,
                        n,
                        inputs=[
                            e if field == 0 else h,
                            h if field == 0 else e,
                            eps,
                            histories[field],
                            coeffs[field],
                            dummy,
                            dummy,
                            step,
                            dummy,
                            1000,
                            *shape,
                            -1,
                            0.2,
                            0,
                        ],
                    )
        for component in range(2):
            np.testing.assert_array_equal(
                states[0][component].numpy(), states[1][component].numpy()
            )
        # Verify every active auxiliary value, not only final fields.
        x, y, z = np.indices(shape)
        for field in range(2):
            packed = states[1][3][field].numpy()
            full_history = states[0][2][field].numpy().reshape(6, *shape)
            compact_history = states[1][2][field].numpy()
            for axis, length in enumerate(shape):
                rank = packed[axis, :length, 3].astype(int)
                coords = [x, y, z]
                index = (
                    (rank[x] * shape[1] + y) * shape[2] + z
                    if axis == 0
                    else (
                        (rank[y] * shape[0] + x) * shape[2] + z
                        if axis == 1
                        else (rank[z] * shape[0] + x) * shape[1] + y
                    )
                )
                active = packed[axis, coords[axis], 0] != 0
                for slot in (2 * axis, 2 * axis + 1):
                    np.testing.assert_array_equal(
                        full_history[slot][active], compact_history[slot, index[active]]
                    )

    def test_reference_and_replay(self):
        solver = Solver("results/warp_reference/forward50_gpu", cpml_storage="packed")
        attach(solver)
        solver.compile()
        solver.run()
        first = solver.fields()
        with np.load("results/forward50_gpu/arrays.npz") as ref:
            for name, a in first.items():
                self.assertLess(
                    np.linalg.norm(a - ref[name]) / np.linalg.norm(ref[name]), 1e-3
                )
            for mon in solver.monitors:
                self.assertLess(
                    np.linalg.norm(mon.flux.numpy() - ref[mon.name].ravel())
                    / np.linalg.norm(ref[mon.name]),
                    1e-3,
                )
        solver.reset()
        solver.run()
        for name, a in solver.fields().items():
            np.testing.assert_array_equal(a, first[name])
        self.assertEqual(int(solver.step.numpy()[0]), solver.steps)
        self.assertLess(
            solver.psi_e.capacity + solver.psi_h.capacity, 12 * solver.n * 4
        )
        self.assertGreater(owned_array_bytes(solver), solver.E.capacity)


if __name__ == "__main__":
    unittest.main()
