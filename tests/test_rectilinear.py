"""Compare the GPU operator with independent NumPy curls on unequal axes."""

import unittest

import numpy as np

from warp_fdtd.rectilinear import RectilinearPeriodic, curl_numpy


class RectilinearTests(unittest.TestCase):
    def test_nonuniform_material_update_and_graph_replay(self):
        for precision, compensated in (
            ("float32", False),
            ("float32", True),
            ("float64", False),
        ):
            with self.subTest(precision=precision, compensated=compensated):
                edges = [
                    np.array([0, 0.1, 0.4, 0.7, 1.0]),
                    np.array([0, 0.12, 0.3, 0.5, 0.8, 1.0]),
                    np.array([0, 0.1, 0.25, 0.4, 0.6, 0.85, 1.0]),
                ]
                rng = np.random.default_rng(72)
                eps = rng.uniform(1, 4, size=(3, 4, 5, 6))
                solver = RectilinearPeriodic(
                    edges,
                    permittivity=eps,
                    cfl=0.7,
                    precision=precision,
                    compensated=compensated,
                )
                e, h = [rng.normal(size=eps.shape).astype("f") for _ in range(2)]
                solver.set_fields(e, h)
                solver.compile((19,))
                for actual, expected in zip(solver.fields(), (e, h)):
                    np.testing.assert_array_equal(actual, expected)
                ee, hh = e.astype(float), h.astype(float)
                initial_energy = solver.energy(ee, hh)
                for _ in range(19):
                    ee += solver.dt * curl_numpy(hh, solver.dual, False) / eps
                    hh -= solver.dt * curl_numpy(ee, solver.primal, True)
                solver.run(19)
                actual = solver.fields()
                for a, b in zip(actual, (ee, hh)):
                    np.testing.assert_allclose(
                        a,
                        b,
                        rtol=2e-5 if precision == "float32" else 2e-12,
                        atol=3e-6 if precision == "float32" else 2e-12,
                    )
                self.assertLess(abs(solver.energy(*actual) / initial_energy - 1), 1e-6)
                corrections = (solver.correction_e.numpy(), solver.correction_h.numpy())
                solver.compile((7,))
                for a, b in zip(solver.fields(), actual):
                    np.testing.assert_array_equal(a, b)
                for a, b in zip(
                    (solver.correction_e.numpy(), solver.correction_h.numpy()),
                    corrections,
                ):
                    np.testing.assert_array_equal(a, b)
                solver.set_fields(e, h)
                solver.run(19)
                for a, b in zip(solver.fields(), actual):
                    np.testing.assert_array_equal(a, b)


if __name__ == "__main__":
    unittest.main()
