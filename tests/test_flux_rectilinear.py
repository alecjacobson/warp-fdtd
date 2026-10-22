"""Flux updates must retain the Maxwell operator and its constraints."""

import os
import unittest

import numpy as np

from warp_fdtd.flux_rectilinear import FluxRectilinearPeriodic
from warp_fdtd.rectilinear import curl_numpy, divergence_numpy


class FluxRectilinearTests(unittest.TestCase):
    def test_material_operator_and_exact_capture_state(self):
        for accumulation in ("plain", "two_sum"):
            with self.subTest(accumulation=accumulation):
                self._check_material_operator(accumulation)

    def _check_material_operator(self, accumulation):
        edges = [
            np.array([0, 0.1, 0.4, 0.7, 1]),
            np.array([0, 0.12, 0.3, 0.5, 0.8, 1]),
            np.array([0, 0.1, 0.25, 0.4, 0.6, 0.85, 1]),
        ]
        rng = np.random.default_rng(72)
        eps = rng.uniform(1, 12, size=(3, 4, 5, 6))
        s = FluxRectilinearPeriodic(
            edges, eps, cfl=0.7, device=os.getenv("WARP_TEST_DEVICE", "cuda:0")
        )
        e, h = [rng.normal(size=eps.shape).astype("f") for _ in range(2)]
        s.set_fields(e, h)
        ee, hh = e.astype(float), h.astype(float)
        for _ in range(19):
            ee += s.dt * curl_numpy(hh, s.dual, False) / eps
            hh -= s.dt * curl_numpy(ee, s.primal, True)
        s.run(19)
        for actual, expected in zip(s.fields(), (ee, hh)):
            np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=3e-6)
        arrays = (s.E, s.H, s.correction_e, s.correction_h)
        before = [a.numpy().copy() for a in arrays]
        s.compile((7,))
        for a, expected in zip(arrays, before):
            np.testing.assert_array_equal(a.numpy(), expected)
        s.set_fields(e, h)
        s.run(19)
        for a, expected in zip(arrays, before):
            np.testing.assert_array_equal(a.numpy(), expected)

    def test_material_gauss_and_energy(self):
        n = 8
        u = np.linspace(0, 1, n + 1)
        edges = [u + g / (2 * np.pi) * np.sin(2 * np.pi * u) for g in (0.6, 0.3, 0.1)]
        rng = np.random.default_rng(513)
        eps = rng.choice([1.0, 12.0], size=(3, n, n, n))
        s = FluxRectilinearPeriodic(
            edges, eps, cfl=0.99, device=os.getenv("WARP_TEST_DEVICE", "cuda:0")
        )
        d = curl_numpy(rng.normal(size=eps.shape), s.dual, False)
        h = curl_numpy(rng.normal(size=eps.shape), s.primal, True)
        s.set_fields((d / eps).astype("f"), h.astype("f"))
        initial = s.fields()
        energy = s.energy(*initial)
        s.run(2048)
        ef, hf = s.fields()
        self.assertLess(abs(s.energy(ef, hf) / energy - 1), 1e-5)
        self.assertLess(
            np.linalg.norm(divergence_numpy(ef * eps, s.dual, False))
            / np.linalg.norm(ef * eps),
            1e-5,
        )
        self.assertLess(
            np.linalg.norm(divergence_numpy(hf, s.primal, True)) / np.linalg.norm(hf),
            1e-5,
        )


if __name__ == "__main__":
    unittest.main()
