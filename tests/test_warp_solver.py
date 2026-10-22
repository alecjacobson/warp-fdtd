"""GPU regressions against independent, frozen FDTDX outputs."""

import sys
import unittest
import numpy as np
import warp as wp
from warp_fdtd.solver import Solver
from warp_fdtd.detectors import attach


class SolverTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.s = Solver("results/warp_reference/forward50_gpu", graph_steps=64)
        attach(cls.s)
        cls.s.compile()

    def test_source_staggering(self):
        s = self.s
        s.reset()
        s.launch_step()
        wp.synchronize()
        # At n=0 E's ramp is zero, while H samples n+1/2 and must be nonzero.
        self.assertEqual(float(np.max(np.abs(s.E.numpy()))), 0)
        self.assertGreater(float(np.max(np.abs(s.H.numpy()))), 1e-5)

    def test_full_3d_fields_flux_and_reset(self):
        s = self.s
        s.reset()
        s.run()
        a = s.fields()
        self.assertEqual(int(s.step.numpy()[0]), s.steps)
        with np.load("results/forward50_gpu/arrays.npz") as ref:
            for name in ("E", "H"):
                self.assertTrue(np.isfinite(a[name]).all())
                err = np.linalg.norm(a[name] - ref[name]) / np.linalg.norm(ref[name])
                self.assertLess(err, 1e-3)
            for m in s.monitors:
                b = m.flux.numpy()
                r = ref[m.name].ravel()
                self.assertLess(np.linalg.norm(b - r) / np.linalg.norm(r), 1e-3)
        s.reset()
        s.run()
        b = s.fields()
        for name in ("E", "H"):
            np.testing.assert_array_equal(a[name], b[name])
        self.assertNotIn("jax", sys.modules)
        self.assertNotIn("fdtdx", sys.modules)


if __name__ == "__main__":
    unittest.main()
