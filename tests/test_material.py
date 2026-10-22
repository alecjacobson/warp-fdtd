import unittest
import numpy as np
from warp_fdtd.material import Material


class MaterialTests(unittest.TestCase):
    def test_pullback_finite_difference(self):
        rng = np.random.default_rng(51)
        p = rng.uniform(0.2, 0.8, (16, 16)).astype(np.float32)
        weights = rng.normal(size=p.shape).astype(np.float32)
        direction = rng.normal(size=p.shape).astype(np.float32)
        direction /= np.linalg.norm(direction)
        material = Material(p, 50)
        gradient = material.pullback(weights)
        analytic = float(np.sum(gradient.astype(float) * direction))
        errors = []
        for h in (0.01, 0.003, 0.001):
            plus = Material(p + h * direction, 50).inv_eps.numpy()
            minus = Material(p - h * direction, 50).inv_eps.numpy()
            finite = float(np.sum((plus.astype(float) - minus) * weights) / (2 * h))
            errors.append(abs(finite - analytic) / max(abs(analytic), 1e-12))
        self.assertLess(min(errors), 0.01)


if __name__ == "__main__":
    unittest.main()
