"""A mode varying in all three axes exercises more than the trend fixture."""

import unittest
import numpy as np
from warp_fdtd.periodic import Periodic
from warp_fdtd.convergence import fit_frequency, PERIOD


class ObliqueModeTests(unittest.TestCase):
    def test_oblique_frequency_and_gauss_law(self):
        n = 16
        x, y, z = np.indices((n, n, n))
        for method in ("yee", "rk4"):
            with self.subTest(method=method):
                s = Periodic(n, 0.5, method, shape=(n, n, n))
                s.initial.fill(0)
                wave = (
                    np.cos(2 * np.pi * (x + y + z + 0.5) / n).astype(np.float32).ravel()
                )
                s.initial[0] = wave
                s.initial[1] = -wave
                s.reset()
                s.compile()
                s.run()
                ratio, residual, _ = fit_frequency(
                    s.signal.numpy(), s.dt * np.sqrt(3), damped=method == "rk4"
                )
                symbol = np.sin(np.pi / n)
                if method == "rk4":
                    symbol = 1.125 * symbol - np.sin(3 * np.pi / n) / 24
                space = 2 * 299792458 / (1550e-9 / n) * np.sqrt(3) * symbol
                omega = 2 * np.pi / PERIOD * np.sqrt(3)
                if method == "yee":
                    pred = 2 / s.dt * np.arcsin(space * s.dt / 2) / omega
                else:
                    a = 1j * space * s.dt
                    pred = (
                        np.angle(1 + a + a * a / 2 + a**3 / 6 + a**4 / 24)
                        / s.dt
                        / omega
                    )
                self.assertLess(abs(ratio / pred - 1), 2e-6)
                self.assertLess(residual, 1e-4)
                fields = s.state.numpy().reshape(6, n, n, n)
                div = sum(fields[c] - np.roll(fields[c], 1, axis=c) for c in range(3))
                self.assertLess(np.linalg.norm(div) / np.linalg.norm(fields[:3]), 2e-5)


if __name__ == "__main__":
    unittest.main()
