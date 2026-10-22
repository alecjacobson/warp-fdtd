"""Independently check the analytic Gaussian-weighted Fabry–Perot series."""

import unittest

import numpy as np
from scipy.integrate import quad

from warp_fdtd.slab_validation import analytic_transmission


class SlabAnalyticTests(unittest.TestCase):
    def test_series_against_spectral_quadrature(self):
        k0 = 2 * np.pi / 0.05
        for contrast, sigma, thickness in (
            (2.25, 0.06, 0.02),
            (12.25, 0.06, 0.02),
            (12.25, 0.025, 0.04),
            (2.25, 0.006, 0.02),
            (1.0, 0.06, 0.02),
        ):
            with self.subTest(contrast=contrast, sigma=sigma, thickness=thickness):
                n = np.sqrt(contrast)
                rho = ((1 - n) / (1 + n)) ** 2

                def density(k):
                    return (
                        np.exp(-0.5 * sigma**2 * (k - k0) ** 2)
                        + np.exp(-0.5 * sigma**2 * (k + k0) ** 2)
                    ) ** 2

                def transmitted(k):
                    return (
                        density(k)
                        * (1 - rho) ** 2
                        / (1 - 2 * rho * np.cos(2 * n * k * thickness) + rho**2)
                    )

                end = k0 + 10 / sigma
                denominator = quad(
                    density, 0, end, epsabs=1e-11, epsrel=1e-12, limit=500
                )[0]
                numerator = quad(
                    transmitted, 0, end, epsabs=1e-11, epsrel=1e-12, limit=500
                )[0]
                series, tail = analytic_transmission(contrast, thickness, sigma)
                self.assertLess(abs(series - numerator / denominator), 1e-12 + tail)
                self.assertLess(tail, 1e-12)


if __name__ == "__main__":
    unittest.main()
