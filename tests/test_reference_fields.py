"""Physical power and arbitrary source/eigenmode phase must cancel independently."""

import unittest

import numpy as np

from warp_fdtd.reference_fields import incident_normalization


class ReferenceFieldTests(unittest.TestCase):
    def test_physical_amplitude_and_phase_invariance(self):
        normalized = []
        for n, dx, phase, amplitude in (
            (4, 0.5, 0.3, 2 + 3j),
            (8, 0.25, -0.8, -1 + 0.7j),
        ):
            mode_e = np.zeros((3, 1, n, n), np.complex128)
            mode_h = np.zeros_like(mode_e)
            # 0.5 sum(E cross H*) = 1, and the same physical mode after /dx.
            mode_e[1] = np.sqrt(2) / n * np.exp(1j * phase)
            mode_h[2] = mode_e[1]
            phasors = np.concatenate((mode_e, mode_h)) * amplitude
            a = incident_normalization(
                phasors, mode_e, mode_h, 0, np.ones((1, n, n)), dx
            )
            factor = complex(*a["field_divisor"])
            normalized.append((phasors / factor)[:, 0, 0, 0])
            self.assertAlmostEqual(
                a["area_weighted_incident_power"],
                abs(amplitude) ** 2 * dx**2,
                places=12,
            )
            self.assertEqual(a["canonical_mode_component"], 1)
        np.testing.assert_allclose(*normalized, atol=1e-14, rtol=1e-14)


class CommonPhysicalSamplesTests(unittest.TestCase):
    def test_padding_translation_and_reject_interpolation(self):
        from warp_fdtd.reference_report import common_indices

        targets = [
            np.arange(200, 3801, 20.0),
            np.arange(200, 3801, 20.0),
            np.arange(210, 1300, 20.0),
        ]
        for origin in ([0, 0, 0], [200, 200, 200], [200, 200, 0]):
            axes = [
                (np.arange(n) * 20.0 + (10 if a == 2 else 0)) * 1e-9
                for a, n in enumerate([220, 220, 95])
            ]
            indices = common_indices(axes, origin, targets)
            grid_origin = np.array([-2e-6, -2e-6, -0.75e-6])
            translated = common_indices(
                [a + o for a, o in zip(axes, grid_origin)], origin, targets, grid_origin
            )
            for a, b in zip(indices, translated):
                np.testing.assert_array_equal(a, b)
            for axis, offset, idx, target in zip(axes, origin, indices, targets):
                np.testing.assert_allclose(axis[idx] * 1e9 - offset, target, atol=1e-10)
        with self.assertRaisesRegex(ValueError, "common physical points"):
            common_indices(axes, [0, 0, 0], [targets[0] + 1, *targets[1:]])

    def test_norm_and_power_scaling(self):
        from warp_fdtd.reference_report import poynting, relative_l2

        fields = np.zeros((6, 3, 4, 5), complex)
        fields[0] = 1 + 2j
        fields[4] = 1 + 2j
        self.assertAlmostEqual(relative_l2(0.01 * fields[:3], fields[:3], 12), 0.01)
        np.testing.assert_allclose(poynting(2 * fields), 4 * poynting(fields))


if __name__ == "__main__":
    unittest.main()
