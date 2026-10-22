"""Physical contracts must not move with mesh resolution or outer padding."""

import unittest
from types import SimpleNamespace

from baseline.reference_protocol import layout, modal_steps


def settings(**kwargs):
    values = dict(
        reference_protocol="fixed_physical_v1",
        parameterization="smooth",
        voxel_nm=20.0,
        courant_factor=0.99,
        scene="bend",
        gradient="none",
        spacing_nm=20.0,
        domain_um=4.0,
        pml_cells=10,
        pad_domain=False,
        modal_window_fs=40.0,
        time_fs=300.0,
    )
    values.update(kwargs)
    return SimpleNamespace(**values)


class PhysicalReferenceTests(unittest.TestCase):
    def test_resolution_and_padding_preserve_design_coordinates(self):
        for spacing in (20, 10, 20 / 3, 5):
            for domain, pml in ((4.0, 200), (4.0, 400), (4.4, 400)):
                s = settings(
                    spacing_nm=spacing,
                    pml_cells=round(pml / spacing),
                    domain_um=domain,
                    pad_domain=pml != 200,
                )
                a = layout(s)
                for name, p in a["planes"].items():
                    self.assertAlmostEqual(
                        p["grid_index"] * spacing - a["origin_nm"][p["axis"]],
                        p["design_anchor_nm"],
                        places=10,
                    )
                self.assertEqual(
                    [
                        i * spacing - o
                        for i, o in zip(a["device_start"], a["origin_nm"])
                    ],
                    [2000, 400, 600],
                )
        with self.assertRaisesRegex(ValueError, "representable"):
            layout(settings(spacing_nm=30, pml_cells=10, pad_domain=True))
        with self.assertRaisesRegex(ValueError, "PML"):
            layout(settings(pml_cells=20))

    def test_window_uses_physical_time_across_step_sizes(self):
        s = settings()
        for dt_fs in (0.023, 0.04, 0.09):
            dt = dt_fs * 1e-15
            n = round(s.time_fs / dt_fs)
            samples = modal_steps(s, dt, n)
            self.assertGreaterEqual(samples[0] * dt * 1e15, 260)
            self.assertLess(samples[0] * dt * 1e15, 260 + dt_fs + 1e-10)
            self.assertLessEqual(samples[-1] * dt * 1e15, 300)
            self.assertGreater(samples[-1] * dt * 1e15, 300 - 2 * dt_fs - 1e-10)
            self.assertTrue(
                all(260 - 1e-10 <= i * dt * 1e15 <= 300 + 1e-10 for i in samples)
            )


if __name__ == "__main__":
    unittest.main()
