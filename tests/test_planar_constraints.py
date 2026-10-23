"""Independent physical checks of the composite interface diagnostics."""

import unittest
import numpy as np
from warp_fdtd.planar_interface import PlanarOperator
from warp_fdtd.planar_constraints import (
    electric_pillboxes,
    normal_magnetic_fluxes,
    magnetic_gradient_probe,
)


class PlanarConstraintTests(unittest.TestCase):
    def test_pillbox_gauss_for_affine_displacement(self):
        for ratio in (1, 3, 5):
            op = PlanarOperator(
                nx=4,
                ny_coarse=2,
                nz=4,
                h=0.25,
                ratio=ratio,
                trace="circulation",
                epsilon_coarse=12.25,
                epsilon_fine=lambda p: 1 + 99 * (p[:, 0] < 0.5),
            )
            eps = np.concatenate(
                (
                    op.coarse.me / op._e_volumes(op.coarse),
                    op.fine.me / op._e_volumes(op.fine),
                )
            )
            # D=(x,2y,3z) has divergence 6. Use interior transverse cells so
            # the deliberately nonperiodic affine field doesn't cross the seam.
            displacement = (op.e_components + 1) * op.ecoords[
                np.arange(len(eps)), op.e_components
            ]
            charge = (electric_pillboxes(op) @ (displacement / eps)).reshape(4, 4)
            volume = op.coarse.h**2 * (op.coarse.h + op.fine.h) / 2
            np.testing.assert_allclose(charge[1:-1, 1:-1], 6 * volume, atol=1e-14)
            displacement = np.array([2.0, -3.0, 4.0])[op.e_components]
            np.testing.assert_allclose(
                electric_pillboxes(op) @ (displacement / eps), 0, atol=1e-14
            )

    def test_normal_flux_quadrature_on_affine_fields(self):
        op = PlanarOperator(
            nx=4, ny_coarse=2, nz=4, h=0.25, ratio=3, trace="circulation"
        )
        # Point midpoints and fine-face midpoint sums integrate affine B exactly.
        h = np.zeros(len(op.mh))
        normal = op.h_components == 1
        h[normal] = 1 + op.hcoords[normal, 0] + 2 * op.hcoords[normal, 2]
        np.testing.assert_allclose(normal_magnetic_fluxes(op) @ h, 0, atol=1e-14)

    def test_circulation_preserves_charge_and_normal_flux_for_materials(self):
        for ratio in (1, 3, 5):
            op = PlanarOperator(
                nx=4,
                ny_coarse=2,
                nz=4,
                h=0.25,
                ratio=ratio,
                trace="circulation",
                epsilon_coarse=12.25,
                epsilon_fine=lambda p: 1 + 99 * (p[:, 0] < 0.5),
            )
            a = electric_pillboxes(op) @ op.p @ op.e_update
            b = normal_magnetic_fluxes(op) @ op.h_update
            self.assertLess(np.max(np.abs(a.data), initial=0), 1e-13)
            self.assertLess(np.max(np.abs(b.data), initial=0), 1e-13)
            probe = magnetic_gradient_probe(op)
            self.assertLess(probe["fine_curl_maxabs"], 1e-11)
            self.assertTrue(probe["commuting_trace_gate_passed"])

    def test_negative_controls_detect_different_trace_failures(self):
        area = PlanarOperator(trace="area")
        self.assertFalse(magnetic_gradient_probe(area)["commuting_trace_gate_passed"])
        rate = normal_magnetic_fluxes(area) @ area.h_update
        self.assertGreater(np.max(np.abs(rate.data)), 1e-3)
        published = PlanarOperator(trace="published")
        rate = electric_pillboxes(published) @ published.p @ published.e_update
        self.assertGreater(np.max(np.abs(rate.data)), 1e-3)


if __name__ == "__main__":
    unittest.main()
