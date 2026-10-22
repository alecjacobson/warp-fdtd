"""Independent uniform-grid and full constrained-system coupling checks."""

import unittest

import numpy as np
import scipy.linalg as la

from warp_fdtd.planar_interface import PlanarOperator, PlanarStepper, region


class PlanarInterfaceTests(unittest.TestCase):
    def test_constraint_equations_and_power(self):
        op = PlanarOperator(
            nx=2,
            ny_coarse=1,
            nz=2,
            h=0.5,
            ratio=3,
            epsilon_coarse=2.0,
            epsilon_fine=lambda p: 1.0 + p[:, 0],
        )
        rng = np.random.default_rng(74)
        e = rng.normal(size=op.me.shape[0])
        h = rng.normal(size=len(op.mh))
        # Independent dense solve of the constrained full-region equations.
        p = op.p.toarray()
        k = op.k_full.toarray()
        mass = p.T @ np.diag(op.me_full) @ p
        edot = la.solve(mass, p.T @ k.T @ h, assume_a="pos")
        hdot = -(k @ (p @ e)) / op.mh
        np.testing.assert_allclose(op.e_update @ h, edot, atol=1e-12)
        np.testing.assert_allclose(op.h_update @ e, hdot, atol=1e-12)
        self.assertLess(abs(edot @ mass @ e + np.dot(hdot * op.mh, h)), 1e-12)
        audit = op.audit()
        self.assertEqual(audit["largest_electric_mass_block"], 3)
        self.assertLess(audit["mass_inverse_maxabs"], 1e-14)
        # Actual trace equations, independent of matrix transpose construction.
        physical = op.p @ e
        c, f = op.coarse, op.fine
        r = 3
        for component in (0, 2):
            coarse = physical[c.eids[component][:, -1, :]]
            fine = physical[len(c.me) + f.eids[component][:, 0, :]]
            for i, k in np.ndindex(coarse.shape):
                samples = []
                for q in range(r):
                    along = []
                    for j in range(r):
                        fi = (
                            i * r + j
                            if component == 0
                            else (i * r + q - r // 2) % fine.shape[0]
                        )
                        fk = (
                            (k * r + q - r // 2) % fine.shape[1]
                            if component == 0
                            else k * r + j
                        )
                        along.append(fine[fi, fk])
                    np.testing.assert_array_equal(along, np.full(r, along[0]))
                    samples.append(along[0])
                self.assertAlmostEqual(coarse[i, k], np.mean(samples), places=14)

    def test_ratio_one_recovers_single_uniform_region(self):
        op = PlanarOperator(nx=3, ny_coarse=2, nz=3, h=0.2, ratio=1)
        g = region((3, 4, 3), 0.2, 0.0)

        def lookup(coords, ids):
            return {
                (c, *np.round(coords[j], 10)): j
                for c, a in enumerate(ids)
                for j in a.ravel()
            }

        ei, hi = lookup(g.ecoords, g.eids), lookup(g.hcoords, g.hids)
        e_map = np.array(
            [ei[(c, *np.round(p, 10))] for c, p in zip(op.e_components, op.ecoords)]
        )
        h_map = np.array(
            [hi[(c, *np.round(p, 10))] for c, p in zip(op.h_components, op.hcoords)]
        )
        rng = np.random.default_rng(71)
        e = rng.normal(size=len(g.me))
        h = rng.normal(size=len(g.mh))
        fixed = np.concatenate(
            [g.eids[c][:, j, :].ravel() for c in (0, 2) for j in (0, 4)]
        )
        e[fixed] = 0
        solver = PlanarStepper(op, device="cpu")
        solver.set_fields(op.project_e(e[e_map]), h[h_map])
        for _ in range(13):
            h -= solver.dt * (g.coupling @ e) / g.mh
            e += solver.dt * (g.coupling.T @ h) / g.me
            e[fixed] = 0
        solver.run(13)
        actual_e, actual_h = solver.fields()
        np.testing.assert_allclose(op.p @ actual_e, e[e_map], atol=2e-14)
        np.testing.assert_allclose(actual_h, h[h_map], atol=2e-14)

    def test_warp_update_energy_and_magnetic_divergence(self):
        op = PlanarOperator(nx=2, ny_coarse=1, nz=2, h=0.5)
        rng = np.random.default_rng(72)
        e = rng.normal(size=op.me.shape[0])
        h = rng.normal(size=len(op.mh))
        for precision, tol in (("float64", 1e-12), ("float32", 2e-5)):
            solver = PlanarStepper(op, device="cpu", precision=precision)
            solver.set_fields(e, h)
            initial_e, initial_h = solver.fields()
            initial_energy = op.energy(initial_e, initial_h, solver.dt)
            expected_h = initial_h + solver.dt * (op.h_update @ initial_e)
            expected_e = initial_e + solver.dt * (op.e_update @ expected_h)
            solver.run(1)
            a, b = solver.fields()
            np.testing.assert_allclose(a, expected_e, atol=tol, rtol=tol)
            np.testing.assert_allclose(b, expected_h, atol=tol, rtol=tol)
            solver.run(999)
            a, b = solver.fields()
            self.assertLess(abs(op.energy(a, b, solver.dt) / initial_energy - 1), tol)
            residual = op.magnetic_divergence(b - initial_h)
            self.assertLess(
                np.linalg.norm(residual) / np.linalg.norm(initial_h), tol * 30
            )


if __name__ == "__main__":
    unittest.main()
