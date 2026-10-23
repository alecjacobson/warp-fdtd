import unittest
import numpy as np
from warp_fdtd.block_mesh import BlockMesh
from warp_fdtd.block_operator import BlockOperator
from warp_fdtd.block_validation import make_mesh
from warp_fdtd.planar_interface import PlanarStepper


class BlockOperatorTests(unittest.TestCase):
    def test_uniform_partition_recovers_single_region(self):
        one = BlockOperator(
            BlockMesh((1, 1, 1), 3.0, cells_per_block=9, ratio=3, max_level=0)
        )
        split = BlockOperator(
            BlockMesh((3, 3, 3), 1.0, cells_per_block=3, ratio=3, max_level=0)
        )

        def mapping(coords, components, target_coords, target_components):
            lookup = {
                (int(c), *np.round(p, 10)): i
                for i, (c, p) in enumerate(zip(components, coords))
            }
            return np.array(
                [
                    lookup[(int(c), *np.round(p, 10))]
                    for c, p in zip(target_components, target_coords)
                ]
            )

        emap = mapping(one.ecoords, one.e_components, split.ecoords, split.e_components)
        hmap = mapping(one.hcoords, one.h_components, split.hcoords, split.h_components)
        rng = np.random.default_rng(66)
        e = rng.normal(size=one.me.shape[0])
        h = rng.normal(size=len(one.mh))
        a, b = PlanarStepper(one), PlanarStepper(split)
        a.set_fields(e, h)
        b.set_fields(split.project_e((one.p @ e)[emap]), h[hmap])
        a.run(23)
        b.run(23)
        ea, ha = a.fields()
        eb, hb = b.fields()
        np.testing.assert_allclose(split.p @ eb, (one.p @ ea)[emap], atol=1e-13)
        np.testing.assert_allclose(hb, ha[hmap], atol=1e-13)

    def test_closed_patch_material_energy_and_face_edge_corner_flux(self):
        op = BlockOperator(
            make_mesh(3), epsilon=lambda p: np.where(p[:, 0] < 1.5, 12.25, 1.0)
        )
        audit = op.audit()
        self.assertEqual(audit["largest_mass_block"], 3)
        self.assertLess(audit["weighted_transpose_maxabs"], 1e-13)
        self.assertLess(audit["normal_flux_rate_maxabs"], 1e-13)
        rng = np.random.default_rng(67)
        s = PlanarStepper(op, cfl=0.99)
        e = rng.normal(size=op.me.shape[0])
        h = op.h_update @ rng.normal(size=op.me.shape[0])
        s.set_fields(e, h)
        initial = op.energy(e, h, s.dt)
        flux = op.normal_flux @ h
        s.run(1000)
        en, hn = s.fields()
        self.assertLess(abs(op.energy(en, hn, s.dt) / initial - 1), 1e-12)
        np.testing.assert_allclose(op.normal_flux @ hn, flux, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
