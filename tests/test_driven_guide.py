import unittest
import numpy as np
from warp_fdtd.guide_problem import GuideProblem
from warp_fdtd.block_mesh import BlockMesh
from warp_fdtd.block_operator import BlockOperator
from warp_fdtd.planar_interface import PlanarStepper
from warp_fdtd.driven_blocks import DrivenBlocks
from warp_fdtd.driven_uniform import DrivenUniform
from warp_fdtd.guide_observer import stencil


class DrivenGuideTests(unittest.TestCase):
    def test_structured_block_update_matches_sparse_reference(self):
        from warp_fdtd.block_validation import make_mesh

        op = BlockOperator(
            make_mesh(3),
            epsilon=lambda p: np.where(p[:, 0] < 1.5, 12.25, 1.0),
            trace="linear",
        )
        a = PlanarStepper(op, device="cpu")
        b = DrivenBlocks(op, device="cpu", precision="float64", steps=64)
        rng = np.random.default_rng(95)
        e = rng.normal(size=op.me.shape[0])
        h = rng.normal(size=len(op.mh))
        a.set_fields(e, h)
        b.e.assign(e)
        b.h.assign(h)
        a.run(13)
        b.run(13)
        for x, y in zip(a.fields(), b.fields()):
            np.testing.assert_allclose(x, y, atol=1e-13)

    def test_uniform_partition_matches_direct_driven_kernel(self):
        problem = GuideProblem(
            length=0.8,
            width=0.8,
            height=0.8,
            source_x=0.4,
            sponge_width=0.1,
            pulse_center=0.3,
            pulse_sigma=0.1,
        )
        op = BlockOperator(
            BlockMesh((2, 2, 2), 0.4, cells_per_block=4, ratio=2, max_level=0),
            epsilon=problem,
            trace="linear",
        )
        a = DrivenBlocks(op, problem=problem, steps=128, device="cpu")
        b = DrivenUniform(problem, 0.1, 128, device="cpu")
        a.compile()
        b.compile()
        a.run(64)
        b.run(64)

        def direct(coords, components, shapes):
            offsets = np.cumsum([0] + [int(np.prod(s)) for s in shapes])
            result = []
            for p, c in zip(coords, components):
                shift = np.zeros(3) if shapes == b.eshapes else np.full(3, 0.5)
                shift[c] = 0.5 if shapes == b.eshapes else 0
                index = np.rint(p / 0.1 - shift).astype(int)
                result.append(
                    offsets[c] + np.ravel_multi_index(tuple(index), shapes[c])
                )
            return np.array(result)

        en, hn = a.fields()
        e_uniform = b.e.numpy()[direct(op.ecoords, op.e_components, b.eshapes)]
        h_uniform = b.h.numpy()[direct(op.hcoords, op.h_components, b.hshapes)]
        np.testing.assert_allclose(op.p @ en, e_uniform, atol=1e-7, rtol=2e-4)
        np.testing.assert_allclose(hn, h_uniform, atol=1e-7, rtol=2e-4)
        a.reset()
        b.reset()
        a.run(64)
        b.run(64)
        np.testing.assert_array_equal(a.e.numpy(), en)

    def test_sampler_affine_exact_even_at_block_boundary(self):
        shape = (4, 5, 5)
        h = 0.1
        origin = np.array([0.2, 0.3, 0.4])
        c = 0
        coords = np.indices(shape).reshape(3, -1).T.astype(float)
        coords[:, c] += 0.5
        coords = origin + coords * h
        values = 2 + coords @ np.array([3.0, -2.0, 4.0])
        for point in (origin, origin + [0.4, 0.4, 0.4], origin + [0.13, 0.21, 0.27]):
            indices, weights = stencil(point, origin, h, shape, c, True, 0)
            self.assertAlmostEqual(
                np.dot(values[indices], weights),
                2 + point @ [3.0, -2.0, 4.0],
                places=13,
            )


if __name__ == "__main__":
    unittest.main()
