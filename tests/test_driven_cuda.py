"""Run serially with the guide benchmarks; these checks use the GPU."""

import unittest
import numpy as np
import warp as wp
from warp_fdtd.block_mesh import BlockMesh
from warp_fdtd.block_operator import BlockOperator
from warp_fdtd.driven_blocks import DrivenBlocks
from warp_fdtd.driven_uniform import DrivenUniform
from warp_fdtd.guide_problem import GuideProblem


@unittest.skipUnless(wp.is_cuda_available(), "CUDA required")
class DrivenCudaTests(unittest.TestCase):
    def test_graph_source_counter_reset_and_factorization(self):
        p = GuideProblem(
            length=0.8,
            width=0.8,
            height=0.8,
            source_x=0.4,
            pulse_center=0.3,
            pulse_sigma=0.1,
            sponge_width=0.1,
        )
        mesh = BlockMesh((2, 2, 2), 0.4, cells_per_block=4, ratio=2, max_level=1)
        mesh.split(sorted(mesh.leaves)[0])
        mesh.geometry_leaf_count = len(mesh.leaves)
        mesh.balance()
        op = BlockOperator(mesh, p, trace="linear")
        for precision, atol in [("float64", 1e-12), ("float32", 2e-7)]:
            cpu = DrivenBlocks(
                op, p, 128, device="cpu", precision=precision, coupling="fused"
            )
            gpu = DrivenBlocks(op, p, 128, precision=precision)
            cpu.run(128)
            gpu.compile()
            gpu.run(128)
            for a, b in zip(cpu.fields(), gpu.fields()):
                np.testing.assert_allclose(a, b, rtol=1e-4, atol=atol)
            before = gpu.fields()
            gpu.reset()
            gpu.run(128)
            for a, b in zip(before, gpu.fields()):
                np.testing.assert_array_equal(a, b)
            self.assertEqual(int(gpu.counter.numpy()[0]), 128)
        a = DrivenUniform(p, 0.1, 128, device="cuda:0", layout="flat")
        b = DrivenUniform(p, 0.1, 128, device="cuda:0", layout="xyz")
        a.compile()
        b.compile()
        a.run(128)
        b.run(128)
        np.testing.assert_allclose(a.e.numpy(), b.e.numpy(), rtol=1e-4, atol=2e-7)
        np.testing.assert_allclose(a.h.numpy(), b.h.numpy(), rtol=1e-4, atol=2e-7)


if __name__ == "__main__":
    unittest.main()
