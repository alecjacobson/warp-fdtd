"""Small exact-transpose identity including CPML and boundary padding."""

import unittest
import numpy as np
import warp as wp
from warp_fdtd import kernels as k
from warp_fdtd.adjoint import prepare, gather


class AdjointTests(unittest.TestCase):
    def test_cpml_full_state_transpose(self):
        rng = np.random.default_rng(71)
        shape = (6, 7, 8)
        n = int(np.prod(shape))
        device = "cuda:0"

        def array(a):
            return wp.array(np.asarray(a, np.float32), dtype=float, device=device)

        initial = [
            rng.normal(size=(c, n)).astype(np.float32) * 0.1 for c in (3, 3, 6, 6)
        ]
        cot = [rng.normal(size=a.shape).astype(np.float32) * 0.1 for a in initial]
        e, h, pe, ph = [array(a) for a in initial]
        qe, qh, qpe, qph = [array(a) for a in cot]
        eps = array(rng.uniform(0.2, 1, n))
        ce = np.zeros((3, max(shape), 3), np.float32)
        ce[:, :, 1:] = 1
        ch = ce.copy()
        for axis, length in enumerate(shape):
            ce[axis, [0, length - 1]] = [-0.1, 0.8, 1.2]
            ch[axis, [0, length - 1]] = [-0.13, 0.75, 1.1]
        ce = wp.array(ce, dtype=wp.vec3, device=device)
        ch = wp.array(ch, dtype=wp.vec3, device=device)
        step = wp.zeros(1, dtype=int, device=device)
        dummy = wp.zeros((3, n), dtype=float, device=device)
        shared = [dummy, dummy, step, dummy, 1, *shape, -1, 0.4, 0]
        wp.launch(k.update_e, n, inputs=[e, h, eps, pe, ce, *shared], device=device)
        wp.launch(k.update_h, n, inputs=[h, e, eps, ph, ch, *shared], device=device)
        outputs = [a.numpy() for a in (e, h, pe, ph)]
        d = wp.empty_like(pe)
        hist = wp.zeros((1, 3, 1), dtype=float, device=device)
        grad = wp.zeros(1, dtype=float, device=device)
        wp.launch(
            prepare, n, inputs=[qh, eps, qph, ch, d, *shape, 0.4, 0], device=device
        )
        args = [d, hist, grad, step, *shape, 0, 0, 0, 0, 0, 0]
        wp.launch(gather, n, inputs=[qe, *args, 1], device=device)
        wp.launch(
            prepare, n, inputs=[qe, eps, qpe, ce, d, *shape, 0.4, 1], device=device
        )
        wp.launch(gather, n, inputs=[qh, *args, 0], device=device)
        transposed = [a.numpy() for a in (qe, qh, qpe, qph)]
        # Outside PML the unused psi variables are identity, as in the kernels.
        lhs = sum(np.sum(a.astype(float) * b) for a, b in zip(outputs, cot))
        rhs = sum(np.sum(a.astype(float) * b) for a, b in zip(initial, transposed))
        self.assertLess(abs(lhs - rhs) / max(abs(lhs), abs(rhs)), 2e-6)


class DetectorTransposeTests(unittest.TestCase):
    def test_space_time_colocation_transpose(self):
        from warp_fdtd.detectors import record
        from warp_fdtd.adjoint import inject

        rng = np.random.default_rng(17)
        shape = (6, 7, 8)
        n = int(np.prod(shape))
        device = "cuda:0"
        e, h, hp = [
            wp.array(
                rng.normal(size=(3, n)).astype(np.float32), dtype=float, device=device
            )
            for _ in range(3)
        ]
        sx, sy, sz = 1, 7, 8
        count = sx * sy * sz
        step = wp.zeros(1, dtype=int, device=device)
        indices = wp.zeros(1, dtype=int, device=device)
        weights = wp.array(
            np.array([[0.3, 0.7]], np.float32), dtype=wp.vec2, device=device
        )
        ph = wp.zeros((6, count), dtype=wp.vec2, device=device)
        flux = wp.zeros(1, dtype=float, device=device)
        area = wp.ones(count, dtype=float, device=device)
        wp.launch(
            record,
            count,
            inputs=[
                e,
                h,
                hp,
                step,
                weights,
                indices,
                ph,
                flux,
                area,
                *shape,
                2,
                0,
                0,
                sy,
                sz,
                0,
                1,
                1,
            ],
            device=device,
        )
        source_np = rng.normal(size=(6, count, 2)).astype(np.float32)
        source = wp.array(source_np, dtype=wp.vec2, device=device)
        qe = wp.zeros_like(e)
        qh = wp.zeros_like(h)
        args = [qe, qh, source, weights, step, *shape, 2, 0, 0, sy, sz]
        wp.launch(inject, count, inputs=[*args, 0], device=device)
        rhs = np.sum(qe.numpy().astype(float) * e.numpy()) + np.sum(
            qh.numpy().astype(float) * h.numpy()
        )
        qh.zero_()
        wp.launch(inject, count, inputs=[*args, 1], device=device)
        rhs += np.sum(qh.numpy().astype(float) * hp.numpy())
        v = ph.numpy()
        lhs = np.sum(
            source_np[..., 0].astype(float) * v[..., 0]
            - source_np[..., 1].astype(float) * v[..., 1]
        )
        self.assertLess(abs(lhs - rhs) / max(abs(lhs), abs(rhs)), 2e-6)


if __name__ == "__main__":
    unittest.main()
