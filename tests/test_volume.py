"""Independent affine-field and legacy-plane checks for volume phasors."""

import unittest

import numpy as np
import warp as wp

from warp_fdtd.detectors import record
from warp_fdtd.volume import record_volume


class VolumeTests(unittest.TestCase):
    def test_affine_physical_sampling_and_time_centering(self):
        shape = (7, 8, 9)
        xyz = np.indices(shape).astype(float)
        offsets = np.array(
            [
                [0.5, 0, 0],
                [0, 0.5, 0],
                [0, 0, 0.5],
                [0, 0.5, 0.5],
                [0.5, 0, 0.5],
                [0.5, 0.5, 0],
            ]
        )
        points = np.array([[2.1, 3.4, 4.2], [1.3, 2.5, 3.7], [4.8, 5.1, 6.3]], "f")
        fields = []
        for c in range(6):
            position = xyz + offsets[c, :, None, None, None]
            fields.append(
                (position[0] + 2 * position[1] - 3 * position[2] + c)
                .astype("f")
                .ravel()
            )
        e = wp.array(np.array(fields[:3]))
        h = wp.array(4 * np.array(fields[3:]))
        prev = wp.array(2 * np.array(fields[3:]))
        weights = wp.array(np.array([[0, 0], [0.2, 0.3]], "f"), dtype=wp.vec2)
        step = wp.array(np.array([0], "i"))
        out = wp.zeros((6, len(points)), dtype=wp.vec2)
        args = [
            e,
            h,
            prev,
            wp.array(points, dtype=wp.vec3),
            weights,
            step,
            out,
            *shape,
            1,
        ]
        wp.launch(record_volume, len(points), inputs=args)
        np.testing.assert_array_equal(out.numpy(), 0)
        step.assign(np.array([1], "i"))
        wp.launch(record_volume, len(points), inputs=args)
        raw = out.numpy()
        actual = raw[..., 0] + 1j * raw[..., 1]
        for c in range(6):
            v = points[:, 0] + 2 * points[:, 1] - 3 * points[:, 2] + c
            expected = v * (1 if c < 3 else 3) * (0.2 + 0.3j)
            np.testing.assert_allclose(actual[c], expected, rtol=2e-6, atol=3e-6)

    def test_matches_legacy_colocation_including_zero_halos(self):
        shape = (7, 8, 9)
        n = int(np.prod(shape))
        rng = np.random.default_rng(18)
        e, h, prev = [wp.array(rng.normal(size=(3, n)).astype("f")) for _ in range(3)]
        weights = wp.array(np.array([[0.3, -0.2]], "f"), dtype=wp.vec2)
        step = wp.zeros(1, dtype=int)
        indices = wp.zeros(1, dtype=int)
        a = wp.zeros((6, n), dtype=wp.vec2)
        b = wp.zeros_like(a)
        points = np.indices(shape).astype("f")
        points[2] += 0.5
        points = wp.array(points.reshape(3, -1).T.copy(), dtype=wp.vec3)
        wp.launch(
            record_volume, n, inputs=[e, h, prev, points, weights, step, a, *shape, 0]
        )
        wp.launch(
            record,
            n,
            inputs=[
                e,
                h,
                prev,
                step,
                weights,
                indices,
                b,
                wp.zeros(1),
                wp.ones(n),
                *shape,
                0,
                0,
                0,
                shape[1],
                shape[2],
                2,
                1,
                1,
            ],
        )
        np.testing.assert_allclose(a.numpy(), b.numpy(), rtol=2e-6, atol=2e-7)


if __name__ == "__main__":
    unittest.main()
