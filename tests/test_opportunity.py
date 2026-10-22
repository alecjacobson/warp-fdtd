"""Analytic reconstruction and integrity checks for the offline diagnostic."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from warp_fdtd.opportunity import load_volume, reconstruct


class OpportunityTests(unittest.TestCase):
    def test_complex_affine_reconstruction_with_partial_edge_cells(self):
        x, y, z = np.indices((9, 10, 11))
        f = (2 * x - y + 0.3 * z + 1j * (x + 3 * y - z)).astype(np.complex64)
        for ratio in (2, 4, 8):
            np.testing.assert_allclose(reconstruct(f, ratio), f, rtol=2e-7, atol=2e-6)

    def test_missing_and_overlapping_volume_chunks_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "chunk.npz"
            field = np.ones((6, 2, 3, 2), np.complex64) * (1 + 2j)
            np.savez_compressed(path, phasors=field)
            chunk = {
                "file": path.name,
                "z_start": 0,
                "z_stop": 2,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            meta = {"sample_shape": [2, 3, 4], "files": [chunk]}
            manifest = root / "result.json"
            manifest.write_text(json.dumps(meta))
            with self.assertRaisesRegex(ValueError, "Incomplete"):
                load_volume(root)
            meta["sample_shape"] = [2, 3, 2]
            meta["files"] = [chunk, chunk]
            manifest.write_text(json.dumps(meta))
            with self.assertRaisesRegex(ValueError, "overlapping"):
                load_volume(root)
            meta["files"] = [chunk]
            manifest.write_text(json.dumps(meta))
            np.testing.assert_array_equal(load_volume(root)[1], field)


if __name__ == "__main__":
    unittest.main()
