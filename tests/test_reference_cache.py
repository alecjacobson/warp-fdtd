"""Published metadata must not hide absent or stale regenerable field caches."""

import json
import tempfile
import unittest
from pathlib import Path

from baseline.reference_study import bundle_ready, digest, volume_ready


class ReferenceCacheTests(unittest.TestCase):
    def test_missing_stale_and_corrupt_cached_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bundle = root / "bundle"
            volume = root / "volume"
            bundle.mkdir()
            volume.mkdir()
            result = root / "result.json"
            result.write_text('{"status":"complete"}')
            inputs = bundle / "inputs.npz"
            inputs.write_bytes(b"exported inputs")
            manifest = dict(
                reference_result_sha256=digest(result),
                inputs_sha256=digest(inputs),
                physical_contract={"name": "fixed_physical_v1"},
            )
            (bundle / "manifest.json").write_text(json.dumps(manifest))
            self.assertTrue(bundle_ready(bundle, result))
            inputs.unlink()
            self.assertFalse(bundle_ready(bundle, result))
            inputs.write_bytes(b"wrong inputs")
            self.assertFalse(bundle_ready(bundle, result))
            inputs.write_bytes(b"exported inputs")
            chunk = volume / "chunk.npz"
            chunk.write_bytes(b"phasors")
            record = dict(
                status="complete",
                inputs_sha256=manifest["inputs_sha256"],
                physical_contract=manifest["physical_contract"],
                incident_normalization={"field_divisor": [1, 0]},
                files=[dict(file="chunk.npz", sha256=digest(chunk))],
            )
            path = volume / "result.json"
            path.write_text(json.dumps(record))
            self.assertTrue(volume_ready(volume, bundle))
            chunk.unlink()
            self.assertFalse(volume_ready(volume, bundle))
            chunk.write_bytes(b"corrupt")
            self.assertFalse(volume_ready(volume, bundle))
            chunk.write_bytes(b"phasors")
            record.pop("incident_normalization")
            path.write_text(json.dumps(record))
            self.assertFalse(volume_ready(volume, bundle))
            result.write_text('{"status":"changed"}')
            self.assertFalse(bundle_ready(bundle, result))


if __name__ == "__main__":
    unittest.main()
