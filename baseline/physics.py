"""Execute upstream analytic physics checks on GPU, outside CPU-only conftest.

This deliberately uses unmodified upstream test functions. Each test asserts its
analytic tolerance. We record exactly which functions ran and on which backend.
"""

import importlib.util
import time
import traceback

import jax

from .common import ROOT, environment, write_json


def main():
    result = {"environment": environment(), "tests": []}
    assert jax.devices()[0].platform == "gpu"
    for filename in ["test_plane_wave.py", "test_fresnel.py"]:
        path = ROOT / "upstream/fdtdx/tests/simulation/physics" / filename
        spec = importlib.util.spec_from_file_location(path.stem, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        for name in sorted(n for n in vars(module) if n.startswith("test_")):
            start = time.perf_counter()
            row = {"file": str(path.relative_to(ROOT)), "function": name}
            try:
                getattr(module, name)()
                row["status"] = "passed"
            except Exception:
                row.update(status="failed", error=traceback.format_exc())
            row["seconds"] = time.perf_counter() - start
            result["tests"].append(row)
            print(row, flush=True)
            write_json(ROOT / "results/physics_gpu.json", result)
    if any(row["status"] != "passed" for row in result["tests"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
