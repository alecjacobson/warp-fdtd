"""Numerical comparisons and explicit acceptance gates, using saved evidence."""

import json

import numpy as np

from .common import ROOT, write_json


def main():
    result = {}
    root = ROOT / "results"
    cpu = root / "forward50_cpu/arrays.npz"
    gpu = root / "forward50_gpu/arrays.npz"
    if cpu.exists() and gpu.exists():
        with np.load(cpu) as c, np.load(gpu) as g:
            scope = "full_3d_fields" if c["E"].ndim == 4 else "cross_sections"
            errors = {
                k: float(np.linalg.norm(c[k] - g[k]) / max(np.linalg.norm(c[k]), 1e-30))
                for k in ("E", "H", "input", "output")
            }
        ct = json.loads((cpu.parent / "result.json").read_text())["median_seconds"]
        gt = json.loads((gpu.parent / "result.json").read_text())["median_seconds"]
        result["cpu_gpu"] = {
            "scope": scope,
            "relative_l2_errors": errors,
            "passed": max(errors.values()) < 1e-3,
            "speedup": ct / gt,
        }
    refpath = root / "gradient50_checkpointed/gradients.npz"
    if refpath.exists():
        with np.load(refpath) as a:
            ref = np.concatenate([a[k].ravel().astype(float) for k in a.files])
        result["gradients"] = {}
        for method in ("exact", "paper", "compressed"):
            path = root / f"gradient50_{method}"
            with np.load(path / "gradients.npz") as a:
                g = np.concatenate([a[k].ravel().astype(float) for k in a.files])
            err = float(np.linalg.norm(g - ref) / np.linalg.norm(ref))
            cosine = float(g @ ref / (np.linalg.norm(g) * np.linalg.norm(ref)))
            record = json.loads((path / "result.json").read_text())
            fd_error = min(row["relative_error"] for row in record["finite_difference"])
            result["gradients"][method] = {
                "relative_l2_error": err,
                "cosine": cosine,
                "best_finite_difference_relative_error": fd_error,
                "passed": err < (1e-4 if method == "exact" else 0.05)
                and cosine > 0.99
                and fd_error < 0.01,
                "thresholds": {
                    "relative_l2_error": 1e-4 if method == "exact" else 0.05,
                    "cosine_min": 0.99,
                    "finite_difference_error": 0.01,
                },
            }
    adjoint_path = root / "gradient50_adjoint/gradients.npz"
    if adjoint_path.exists():
        with np.load(root / "gradient50_modal/gradients.npz") as a:
            ref = np.concatenate([a[k].ravel().astype(float) for k in a.files])
        with np.load(adjoint_path) as a:
            g = np.concatenate([a[k].ravel().astype(float) for k in a.files])
        err = float(np.linalg.norm(g - ref) / np.linalg.norm(ref))
        cosine = float(g @ ref / (np.linalg.norm(g) * np.linalg.norm(ref)))
        record = json.loads((adjoint_path.parent / "result.json").read_text())
        fd_error = min(row["relative_error"] for row in record["finite_difference"])
        result["frequency_adjoint"] = {
            "relative_l2_error_vs_segmented": err,
            "cosine": cosine,
            "best_finite_difference_relative_error": fd_error,
            "passed": err < 0.01 and fd_error < 0.01 and cosine > 0.999,
        }
    for window in ("modal", "hann"):
        full_adjoint = root / f"gradient20_{window}_adjoint"
        full_exact = root / f"gradient20_{window}_exact"
        if (full_adjoint / "gradients.npz").exists() and (
            full_exact / "gradients.npz"
        ).exists():
            ar = json.loads((full_adjoint / "result.json").read_text())
            er = json.loads((full_exact / "result.json").read_text())
            assert ar["loaded_params_sha256"] == er["loaded_params_sha256"], (
                "Gradient candidates differ"
            )
            assert {k: v for k, v in ar["settings"].items() if k != "gradient"} == {
                k: v for k, v in er["settings"].items() if k != "gradient"
            }
            with np.load(full_exact / "gradients.npz") as a:
                ref = np.concatenate([a[k].ravel().astype(float) for k in a.files])
            with np.load(full_adjoint / "gradients.npz") as a:
                g = np.concatenate([a[k].ravel().astype(float) for k in a.files])
            err = float(np.linalg.norm(g - ref) / np.linalg.norm(ref))
            cosine = float(g @ ref / (np.linalg.norm(g) * np.linalg.norm(ref)))
            result[
                "frequency_adjoint_full"
                if window == "modal"
                else "frequency_adjoint_hann"
            ] = {
                "relative_l2_error_vs_segmented": err,
                "cosine": cosine,
                "speedup": er["median_seconds"] / ar["median_seconds"],
                "passed": err < 0.01 and cosine > 0.999,
            }
    write_json(root / "comparisons.json", result)
    print(json.dumps(result, indent=2))
    if not result or not all(
        row["passed"]
        for row in [
            result.get("cpu_gpu", {"passed": True}),
            result.get("gradients", {}).get("exact", {"passed": True}),
        ]
    ):
        raise SystemExit("Exact baseline validation failed or no evidence found")


if __name__ == "__main__":
    main()
