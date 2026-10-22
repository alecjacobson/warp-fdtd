"""Validate and benchmark the Warp adjoint against saved FDTDX derivatives."""

import argparse
import hashlib
import subprocess
import warp as wp
import json
from pathlib import Path
import time
import numpy as np
from .adjoint import Adjoint
from .detectors import attach, measurements
from .material import Material


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reps", type=int, default=2)
    args = p.parse_args()
    if args.reps < 1:
        p.error("--reps must be positive")
    out = Path("results/warp/gradient20")
    out.mkdir(parents=True, exist_ok=True)
    s = Adjoint("results/warp_reference/gradient20_highest_exact")
    attach(s)
    material = Material(s.data["parameters/p0"].squeeze(-1), s.meta["beta"])
    projected = material.inv_eps.numpy()
    reference = s.data["inv_permittivity"][0, 100:180, 20:100, 30]
    material_error = float(np.max(np.abs(projected - reference)))
    print("material max error", material_error, flush=True)
    np.testing.assert_allclose(projected, reference, atol=2e-5, rtol=2e-5)
    compilation = s.compile()
    s.reset()
    s.run()
    s.prepare_sources()
    s.compile_reverse()
    times = []
    forward_times = []
    backward_times = []
    # Captured source buffers are updated in place on every evaluation.
    for rep in range(args.reps + 1):
        s.reset()
        s.reset_reverse()
        start = time.perf_counter()
        s.run()
        forward_end = time.perf_counter()
        s.prepare_sources()
        s.backward()
        backward_end = time.perf_counter()
        voxel = s.grad.numpy().reshape(80, 80, 20).sum(axis=2)
        gradient = material.pullback(voxel)
        elapsed = time.perf_counter() - start
        print("gradient run", rep, elapsed, flush=True)
        if rep:
            times.append(elapsed)
            forward_times.append(forward_end - start)
            backward_times.append(backward_end - forward_end)
    reference_gradient = Path("results/gradient20_highest_exact/gradients.npz")
    if (
        hashlib.sha256(reference_gradient.read_bytes()).hexdigest()
        != s.meta["reference_gradients_sha256"]
    ):
        raise ValueError("Gradient reference checksum mismatch")
    ref = np.load(reference_gradient)["g0"].squeeze(-1)
    a = gradient.ravel().astype(float)
    b = ref.ravel().astype(float)
    error = float(np.linalg.norm(a - b) / np.linalg.norm(b))
    cosine = float(np.dot(a, b) / np.linalg.norm(a) / np.linalg.norm(b))
    exact = json.loads(
        Path("results/gradient20_highest_exact/result.json").read_text()
    )["median_seconds"]
    fast = json.loads(
        Path("results/gradient20_highest_adjoint/result.json").read_text()
    )["median_seconds"]
    result = dict(
        status="gradient_checked",
        git_commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        code_sha256=hashlib.sha256(
            b"".join(p.read_bytes() for p in sorted(Path("warp_fdtd").glob("*.py")))
        ).hexdigest(),
        inputs_sha256=s.meta["inputs_sha256"],
        warp_version=wp.__version__,
        gpu=s.device.name,
        material_filter_precision="float32; FDTDX highest matmul precision",
        seconds=times,
        forward_seconds=forward_times,
        backward_seconds=backward_times,
        median_seconds=float(np.median(times)),
        compile_seconds=compilation,
        relative_l2=error,
        cosine=cosine,
        material_max_absolute_error=material_error,
        history_bytes=s.history.size * 4,
        exact_fdtdx_seconds=exact,
        fast_fdtdx_seconds=fast,
        speedup_vs_exact=exact / np.median(times),
        speedup_vs_fast=fast / np.median(times),
        measurements=measurements(s),
        passed=bool(np.isfinite(a).all() and error < 0.01 and cosine > 0.999),
        scope="Exact discrete transpose and design-volume history; cotangents refreshed each evaluation; material pullback and gradient transfer included",
    )
    np.savez_compressed(out / "gradient.npz", gradient=gradient, voxel_gradient=voxel)
    (out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    assert result["passed"], result
    direction = gradient / np.linalg.norm(gradient)
    analytic = float(np.sum(gradient.astype(float) * direction))
    differences = []
    for h in (0.03, 0.01, 0.003):
        values = []
        for sign in (1, -1):
            perturbed = Material(
                s.data["parameters/p0"].squeeze(-1) + sign * h * direction,
                s.meta["beta"],
            )
            eps = s.data["inv_permittivity"].copy()
            eps[0, 100:180, 20:100, 30:50] = perturbed.inv_eps.numpy()[:, :, None]
            s.eps.assign(eps.ravel())
            s.reset()
            s.run()
            values.append(-measurements(s)["modal_transmission"])
        finite = (values[0] - values[1]) / (2 * h)
        row = dict(
            step=h,
            analytic=analytic,
            finite_difference=finite,
            relative_error=abs(finite - analytic) / abs(analytic),
        )
        print("finite difference", json.dumps(row), flush=True)
        differences.append(row)
    result["finite_difference"] = differences
    result["finite_difference_direction"] = (
        "Normalized computed gradient; full vector separately compared to independent FDTDX exact gradient"
    )
    result["passed"] = (
        result["passed"] and min(r["relative_error"] for r in differences) < 0.01
    )
    changed = np.clip(s.data["parameters/p0"].squeeze(-1) - 0.2 * direction, 0, 1)
    old_ptrs = [v.ptr for m, v in s.adjoint_sources]
    changed_loss, changed_grad = s.differentiate(changed)
    assert [v.ptr for m, v in s.adjoint_sources] == old_ptrs
    test_direction = changed_grad / np.linalg.norm(changed_grad)
    test_analytic = float(np.sum(changed_grad.astype(float) * test_direction))
    changed_values = []
    for sign in (1, -1):
        s.set_design(changed + sign * 0.01 * test_direction)
        s.reset()
        s.run()
        changed_values.append(-measurements(s)["modal_transmission"])
    test_fd = (changed_values[0] - changed_values[1]) / 0.02
    changed_error = abs(test_fd - test_analytic) / abs(test_analytic)
    result["changed_design"] = {
        "loss": changed_loss,
        "analytic": test_analytic,
        "finite_difference": test_fd,
        "relative_error": changed_error,
        "source_buffer_pointers_preserved": True,
    }
    print("changed design", json.dumps(result["changed_design"]), flush=True)
    result["passed"] = result["passed"] and changed_error < 0.01
    result["status"] = "complete"
    (out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    assert result["passed"], result


if __name__ == "__main__":
    main()
