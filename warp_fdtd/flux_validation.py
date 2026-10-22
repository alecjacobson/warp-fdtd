"""Long-time material/CFL/metric sweep and explicit cost of the flux control.

All jobs are serial. CPU results are correctness controls, never GPU throughput
claims. The periodic domain has normalized length one in each direction.
"""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import warp as wp

from .flux_rectilinear import FluxRectilinearPeriodic
from .rectilinear import RectilinearPeriodic, curl_numpy, divergence_numpy


def owned_bytes(s):
    seen = set()

    def size(value):
        if isinstance(value, wp.array):
            if value.ptr in seen:
                return 0
            seen.add(value.ptr)
            return int(value.capacity)
        if isinstance(value, (list, tuple)):
            return sum(size(a) for a in value)
        if isinstance(value, dict):
            return sum(size(a) for a in value.values())
        return 0

    return size(vars(s))


def create(backend, edges, epsilon, cfl, device):
    if backend == "flux":
        return FluxRectilinearPeriodic(edges, epsilon, cfl, device)
    return RectilinearPeriodic(
        edges,
        epsilon,
        cfl,
        device,
        precision="float64" if backend == "float64" else "float32",
    )


def stress(steps, backend, device):
    rows = []
    n = 8
    u = np.linspace(0, 1, n + 1)
    for grading, contrast in ((0.0, 1.0), (0.45, 1.0), (0.8, 12.25), (0.8, 100.0)):
        edges = [
            u + grading * g / (2 * np.pi) * np.sin(2 * np.pi * u)
            for g in (1.0, 0.7, 0.4)
        ]
        rng = np.random.default_rng(51)
        # Sharp heterogeneous positive diagonal dielectric, seeded and fixed.
        epsilon = rng.choice([1.0, contrast], size=(3, n, n, n))
        for cfl in (0.3, 0.9, 0.99):
            s = create(backend, edges, epsilon, cfl, device)
            rng = np.random.default_rng(52)
            d = curl_numpy(rng.normal(size=epsilon.shape), s.dual, False)
            h = curl_numpy(rng.normal(size=epsilon.shape), s.primal, True)
            precision = "float64" if backend == "float64" else "float32"
            s.set_fields((d / epsilon).astype(precision), h.astype(precision))
            e0, h0 = s.fields()
            energy = s.energy(e0, h0)
            count = max(1, steps // 10)
            s.compile((min(count, 256), count % 256, steps % count))
            checkpoints = []
            done = 0
            started = time.perf_counter()
            while done < steps:
                advance = min(count, steps - done)
                s.run(advance)
                done += advance
                ef, hf = s.fields()
                checkpoints.append(
                    dict(
                        step=done,
                        energy_relative_drift=abs(s.energy(ef, hf) / energy - 1),
                        gauss_d=float(
                            np.linalg.norm(
                                divergence_numpy(ef * epsilon, s.dual, False)
                            )
                            / np.linalg.norm(ef * epsilon)
                        ),
                        gauss_b=float(
                            np.linalg.norm(divergence_numpy(hf, s.primal, True))
                            / np.linalg.norm(hf)
                        ),
                    )
                )
            passed = (
                max(
                    max(r["energy_relative_drift"], r["gauss_d"], r["gauss_b"])
                    for r in checkpoints
                )
                < 1e-4
            )
            row = dict(
                grading=grading,
                contrast=contrast,
                cfl=cfl,
                minimum_primal_width=min(w.min() for w in s.primal),
                maximum_primal_width=max(w.max() for w in s.primal),
                owned_device_array_bytes=owned_bytes(s),
                seconds_including_checkpoints=time.perf_counter() - started,
                passed=passed,
                checkpoints=checkpoints,
            )
            rows.append(row)
            print(
                json.dumps({k: v for k, v in row.items() if k != "checkpoints"}),
                flush=True,
            )
    return rows


def costs(device, sizes, reps, steps):
    if not wp.get_device(device).is_cuda:
        raise ValueError("GPU cost measurements require a CUDA device")
    rows = []
    for n in sizes:
        u = np.linspace(0, 1, n + 1)
        edges = [u + 0.45 / (2 * np.pi) * np.sin(2 * np.pi * u)] * 3
        for backend in ("float32", "float64", "flux"):
            s = create(backend, edges, 1.0, 0.9, device)
            rng = np.random.default_rng(51)
            e, h = [rng.normal(size=(3, n, n, n)).astype("f") for _ in range(2)]
            s.set_fields(e, h)
            started = time.perf_counter()
            s.compile((min(steps, 256), steps % 256))
            compiled = time.perf_counter() - started
            s.run(steps)
            samples = []
            for _ in range(reps):
                s.set_fields(e, h)
                wp.synchronize_device(s.device)
                started = time.perf_counter()
                s.run(steps)
                samples.append(time.perf_counter() - started)
            row = dict(
                backend=backend,
                n=n,
                cells=n**3,
                steps=steps,
                owned_device_array_bytes=owned_bytes(s),
                compile_seconds=compiled,
                warm_seconds=samples,
                median_seconds=float(np.median(samples)),
            )
            rows.append(row)
            print(json.dumps(row), flush=True)
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=Path("results/warp/flux_sweep"))
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--backend", choices=["flux", "float32", "float64"], default="flux")
    p.add_argument("--long-steps", type=int, default=1000000)
    p.add_argument("--cost-only", action="store_true")
    p.add_argument("--sizes", type=int, nargs="+", default=[64, 128])
    p.add_argument("--reps", type=int, default=5)
    p.add_argument("--cost-steps", type=int, default=1024)
    args = p.parse_args()
    if (
        args.long_steps < 1
        or args.cost_steps < 1
        or args.reps < 5
        or min(args.sizes) < 4
    ):
        p.error("Positive step counts, >=5 repetitions and sizes >=4 required")
    if args.device == "cpu" and args.backend != "flux":
        p.error("Only the flux correctness control supports CPU execution")
    rows = (
        costs(args.device, args.sizes, args.reps, args.cost_steps)
        if args.cost_only
        else stress(args.long_steps, args.backend, args.device)
    )
    result = dict(
        status="complete",
        backend=args.backend,
        device=wp.get_device(args.device).name,
        warp_version=wp.__version__,
        cost_only=args.cost_only,
        scope="Periodic rectilinear control; no flagship, CPML, interface, source or adjoint acceptance.",
        precision="Two-float flux accumulators; float32 metric coefficients and edge impulses"
        if args.backend == "flux"
        else args.backend,
        memory_scope="Owned device arrays, excluding graph/driver/allocator overhead; not peak memory.",
        long_steps=args.long_steps,
        threshold=1e-4,
        passed=None if args.cost_only else all(r["passed"] for r in rows),
        code_sha256={
            n: hashlib.sha256(Path("warp_fdtd", n).read_bytes()).hexdigest()
            for n in ["rectilinear.py", "flux_rectilinear.py", "flux_validation.py"]
        },
        rows=rows,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "result.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    if result["passed"] is False:
        raise AssertionError("Flux stress sweep failed; see result.json")


if __name__ == "__main__":
    main()
