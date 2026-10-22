"""Synchronized full-trajectory benchmarks against immutable FDTDX inputs."""

import argparse
import hashlib
import platform
import json
from pathlib import Path
import subprocess
import time
import numpy as np
import warp as wp
from .solver import Solver, owned_array_bytes
from .detectors import attach, measurements


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--case", default="results/warp_reference/forward50_gpu")
    p.add_argument("--output", default="results/warp/forward50_gpu")
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--cpml-storage", choices=("full", "packed"), default="full")
    args = p.parse_args()
    if args.reps < 1:
        p.error("--reps must be positive")
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    s = Solver(args.case, cpml_storage=args.cpml_storage)
    setup = time.perf_counter() - start
    if s.meta["settings"].get("gradient", "none") != "none":
        raise ValueError("Forward benchmarks require forward-only reference timing")
    attach(s)
    compile_seconds = s.compile()
    times = []
    for rep in range(args.reps + 1):
        s.reset()
        start = time.perf_counter()
        s.run()
        elapsed = time.perf_counter() - start
        print(f"run {rep}: {elapsed:.6f}s", flush=True)
        if rep == 0:
            warm = elapsed
        else:
            times.append(elapsed)
    fields = s.fields()
    refpath = (
        Path("results") / Path(s.meta["reference_arrays"]).parent.name / "arrays.npz"
    )
    if (
        hashlib.sha256(refpath.read_bytes()).hexdigest()
        != s.meta["reference_arrays_sha256"]
    ):
        raise ValueError("Reference field checksum mismatch")
    with np.load(refpath) as ref:
        errors = {}
        for name in ("E", "H"):
            a = fields[name]
            if ref[name].ndim == 3:
                a = a[..., a.shape[-1] // 2]
            errors[name] = float(
                np.linalg.norm((a - ref[name]).ravel().astype(float))
                / np.linalg.norm(ref[name].ravel().astype(float))
            )
    finite = all(np.isfinite(a).all() for a in fields.values())
    measured = measurements(s)
    if measured:
        errors["modal_transmission_absolute"] = abs(
            measured["modal_transmission"]
            - s.meta["reference_measurements"]["modal_transmission"]
        )
    for mon in s.monitors:
        if mon.is_phasor:
            np.savez_compressed(
                out / (mon.name + "_phasors.npz"), phasors=mon.phasors()
            )
        else:
            with np.load(refpath) as ref:
                a = mon.flux.numpy()
                b = ref[mon.name].ravel()
                errors[mon.name + "_flux"] = float(
                    np.linalg.norm(a - b) / np.linalg.norm(b)
                )
    median = float(np.median(times))
    result = dict(
        measurements=measured,
        field_reference_scope=s.meta["field_reference_scope"],
        code_sha256=hashlib.sha256(
            b"".join(p.read_bytes() for p in sorted(Path("warp_fdtd").glob("*.py")))
        ).hexdigest(),
        python=platform.python_version(),
        numpy=np.__version__,
        timing_scope="Synchronized complete trajectory, source and all exported detectors; reset/compilation/host postprocessing excluded",
        status="complete",
        cpml_storage=s.cpml_storage,
        owned_device_array_bytes=owned_array_bytes(s),
        memory_scope="Live solver and detector array allocations; excludes graph/allocator/driver overhead; not measured peak memory",
        case=args.case,
        inputs_sha256=s.meta["inputs_sha256"],
        shape=s.shape,
        steps=s.steps,
        dt_s=s.meta["dt_s"],
        warp_version=wp.__version__,
        gpu=s.device.name,
        git_commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        setup_seconds=setup,
        compile_graph_seconds=compile_seconds,
        warmup_seconds=warm,
        seconds=times,
        median_seconds=median,
        gcups=s.n * s.steps / median / 1e9,
        reference_seconds=s.meta["baseline_median_seconds"],
        speedup=s.meta["baseline_median_seconds"] / median,
        relative_l2=errors,
        finite_fields=finite,
        passed=finite and max(errors.values()) < 1e-3,
    )
    (out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    np.savez_compressed(
        out / "fields.npz",
        **{
            n: a if a.size < 1e6 else a[..., a.shape[-1] // 2]
            for n, a in fields.items()
        },
    )
    print(json.dumps(result, indent=2), flush=True)
    assert result["passed"], result


if __name__ == "__main__":
    main()
