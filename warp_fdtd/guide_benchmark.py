"""Serial process-per-case 3D octree/uniform comparison; no inferred speedups."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import resource
import subprocess
import time
import numpy as np
import warp as wp
from .guide_problem import GuideProblem
from .driven_uniform import DrivenUniform
from .driven_blocks import DrivenBlocks
from .block_operator import BlockOperator
from .guide_observer import GuideObserver
from .guide_validation import assert_finite


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--kind", choices=["uniform", "octree"], required=True)
    ap.add_argument("--spacing", type=float, required=True)
    ap.add_argument("--levels", type=int, default=2)
    ap.add_argument("--block-cells", type=int, default=4)
    ap.add_argument("--coupling", choices=["factorized", "fused"], default="factorized")
    ap.add_argument("--buffer", type=float, default=0.1)
    ap.add_argument("--width", type=float, default=3.2)
    ap.add_argument("--cfl", type=float, default=0.7)
    ap.add_argument("--duration", type=float, default=14.0)
    ap.add_argument("--normal-average", action="store_true")
    ap.add_argument("--uniform-layout", choices=["xyz", "flat"], default="xyz")
    ap.add_argument("--reference-only", action="store_true")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if (
        not (0 < args.cfl <= 0.7)
        or args.duration <= 0
        or args.spacing <= 0
        or args.repeats < 1
    ):
        ap.error("Require 0 < cfl <= .7 and positive duration, spacing, repeats")
    args.output.mkdir(parents=True, exist_ok=True)
    p = GuideProblem(width=args.width, height=args.width, duration=args.duration)
    max_dt = args.cfl * args.spacing * np.sqrt(p.clad_epsilon / 3)
    steps = 7 * math.ceil(p.duration / max_dt / 7)
    cfl = (p.duration / steps) / (args.spacing * np.sqrt(p.clad_epsilon / 3))
    source_files = [
        "block_operator.py",
        "block_mesh.py",
        "driven_blocks.py",
        "driven_uniform.py",
        "guide_problem.py",
        "guide_observer.py",
        "guide_benchmark.py",
        "guide_validation.py",
    ]
    hashes = {
        f: hashlib.sha256(Path(__file__).with_name(f).read_bytes()).hexdigest()
        for f in source_files
    }
    wp.init()
    device = wp.get_device("cuda:0")
    start = time.perf_counter()
    metadata = {}
    if args.kind == "uniform":
        s = DrivenUniform(
            p,
            args.spacing,
            steps,
            cfl,
            layout=args.uniform_layout,
            normal_average=args.normal_average,
        )
        metadata = {
            "uniform_layout": args.uniform_layout,
            "normal_average": args.normal_average,
        }
        cells = int(np.prod(s.shape))
    else:
        mesh = p.mesh(args.spacing, args.levels, args.block_cells, args.buffer)
        op = BlockOperator(mesh, epsilon=p, trace="linear")
        s = DrivenBlocks(op, p, steps, cfl, coupling=args.coupling)
        cells = len(mesh.leaves) * mesh.cells_per_block**3
        payload = mesh.payload()
        leaves = payload.pop("leaves")
        (args.output / "mesh.json").write_text(
            json.dumps(payload, indent=2)[:-2]
            + ',\n  "leaves": [\n'
            + ",\n".join("    " + json.dumps(v) for v in leaves)
            + "\n  ]\n}\n"
        )
        metadata = {
            "mesh": mesh.accounting(),
            "trace": "linear",
            "coupling": args.coupling,
            "gather_nnz": op.gather_e.nnz,
            "prolongation_nnz": op.p.nnz,
        }
    setup = time.perf_counter() - start
    start = time.perf_counter()
    s.compile()
    compile_time = time.perf_counter() - start
    print(
        f"Built {args.kind}: {cells:,} cells, {steps} steps; setup {setup:.3f}s",
        flush=True,
    )
    s.reset()
    if not args.reference_only:
        s.run(steps)  # one complete untimed warm trajectory
    samples = []
    for _ in range(0 if args.reference_only else args.repeats):
        s.reset()
        wp.synchronize_device(device)
        start = time.perf_counter()
        s.run(steps)
        samples.append(time.perf_counter() - start)
        print(f"trajectory: {samples[-1]:.6f}s", flush=True)
    solver_peak = wp.get_mempool_used_mem_high(device)
    start = time.perf_counter()
    observer = GuideObserver(s, p)
    observer_setup = time.perf_counter() - start
    s.reset()
    snapshots = []
    previous = 0
    checkpoints = [3 * steps // 7, 5 * steps // 7, steps]
    start = time.perf_counter()
    for checkpoint in checkpoints:
        s.run(checkpoint - previous)
        snapshots.append(observer.sample())
        previous = checkpoint
    diagnostic_time = time.perf_counter() - start
    validation_start = time.perf_counter()
    assert_finite(s)
    metadata["full_field_finite"] = True
    metadata["full_field_validation_s"] = time.perf_counter() - validation_start
    if args.kind == "octree":
        final_h = s.h.numpy().astype(np.float64)
        residual = op.normal_flux @ final_h
        scale = abs(op.normal_flux) @ abs(final_h)
        metadata["normal_flux_relative_l2"] = float(
            np.linalg.norm(residual) / max(np.linalg.norm(scale), 1e-30)
        )
        metadata["operator_audit"] = op.audit()
    snapshots = np.array(snapshots)
    if not np.all(np.isfinite(snapshots)):
        raise RuntimeError("Nonfinite field values")
    np.savez_compressed(
        args.output / "fields.npz",
        fields=snapshots,
        xyz=observer.xyz,
        x=observer.axes[0],
        y=observer.axes[1],
        z=observer.axes[2],
        times=np.array(checkpoints) * s.dt,
    )
    result = dict(
        kind=args.kind,
        spacing=args.spacing,
        levels=args.levels,
        buffer=args.buffer,
        block_cells=args.block_cells,
        cells=cells,
        problem=p.parameters(),
        steps=steps,
        dt=s.dt,
        dtype="float32",
        device=device.name,
        warp_version=wp.__version__,
        commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        source_sha256=hashes,
        setup_s=setup,
        compile_capture_s=compile_time,
        trajectory_s=samples,
        median_s=float(np.median(samples)) if samples else None,
        reference_only=args.reference_only,
        owned_bytes=s.owned_bytes,
        solver_pool_peak_bytes=solver_peak,
        diagnostics_pool_peak_bytes=wp.get_mempool_used_mem_high(device),
        host_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        observer_setup_s=observer_setup,
        diagnostic_trajectory_s=diagnostic_time,
        field_max=float(np.max(np.abs(snapshots))),
        scope="Pulsed impressed current, matched conductivity sponge, PEC exterior; no modal T or open-boundary qualification",
        **metadata,
    )
    result["fields_sha256"] = hashlib.sha256(
        (args.output / "fields.npz").read_bytes()
    ).hexdigest()
    result["dirty_worktree"] = bool(
        subprocess.check_output(["git", "status", "--porcelain"], text=True)
    )
    (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: result[k]
                for k in [
                    "kind",
                    "spacing",
                    "cells",
                    "steps",
                    "median_s",
                    "owned_bytes",
                    "setup_s",
                    "field_max",
                ]
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
