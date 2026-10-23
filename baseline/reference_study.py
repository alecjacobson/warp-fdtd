"""Generate controlled physical bend references and Warp volume observations.

Stages execute serially on the GPU. FDTDX reference-only evaluations are not
throughput benchmarks. Successful stages are reused after contract checks;
failed setup/evaluations are rerun without touching archived legacy results.
"""

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path

CANDIDATE = Path("results/ring20_optimized/best_params.npz")
CASES = {
    "mesh20": (20.0, 10, 300.0, 0.99, 4.0, False),
    "mesh10": (10.0, 20, 300.0, 0.99, 4.0, False),
    "mesh6p67": (20 / 3, 30, 300.0, 0.99, 4.0, False),
    "time20_450": (20.0, 10, 450.0, 0.99, 4.0, False),
    "cfl20_half": (20.0, 10, 300.0, 0.495, 4.0, False),
    "pml20_400": (20.0, 20, 300.0, 0.99, 4.0, True),
    "domain20_4p4": (20.0, 10, 300.0, 0.99, 4.4, False),
}

# Expensive additional level is opt-in, so reproducing the initial study stays bounded.
DEFAULT_CASES = tuple(CASES)
CASES["mesh5"] = (5.0, 40, 300.0, 0.99, 4.0, False)


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def bundle_ready(bundle, result):
    manifest = bundle / "manifest.json"
    inputs = bundle / "inputs.npz"
    if not manifest.exists() or not inputs.exists():
        return False
    m = json.loads(manifest.read_text())
    return m["reference_result_sha256"] == digest(result) and m[
        "inputs_sha256"
    ] == digest(inputs)


def volume_ready(volume, bundle):
    path = volume / "result.json"
    if not path.exists():
        return False
    r = json.loads(path.read_text())
    m = json.loads((bundle / "manifest.json").read_text())
    if (
        r.get("status") != "complete"
        or not r.get("incident_normalization")
        or r.get("inputs_sha256") != m["inputs_sha256"]
        or r.get("physical_contract") != m.get("physical_contract")
    ):
        return False
    return bool(r.get("files")) and all(
        (volume / f["file"]).exists() and digest(volume / f["file"]) == f["sha256"]
        for f in r["files"]
    )


def run(command, log, env):
    print("RUN", " ".join(command), flush=True)
    with log.open("w") as stream:
        subprocess.run(
            command, stdout=stream, stderr=subprocess.STDOUT, env=env, check=True
        )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cases", nargs="+", choices=CASES, default=list(DEFAULT_CASES))
    p.add_argument("--root", type=Path, default=Path("results/physical_reference"))
    p.add_argument("--without-volumes", action="store_true")
    p.add_argument(
        "--jax-memory-fraction",
        type=float,
        help="Explicit JAX GPU allocator fraction for a capacity study",
    )
    args = p.parse_args()
    if args.jax_memory_fraction is not None and (
        not math.isfinite(args.jax_memory_fraction)
        or not 0 < args.jax_memory_fraction <= 1
    ):
        p.error("--jax-memory-fraction must be finite and in (0, 1]")
    args.root.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(
        JAX_PLATFORMS="cuda,cpu",
        XLA_PYTHON_CLIENT_PREALLOCATE="false",
        JAX_DEFAULT_MATMUL_PRECISION="highest",
        OPENBLAS_NUM_THREADS="1",
        OMP_NUM_THREADS="4",
    )
    if args.jax_memory_fraction is not None:
        env.pop("XLA_PYTHON_CLIENT_MEM_FRACTION", None)
        env["XLA_CLIENT_MEM_FRACTION"] = str(args.jax_memory_fraction)
    candidate_hash = hashlib.sha256(CANDIDATE.read_bytes()).hexdigest()
    for name in args.cases:
        dx, pml, duration, cfl, domain, padding = CASES[name]
        directory = args.root / name
        directory.mkdir(exist_ok=True)
        result = directory / "result.json"
        if result.exists() and json.loads(result.read_text())["status"] == "complete":
            record = json.loads(result.read_text())
            s = record["settings"]
            expected = dict(
                spacing_nm=dx,
                pml_cells=pml,
                time_fs=duration,
                courant_factor=cfl,
                domain_um=domain,
                pad_domain=padding,
                reference_protocol="fixed_physical_v1",
                voxel_nm=20,
                modal_window_fs=40.0,
                modal_window="hann",
                parameterization="smooth",
                precision="float32",
                source_profile="current",
                quality_monitors=True,
                exact_interpolation=True,
            )
            if (
                any(s[k] != v for k, v in expected.items())
                or record["loaded_params_sha256"] != candidate_hash
                or record["arguments"]["beta"] != 50
            ):
                raise ValueError(
                    f"Existing {directory} does not match the declared study"
                )
        if (
            not result.exists()
            or json.loads(result.read_text())["status"] != "complete"
            or not (directory / "arrays.npz").exists()
        ):
            command = [
                sys.executable,
                "-m",
                "baseline.run",
                "forward",
                "--output",
                str(directory),
                "--spacing-nm",
                str(dx),
                "--pml-cells",
                str(pml),
                "--time-fs",
                str(duration),
                "--courant-factor",
                str(cfl),
                "--domain-um",
                str(domain),
                "--reference-protocol",
                "fixed_physical_v1",
                "--quality-monitors",
                "--modal-window",
                "hann",
                "--modal-window-fs",
                "40",
                "--exact-interpolation",
                "--load",
                str(CANDIDATE),
                "--beta",
                "50",
                "--reference-only",
            ]
            if padding:
                command.append("--pad-domain")
            run(command, directory / "reference.log", env)
        bundle = args.root / "bundles" / name
        if not bundle_ready(bundle, result):
            run(
                [
                    sys.executable,
                    "-m",
                    "baseline.export_case",
                    "--run",
                    str(directory),
                    "--output",
                    str(bundle),
                ],
                directory / "export.log",
                env,
            )
        if not args.without_volumes:
            volume = args.root / "volumes" / name
            if not volume_ready(volume, bundle):
                run(
                    [
                        sys.executable,
                        "-m",
                        "warp_fdtd.volume",
                        "--case",
                        str(bundle),
                        "--output",
                        str(volume),
                        "--sample-nm",
                        "20",
                    ],
                    directory / "volume.log",
                    env,
                )
        print("COMPLETE", name, flush=True)


if __name__ == "__main__":
    main()
