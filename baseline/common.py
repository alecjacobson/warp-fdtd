from __future__ import annotations

import json
import hashlib
import os
import platform
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import jax
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM_SHA = "2cab8dbb2af9a43ec6443712f011f50ecf58790b"


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def environment():
    def command(args):
        return subprocess.run(args, capture_output=True, text=True).stdout.strip()

    sha = command(["git", "-C", str(ROOT / "upstream/fdtdx"), "rev-parse", "HEAD"])
    dirty = command(
        ["git", "-C", str(ROOT / "upstream/fdtdx"), "status", "--porcelain"]
    )
    if sha != UPSTREAM_SHA or dirty:
        raise RuntimeError(
            f"Expected clean upstream {UPSTREAM_SHA}, found {sha}, dirty={bool(dirty)}"
        )
    return {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "upstream_sha": sha,
        "harness_sha256": {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted((ROOT / "baseline").glob("*.py"))
        },
        "python": platform.python_version(),
        "jax": jax.__version__,
        "jax_enable_x64": jax.config.read("jax_enable_x64"),
        "requirements_sha256": hashlib.sha256(
            (ROOT / "requirements.lock").read_bytes()
        ).hexdigest(),
        "cpu_model": next(
            line.split(":", 1)[1].strip()
            for line in Path("/proc/cpuinfo").read_text().splitlines()
            if line.startswith("model name")
        ),
        "devices": [
            {"kind": d.device_kind, "platform": d.platform} for d in jax.devices()
        ],
        "nvidia_smi": command(["nvidia-smi", "-L"]),
        "driver": command(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]
        ),
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "cpu_quota": Path("/sys/fs/cgroup/cpu.max").read_text().strip(),
        "host_memory_limit": Path("/sys/fs/cgroup/memory.max").read_text().strip(),
        "environment": {
            k: v
            for k, v in os.environ.items()
            if k.startswith(("JAX_", "XLA_", "OMP_", "OPENBLAS_"))
        },
    }


def timed(fn, *args):
    start = time.perf_counter()
    result = fn(*args)
    jax.block_until_ready(result)
    return result, time.perf_counter() - start


def memory():
    return jax.devices()[0].memory_stats()


def finite_tree(tree):
    # Reduce on the device so validation doesn't copy full 3D fields to the host.
    return all(
        bool(jax.numpy.all(jax.numpy.isfinite(x)))
        for x in jax.tree.leaves(tree)
        if hasattr(x, "dtype") and np.issubdtype(x.dtype, np.inexact)
    )
