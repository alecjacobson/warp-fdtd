"""Summarize the uniform full/packed CPML experiment, including negative timings."""

import hashlib
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    root = Path("results/warp")
    rows = []
    for spacing in (20, 10):
        paths = [
            root / f"packed_control{spacing}_{kind}" for kind in ("full", "packed")
        ]
        a, b = [json.loads((p / "result.json").read_text()) for p in paths]
        assert a["passed"] and b["passed"]
        for key in ("inputs_sha256", "gpu", "shape", "steps", "dt_s"):
            assert a[key] == b[key], key
        parity = []
        for full in sorted(paths[0].glob("*.npz")):
            packed = paths[1] / full.name
            with np.load(full) as x, np.load(packed) as y:
                assert set(x.files) == set(y.files)
                equality = {
                    name: bool(np.array_equal(x[name], y[name])) for name in x.files
                }
            parity.append(
                dict(
                    file=full.name,
                    arrays_bitwise_equal=equality,
                    full_sha256=hashlib.sha256(full.read_bytes()).hexdigest(),
                    packed_sha256=hashlib.sha256(packed.read_bytes()).hexdigest(),
                )
            )
        assert parity and all(all(p["arrays_bitwise_equal"].values()) for p in parity)
        rows.append(
            dict(
                spacing_nm=spacing,
                shape=a["shape"],
                full_seconds=a["median_seconds"],
                packed_seconds=b["median_seconds"],
                full_owned_bytes=a["owned_device_array_bytes"],
                packed_owned_bytes=b["owned_device_array_bytes"],
                packed_speedup=a["median_seconds"] / b["median_seconds"],
                owned_storage_reduction=a["owned_device_array_bytes"]
                / b["owned_device_array_bytes"],
                parity_scope="Saved mid-z E/H slices and full modal planes, not full-volume flagship fields",
                saved_array_parity=parity,
            )
        )
    out = root / "storage_control"
    out.mkdir(exist_ok=True)
    record = dict(
        status="complete",
        scope="Uniform storage tradeoff, not adaptivity; owned-array bytes are not device peak memory",
        rows=rows,
    )
    (out / "result.json").write_text(json.dumps(record, indent=2) + "\n")
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), layout="constrained")
    x = np.arange(len(rows))
    for shift, kind, label in [
        (-0.18, "full", "Full history"),
        (0.18, "packed", "Packed history"),
    ]:
        axes[0].bar(x + shift, [r[kind + "_seconds"] for r in rows], 0.36, label=label)
        axes[1].bar(
            x + shift, [r[kind + "_owned_bytes"] / 1e9 for r in rows], 0.36, label=label
        )
    axes[0].set(ylabel="Median trajectory time (seconds)", yscale="log")
    axes[1].set(ylabel="Owned solver/detector arrays (GB)", yscale="log")
    for ax in axes:
        ax.set_xticks(x, ["3M / 20 nm", "24M / 10 nm"])
        ax.legend()
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle("Same 3D bend inputs: compact CPML trades runtime for storage")
    fig.savefig(out / "comparison.png", dpi=150)
    plt.close(fig)
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
