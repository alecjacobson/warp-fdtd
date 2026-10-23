"""Reproducible topology counts and mesh slices, explicitly not solver timings."""

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
import numpy as np

from .block_mesh import BlockMesh, Box, GuidePolicy


def run_case(ratio=2, block_cells=8, buffer=0.15, domain_scale=1):
    # Units are micrometers. Same guide cross section for all domain sizes.
    root_width = 0.1 * block_cells
    shape = tuple(int(round(v * domain_scale / root_width)) for v in (4.8, 3.2, 3.2))
    levels = 3 if ratio == 2 else 2
    h = root_width / block_cells / ratio**levels
    center = np.array(shape) * root_width / 2
    box = Box(
        (0.0, center[1] - 0.2, center[2] - 0.2),
        (shape[0] * root_width, center[1] + 0.2, center[2] + 0.2),
    )
    policy = GuidePolicy(
        (box,), h, 0.05, 0.1, boundary_band=h, guide_buffer=buffer, growth=0.5
    )
    mesh = BlockMesh(
        shape, root_width, cells_per_block=block_cells, ratio=ratio, max_level=levels
    )
    start = time.perf_counter()
    mesh.generate(policy)
    generated = time.perf_counter()
    audit = mesh.audit()
    audited = time.perf_counter()
    faces = mesh.face_connections()
    leaves = np.array(
        [(b.level, *b.index) for b in sorted(mesh.leaves)], dtype=np.int64
    )
    return mesh, {
        "ratio": ratio,
        "block_cells": block_cells,
        "guide_buffer_um": buffer,
        "domain_scale": domain_scale,
        "finest_cell_um": h,
        "domain_um": (np.array(shape) * root_width).tolist(),
        "policy": asdict(policy),
        "audit": audit,
        **mesh.accounting(),
        "generation_balance_seconds": generated - start,
        "audit_seconds": audited - generated,
        "exported_int64_topology_bytes": leaves.nbytes + faces.nbytes,
        "topology_sha256": hashlib.sha256(
            leaves.tobytes() + faces.tobytes()
        ).hexdigest(),
    }


def plot_mesh(mesh, path):
    extent = np.array(mesh.root_shape) * mesh.root_width
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    collection = None
    for ax, plane in zip(axes, ("xy", "yz")):
        fixed, a, b = (2, 0, 1) if plane == "xy" else (0, 1, 2)
        cut = extent[fixed] / 2 + 1e-10
        polygons, widths = [], []
        for block in sorted(mesh.leaves):
            lo, hi = mesh.bounds(block)
            if lo[fixed] <= cut < hi[fixed]:
                polygons.append(
                    [(lo[a], lo[b]), (hi[a], lo[b]), (hi[a], hi[b]), (lo[a], hi[b])]
                )
                widths.append(mesh.cell_width(block) * 1000)
        collection = PolyCollection(
            polygons,
            array=np.array(widths),
            cmap="viridis",
            edgecolors="black",
            linewidths=0.35,
        )
        ax.add_collection(collection)
        ax.set(
            xlim=(0, extent[a]),
            ylim=(0, extent[b]),
            aspect="equal",
            xlabel="xyz"[a] + " (µm)",
            ylabel="xyz"[b] + " (µm)",
            title=plane + " slice; outlines are blocks",
        )
        if plane == "xy":
            for y in (extent[1] / 2 - 0.2, extent[1] / 2 + 0.2):
                ax.axhline(y, color="red", linestyle="--", linewidth=1)
        else:
            from matplotlib.patches import Rectangle

            ax.add_patch(
                Rectangle(
                    (extent[1] / 2 - 0.2, extent[2] / 2 - 0.2),
                    0.4,
                    0.4,
                    fill=False,
                    edgecolor="red",
                    linewidth=1.5,
                )
            )
    fig.colorbar(collection, ax=axes, label="Cell width (nm)", shrink=0.8)
    fig.suptitle("Procedural 3D guide mesh — geometry only, no wave solution")
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("results/warp/geometry_mesh")
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    settings = [
        (2, 4, 0.15, 1),
        (2, 8, 0.05, 1),
        (2, 8, 0.15, 1),
        (2, 16, 0.15, 1),
        (2, 8, 0.15, 2),
        (3, 8, 0.15, 1),
    ]
    results = []
    for ratio, block, buffer, scale in settings:
        mesh, result = run_case(ratio, block, buffer, scale)
        results.append(result)
        print(json.dumps(result), flush=True)
        if (ratio, block, buffer, scale) == (2, 8, 0.15, 1):
            mesh.save(args.output / "mesh.json")
            ordered = np.array(
                [(b.level, *b.index) for b in sorted(mesh.leaves)], dtype=np.int64
            )
            np.savez(
                args.output / "arrays_topology.npz",
                leaves=ordered,
                faces=mesh.face_connections(),
            )
            result["topology_npz_sha256"] = hashlib.sha256(
                (args.output / "arrays_topology.npz").read_bytes()
            ).hexdigest()
            plot_mesh(mesh, args.output / "mesh.png")
    source = {
        str(p): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (Path(__file__), Path(__file__).with_name("block_mesh.py"))
    }
    # Stable repository-relative provenance.
    source = {"warp_fdtd/" + Path(p).name: h for p, h in source.items()}
    (args.output / "result.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "scope": "Topology study only; no Maxwell evolution or measured speedup",
                "source_sha256": source,
                "cases": results,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
