"""Report measured cost versus sampled transient field error, including failures."""

import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from .guide_problem import GuideProblem


def relative(a, b, weights):
    return float(np.sqrt(np.sum(weights * (a - b) ** 2) / np.sum(weights * b**2)))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("directory", type=Path)
    ap.add_argument("--reference", default="u0125")
    args = ap.parse_args()
    root = args.directory
    cases = {
        p.parent.name: json.loads(p.read_text())
        for p in sorted(root.glob("*/result.json"))
    }
    data = {k: np.load(root / k / "fields.npz") for k in cases}
    ref = data[args.reference]
    problem = GuideProblem(**cases[args.reference]["problem"])
    xyz = ref["xyz"]
    weights = np.concatenate(
        [np.tile(problem.sample(xyz, 1e-8), (3, 1)), np.ones((3, len(xyz)))]
    )
    core = (np.abs(xyz[:, 1] - problem.width / 2) < problem.core_width / 2) & (
        np.abs(xyz[:, 2] - problem.height / 2) < problem.core_height / 2
    )
    for k, c in cases.items():
        if c["problem"] != problem.parameters():
            raise ValueError("Different physical problems cannot share a comparison")
        np.testing.assert_allclose(data[k]["xyz"], xyz, rtol=0, atol=1e-12)
        np.testing.assert_allclose(data[k]["times"], ref["times"], rtol=0, atol=1e-12)
        c["sampled_field_l2"] = relative(data[k]["fields"], ref["fields"], weights)
        c["core_field_l2"] = relative(
            data[k]["fields"][:, :, core], ref["fields"][:, :, core], weights[:, core]
        )
        c["cladding_field_l2"] = relative(
            data[k]["fields"][:, :, ~core],
            ref["fields"][:, :, ~core],
            weights[:, ~core],
        )
        c["reference_case"] = args.reference
    for k, c in cases.items():
        if c["kind"] != "octree":
            continue
        same = [
            v
            for v in cases.values()
            if v["kind"] == "uniform" and v["spacing"] == c["spacing"]
        ]
        eligible = [
            v
            for key, v in cases.items()
            if v["kind"] == "uniform"
            and v["median_s"] is not None
            and v["sampled_field_l2"] <= c["sampled_field_l2"]
        ]
        if same:
            c["same_finest_speedup"] = same[0]["median_s"] / c["median_s"]
            c["same_finest_memory_ratio"] = same[0]["owned_bytes"] / c["owned_bytes"]
        if eligible:
            best = min(eligible, key=lambda v: v["median_s"])
            c["sampled_accuracy_matched_uniform_spacing"] = best["spacing"]
            c["sampled_accuracy_matched_speedup"] = best["median_s"] / c["median_s"]
            c["sampled_accuracy_matched_memory_ratio"] = (
                best["owned_bytes"] / c["owned_bytes"]
            )
    (root / "comparison.json").write_text(json.dumps(cases, indent=2) + "\n")
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), constrained_layout=True)
    for kind, marker, color in [("uniform", "o", "C0"), ("octree", "s", "C1")]:
        values = sorted(
            [v for k, v in cases.items() if v["kind"] == kind and k != args.reference],
            key=lambda v: v["spacing"],
        )
        if not values:
            continue
        error = [100 * v["sampled_field_l2"] for v in values]
        for ax, metric, ylabel in zip(
            axes,
            ["spacing", "median_s", "owned_bytes"],
            [
                "Cell size (µm; octree minimum)",
                "Full trajectory (seconds)",
                "Owned solver GPU memory (MiB)",
            ],
        ):
            y = [v[metric] / (2**20 if metric == "owned_bytes" else 1) for v in values]
            ax.loglog(error, y, marker=marker, color=color, label=kind)
            for e, yy, v in zip(error, y, values):
                ax.annotate(
                    f"{1000 * v['spacing']:g} nm",
                    (e, yy),
                    xytext=(3, 5),
                    textcoords="offset points",
                    fontsize=8,
                )
            ax.set(xlabel="Sampled transient E/H difference (%)", ylabel=ylabel)
            ax.grid(True, alpha=0.25)
    axes[0].legend()
    fig.suptitle(
        f"3D dielectric channel • reference {1000 * cases[args.reference]['spacing']:g} nm uniform • same source and sponge\nMesh differences, not continuum bounds; setup excluded from warm timings",
        fontsize=11,
    )
    fig.savefig(root / "trends.png", dpi=170)
    plt.close(fig)
    octrees = [k for k, c in cases.items() if c["kind"] == "octree"]
    if octrees:
        chosen = min(octrees, key=lambda k: cases[k]["sampled_field_l2"])
        same = min(
            [k for k, c in cases.items() if c["kind"] == "uniform"],
            key=lambda k: abs(cases[k]["spacing"] - cases[chosen]["spacing"]),
        )
        selected = [args.reference, same, chosen]
        fig, axs = plt.subplots(2, 3, figsize=(12, 6.4), constrained_layout=True)
        x, y, z = [ref[a] for a in ["x", "y", "z"]]
        iz = int(np.argmin(abs(z - problem.height / 2)))
        fields = {
            k: data[k]["fields"][1].reshape(6, len(x), len(y), len(z))[:, :, :, iz]
            for k in selected
        }
        vmax = max(np.max(abs(v[1])) for v in fields.values())
        for col, k in enumerate(selected):
            f = fields[k]
            s = np.cross(f[:3], f[3:], axisa=0, axisb=0, axisc=0)
            im = axs[0, col].pcolormesh(
                x, y, f[1].T, cmap="RdBu_r", vmin=-vmax, vmax=vmax, shading="nearest"
            )
            axs[0, col].quiver(
                x[::3],
                y[::2],
                s[0, ::3, ::2].T,
                s[1, ::3, ::2].T,
                color="black",
                scale=0.002,
            )
            axs[0, col].set_title(f"{k}: Δ={100 * cases[k]['sampled_field_l2']:.2f}%")
            diff = f[1] - fields[args.reference][1]
            axs[1, col].pcolormesh(
                x,
                y,
                diff.T,
                cmap="RdBu_r",
                vmin=-vmax * 0.1,
                vmax=vmax * 0.1,
                shading="nearest",
            )
            for row in (0, 1):
                axs[row, col].axhline(
                    problem.width / 2 - problem.core_width / 2, color="gray", lw=0.6
                )
                axs[row, col].axhline(
                    problem.width / 2 + problem.core_width / 2, color="gray", lw=0.6
                )
                axs[row, col].set(xlabel="x (µm)", ylabel="y (µm)", aspect="equal")
        fig.colorbar(im, ax=axs[0, :], label="E_y, common scale")
        fig.suptitle(
            f"3D pulse at t={ref['times'][1]:g} µm/c, z={z[iz]:.3f} µm: E_y + instantaneous E×H arrows\nBottom: E_y difference from fine uniform, ±10% of top color scale; not time-averaged power",
            fontsize=11,
        )
        fig.savefig(root / "fields.png", dpi=170)
        plt.close(fig)
        c = cases[chosen]
        mesh = problem.mesh(c["spacing"], c["levels"], c["block_cells"], c["buffer"])
        lines = []
        for b in sorted(mesh.leaves):
            lo, hi = mesh.bounds(b)
            if lo[2] <= problem.height / 2 < hi[2]:
                for xx in np.linspace(lo[0], hi[0], mesh.cells_per_block + 1):
                    lines.append([(xx, lo[1]), (xx, hi[1])])
                for yy in np.linspace(lo[1], hi[1], mesh.cells_per_block + 1):
                    lines.append([(lo[0], yy), (hi[0], yy)])
        fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
        ax.add_collection(LineCollection(lines, lw=0.25, color="C0"))
        ax.set(
            xlim=(0, problem.length),
            ylim=(0, problem.width),
            aspect="equal",
            xlabel="x (µm)",
            ylabel="y (µm)",
            title=f"Actual binary octree slice: {c['cells']:,} active cells; {c['spacing'] * 1000:g} nm minimum",
        )
        for yy in (
            problem.width / 2 - problem.core_width / 2,
            problem.width / 2 + problem.core_width / 2,
        ):
            ax.axhline(yy, color="C3")
        fig.savefig(root / "mesh.png", dpi=170)
        plt.close(fig)
    for k, c in cases.items():
        print(
            k,
            "error %",
            round(100 * c["sampled_field_l2"], 4),
            "seconds",
            round(c["median_s"], 5) if c["median_s"] is not None else None,
            "MiB",
            round(c["owned_bytes"] / 2**20, 3),
            "matched speedup",
            c.get("sampled_accuracy_matched_speedup"),
        )


if __name__ == "__main__":
    main()
