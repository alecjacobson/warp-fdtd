"""Render a compact report from completed measurements; no synthetic data."""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import numpy as np

from .common import ROOT, write_json


def main():
    root = ROOT / "results"
    records = {
        p.parent.name: json.loads(p.read_text()) for p in root.glob("*/result.json")
    }
    completed = {k: v for k, v in records.items() if v.get("status") == "complete"}
    quality_file = root / "quality_checks.json"
    quality = json.loads(quality_file.read_text()) if quality_file.exists() else {}
    forward = [(k, v) for k, v in completed.items() if k.startswith("forward")]
    forward.sort(
        key=lambda item: (item[1]["settings"]["backend"], np.prod(item[1]["shape"]))
    )
    comparisons = (
        json.loads((root / "comparisons.json").read_text())
        if (root / "comparisons.json").exists()
        else {}
    )
    lines = [
        "# Measured FDTDX baseline",
        "",
        "Hardware: RTX PRO 6000 Blackwell **MIG 1g.24gb (46 SMs)**; CPU allocation: four cores, 16 GiB host memory.",
        "Upstream: `2cab8dbb2af9a43ec6443712f011f50ecf58790b`; JAX 0.12.0; float32 unless identified otherwise.",
        "",
        "This is a measured single-partition solver and optimized-bend baseline. It is not a reproduction of the paper's five-seed optimization campaign or multi-GPU capacity result.",
        "",
        *(
            [
                f"**Validated bend: {100 * quality['fine_mesh_T']:.2f}% modal transmission at 1550 nm.** All six independent quality gates pass; see [the saved gate results](quality_checks.json).",
                "",
            ]
            if quality.get("passed")
            else []
        ),
        "![Solver performance](performance.png)",
        "",
        "## Complete forward simulations",
        "",
        "All rows use 200 fs and the same physical bend domain. Values are synchronized medians excluding compilation and warmup.",
        "",
        "These throughput cases use 10-cell PMLs, deterministic random devices and legacy flux monitors. Each mesh is a separate workload; convergence is evaluated separately with the saved optimized candidate and fixed physical PML thickness.",
        "",
        "| Backend | Cells | Steps | Median (s) | Min–max (s) | GCUPS | JAX peak (GB) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, r in forward:
        t = r["run_seconds"]
        peak = (r.get("memory_stats") or {}).get("peak_bytes_in_use")
        mem = f"{peak / 1e9:.3f}" if peak is not None else "n/a"
        lines.append(
            f"| {r['settings']['backend']} | {np.prod(r['shape']):,} | {r['time_steps']:,} | "
            f"{r['median_seconds']:.3f} | {min(t):.3f}–{max(t):.3f} | {r['gcups']:.3f} | {mem} |"
        )
    lines += ["", "## Numerical checks", ""]
    physics_path = root / "physics_gpu.json"
    if physics_path.exists():
        tests = json.loads(physics_path.read_text())["tests"]
        lines += [
            f"Upstream analytic GPU tests: {sum(t['status'] == 'passed' for t in tests)}/{len(tests)} passed.",
            "",
        ]
    if "cpu_gpu" in comparisons:
        c = comparisons["cpu_gpu"]
        lines.append(
            f"CPU/GPU parity: **{'PASS' if c['passed'] else 'FAIL'}**; worst relative L2 error "
            f"{max(c['relative_l2_errors'].values()):.3g}; matched-case speedup {c['speedup']:.2f}×."
        )
    lines += [
        "",
        "| Recorder | Relative gradient L2 error vs checkpointed | Cosine | Best directional FD error | Gate |",
        "|---|---:|---:|---:|---|",
    ]
    for method, r in comparisons.get("gradients", {}).items():
        lines.append(
            f"| {method} | {r['relative_l2_error']:.4g} | {r['cosine']:.10f} | "
            f"{r['best_finite_difference_relative_error']:.4g} | {'PASS' if r['passed'] else 'FAIL'} |"
        )
    lines += [
        "",
        "The float8/every-five-step recorder fails the 1% directional-derivative gate on this case. "
        "Use the exact segmented path as the accuracy reference; accept the faster frequency adjoint only on matched cases where its gradient gate passes.",
    ]
    if "gradient50_exact64" in completed:
        r = completed["gradient50_exact64"]
        lines += [
            "",
            "Float64-field/parameter diagnostic finite differences (detectors retain float32; not a float32 speed benchmark):",
            "",
            "| h | Relative error |",
            "|---:|---:|",
        ]
        lines += [
            f"| {x['h']} | {x['relative_error']:.5g} |" for x in r["finite_difference"]
        ]
    if "gradient20_exact" in completed:
        r = completed["gradient20_exact"]
        lines += [
            "",
            f"Full 3-million-cell exact gradient: **{r['median_seconds']:.2f} s**, "
            f"{r['memory_stats']['peak_bytes_in_use'] / 1e9:.2f} GB peak allocator use.",
        ]
    for name in (
        "frequency_adjoint",
        "frequency_adjoint_full",
        "frequency_adjoint_hann",
    ):
        if name in comparisons:
            a = comparisons[name]
            lines += [
                "",
                f"`{name}`: **{'PASS' if a['passed'] else 'FAIL'}**, relative L2 gradient error "
                f"{a['relative_l2_error_vs_segmented']:.4g} versus exact segmented reconstruction.",
            ]
            if "speedup" in a:
                lines.append(
                    f"Matched full-resolution gradient speedup: {a['speedup']:.2f}×."
                )
    if all(
        n in completed for n in ("gradient20_hann_exact", "gradient20_hann_adjoint")
    ):
        lines += [
            "",
            "Validated Hann-objective gradient costs (same saved 3-million-cell design):",
            "",
            "| Method | Median (s) | Peak JAX allocation (GB) |",
            "|---|---:|---:|",
        ]
        for method in ("exact", "adjoint"):
            r = completed[f"gradient20_hann_{method}"]
            lines.append(
                f"| {method} | {r['median_seconds']:.3f} | {r['memory_stats']['peak_bytes_in_use'] / 1e9:.3f} |"
            )
    if "capacity192m" in completed:
        r = completed["capacity192m"]
        lines += [
            "",
            "## Capacity stress (not a physical transmission evaluation)",
            "",
            f"{np.prod(r['shape']):,} cells × {r['time_steps']:,} steps, "
            f"{r['median_seconds']:.2f} s median, {r['gcups']:.3f} GCUPS, "
            f"{r['memory_stats']['peak_bytes_in_use'] / 1e9:.2f} GB peak JAX allocation. "
            "This short prefix establishes capacity and finite-field behavior only.",
        ]
    lines += [
        "",
        "## Device quality",
        "",
        "Legacy core flux and late-time fundamental-mode transmission are distinct measurements.",
        "",
        "![Optimized bend](device.png)",
        "",
        "| Run | Grid (nm) | Time (fs) | Legacy flux ratio | Modal T | Modal R |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, r in sorted(completed.items()):
        if (
            r.get("mode") == "forward"
            and r.get("measurements")
            and "modal_transmission" in r["measurements"][0]
        ):
            m = r["measurements"][0]
            lines.append(
                f"| {name} | {r['settings']['spacing_nm']} | {r['settings']['time_fs']} | "
                f"{m['transmission']:.5f} | {m['modal_transmission']:.5f} | {m['modal_reflection']:.5f} |"
            )
    for name, r in records.items():
        if r.get("mode") == "optimize":
            history_path = root / name / "history.json"
            history = (
                json.loads(history_path.read_text()) if history_path.exists() else []
            )
            lines += [
                "",
                f"Optimization `{name}`: {r['status']}, {len(history)} saved evaluated steps.",
            ]
            if history:
                lines.append(
                    f"Initial objective {history[0]['objective']:.5f}; best {max(x['objective'] for x in history):.5f}."
                )
    quality_path = root / "quality_checks.json"
    rectangular_path = root / "quality_checks_rectangular.json"
    if rectangular_path.exists():
        q = json.loads(rectangular_path.read_text())
        if "time_T_change" in q:
            lines += [
                "",
                f"Rectangular-window duration gate: **{'PASS' if q['time_convergence'] else 'FAIL'}**, "
                f"{100 * q['time_T_change']:.3f} percentage-point variation. "
                "This motivated the built-in Hann-window evaluation; the candidate and thresholds were unchanged.",
            ]
    if quality_path.exists():
        q = json.loads(quality_path.read_text())
        lines += ["", "## Independent quality gates", ""]
        if q["missing"]:
            lines.append("Pending: " + ", ".join(q["missing"]) + ".")
        else:
            lines += [
                f"Overall ({q.get('window', 'rectangular')} window): **{'PASS' if q['passed'] else 'FAIL'}**.",
                "",
            ]
            lines += [
                f"- {name}: {'PASS' if passed else 'FAIL'}"
                for name, passed in q["checks"].items()
            ]
            lines += [
                "",
                f"Duration change: {100 * q['time_T_change']:.4f} percentage points; "
                f"PML change: {100 * q['pml_T_change']:.4f}; mesh change: {100 * q['mesh_T_change']:.4f}.",
            ]
    if (root / "convergence/result.json").exists():
        lines += [
            "",
            "## Additional diagnostics",
            "",
            "[Cycle-averaged bend Poynting flow](bend_flow20/poynting_flow.png) shows a central-plane projection from co-located, time-centered phasors; adding the passive detector left transmission unchanged.",
            "",
            "[Analytic Maxwell-mode accuracy trends](convergence/accuracy_trends.png) separate joint mesh/time refinement from time-step error at fixed mesh. These are measured GPU runs with an analytic reference; methods and limits are in README.md.",
        ]
    lines += [
        "",
        "## Limits",
        "",
        "- No multi-GPU or billion-cell result was measured on this partition.",
        "- A curved-guide initialization and modal objective do not reproduce the publication's five random-start binary/brush optimizations.",
        "- A finite optimized objective alone does not certify 92% physical efficiency, fabrication constraints, mesh/time/PML convergence, or independent-solver agreement.",
        "- The smooth projected design is not certified to meet the publication's binary/100 nm brush fabrication constraint. Independent-solver validation remains for the Warp comparison.",
        "- Peak JAX allocation is not total GPU memory. MIG isolation does not establish exclusive use of all parent-device resources.",
        "- Source, methodology, reproduction commands, and the Warp comparison contract are in [README.md](../README.md).",
        "",
    ]
    (root / "REPORT.md").write_text("\n".join(lines))
    write_json(
        root / "summary.json",
        {"completed_runs": list(completed), "comparisons": comparisons},
    )
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for backend, color in (("gpu", "#007f86"), ("cpu", "#df7852")):
        rows = [r for _, r in forward if r["settings"]["backend"] == backend]
        if rows:
            cells = [np.prod(r["shape"]) / 1e6 for r in rows]
            axes[0].plot(
                cells,
                [r["gcups"] for r in rows],
                "o-",
                color=color,
                label=backend.upper(),
            )
            axes[1].plot(
                cells,
                [r["median_seconds"] for r in rows],
                "o-",
                color=color,
                label=backend.upper(),
            )
    axes[0].set(
        ylabel="Billion cell updates / second", title="Sustained solver throughput"
    )
    axes[1].set(ylabel="Seconds (median)", yscale="log", title="Full 200 fs simulation")
    for ax in axes:
        ax.set(xlabel="Grid cells (millions)", xscale="log")
        ticks = sorted({np.prod(r["shape"]) / 1e6 for _, r in forward})
        ax.set_xticks(ticks, [f"{x:.3g}" for x in ticks])
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.grid(alpha=0.2)
        ax.legend(frameon=False)
    fig.suptitle(
        "FDTDX • RTX PRO 6000 Blackwell MIG 1g.24gb vs four CPU cores", fontsize=12
    )
    fig.savefig(root / "performance.png", dpi=180)
    plt.close(fig)
    candidate = next(
        (
            n
            for n in ("hann_10nm", "hann_300fs", "quality_200fs", "ring20_initial")
            if n in completed
        ),
        None,
    )
    if candidate in completed:
        r = completed[candidate]
        with np.load(root / candidate / "arrays.npz") as data:
            eps = np.squeeze(1 / data["inv_eps"])
            intensity = np.sum(data["E"] ** 2, axis=0)
        extent = (0, r["settings"]["domain_um"], 0, r["settings"]["domain_um"])
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), constrained_layout=True)
        im = axes[0].imshow(
            np.sqrt(eps).T,
            origin="lower",
            extent=extent,
            cmap="copper",
            vmin=1,
            vmax=3.5,
        )
        fig.colorbar(im, ax=axes[0], label="Refractive index")
        axes[0].set_title("Material cross section")
        im = axes[1].imshow(
            np.log10(np.maximum(intensity / intensity.max(), 1e-5)).T,
            origin="lower",
            extent=extent,
            cmap="magma",
            vmin=-4,
            vmax=0,
        )
        axes[1].contour(
            eps.T,
            levels=[6.625],
            colors="white",
            linewidths=0.5,
            alpha=0.65,
            extent=extent,
        )
        fig.colorbar(im, ax=axes[1], label="log₁₀ normalized |E|²")
        axes[1].set_title("Final electric field cross section")
        for ax in axes:
            ax.set(xlabel="x (µm)", ylabel="y (µm)")
        m = r["measurements"][0]
        fig.suptitle(
            f"{candidate}: modal T = {100 * m['modal_transmission']:.2f}%, R = {100 * m['modal_reflection']:.3g}%"
        )
        fig.savefig(root / "device.png", dpi=180)
        plt.close(fig)
    history_path = root / "ring20_optimized/history.json"
    if history_path.exists():
        history = json.loads(history_path.read_text())
        fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
        ax.plot(
            [r["epoch"] for r in history],
            [100 * r["objective"] for r in history],
            "o-",
            color="#007f86",
            markersize=3,
        )
        ax.set(
            xlabel="Evaluated optimization step",
            ylabel="Training objective: rectangular-window T (%)",
            title="20-step exact-gradient optimization from a curved guide",
        )
        ax.grid(alpha=0.2)
        fig.savefig(root / "optimization.png", dpi=180)
        plt.close(fig)


if __name__ == "__main__":
    main()
