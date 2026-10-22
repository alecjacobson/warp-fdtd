"""Build concise current comparisons from measured records."""

import json
from pathlib import Path
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/warp"


def read(path):
    return json.loads(path.read_text())


def performance():
    cases = [
        ("192k / flux", "forward50_gpu"),
        ("3M / modal", "hann20"),
        ("24M / flux", "forward10_gpu"),
        ("24M / modal", "hann10"),
    ]
    rows = [read(OUT / case / "result.json") for label, case in cases]
    assert all(r["passed"] for r in rows)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8), layout="constrained")
    x = np.arange(len(rows))
    axes[0].bar(
        x - 0.18,
        [r["reference_seconds"] for r in rows],
        0.36,
        label="FDTDX",
        color="#dd8750",
    )
    axes[0].bar(
        x + 0.18,
        [r["median_seconds"] for r in rows],
        0.36,
        label="Warp Yee",
        color="#188b93",
    )
    axes[0].set_yscale("log")
    axes[0].set_xticks(x, [a for a, b in cases])
    axes[0].set_ylabel("Full trajectory, seconds (log scale)")
    axes[0].legend()
    axes[0].set_title("Matched inputs, duration and detectors")
    for i, r in enumerate(rows):
        axes[0].text(
            i,
            max(r["reference_seconds"], r["median_seconds"]) * 1.25,
            f"{r['speedup']:.2f}×",
            ha="center",
            fontsize=9,
        )
    for name in ("E", "H"):
        axes[1].semilogy(x, [r["relative_l2"][name] for r in rows], "o-", label=name)
    axes[1].axhline(1e-3, color="gray", ls="--", label="Acceptance limit")
    axes[1].set_xticks(x, [a for a, b in cases])
    axes[1].set_ylabel("Relative field L2 difference")
    axes[1].set_title("Full 3D at 192k; mid-z slices at larger grids")
    axes[1].legend()
    fig.suptitle("RTX PRO 6000 Blackwell MIG 1g.24gb • float32 • median of 3 warm runs")
    for ax in axes:
        ax.grid(axis="y", alpha=0.2)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"performance.{ext}", dpi=160)
    plt.close(fig)
    return rows


def flow():
    ref = np.load(ROOT / "results/bend_flow20/phasors.npz")["phasors"]
    cur = np.load(OUT / "bend_flow20/flow_phasors.npz")["phasors"].squeeze(-1)

    def S(p):
        return 0.5 * np.real(np.cross(p[:3], p[3:].conj(), axis=0))

    a, b = S(ref), S(cur)
    eps = np.load(ROOT / "results/bend_flow20/poynting.npz")["permittivity"]
    error = float(np.linalg.norm(a - b) / np.linalg.norm(a))
    assert error < 1e-3
    scale = np.linalg.norm(a, axis=0).max()
    xy = np.arange(a.shape[1]) * 0.02
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), layout="constrained")
    for ax, data, title in zip(axes[:2], (a, b), ("FDTDX", "Warp Yee")):
        mag = np.linalg.norm(data, axis=0) / scale
        im = ax.imshow(
            np.log10(np.maximum(mag, 1e-5)).T,
            origin="lower",
            extent=(-0.01, 3.99, -0.01, 3.99),
            vmin=-4,
            vmax=0,
            cmap="magma",
        )
        ax.contour(xy, xy, eps.T, levels=[6.625], colors="cyan", linewidths=0.5)
        mask = mag < 1e-4
        ax.streamplot(
            xy,
            xy,
            np.ma.masked_where(mask, data[0]).T,
            np.ma.masked_where(mask, data[1]).T,
            density=1.3,
            color="white",
            linewidth=0.45,
            arrowsize=0.6,
        )
        ax.set_title(title)
    fig.colorbar(im, ax=axes[:2], label="log₁₀(|⟨S⟩| / common maximum)", shrink=0.75)
    diff = np.linalg.norm(a - b, axis=0) / scale
    im = axes[2].imshow(
        diff.T,
        origin="lower",
        extent=(-0.01, 3.99, -0.01, 3.99),
        cmap="viridis",
        vmin=0,
    )
    fig.colorbar(im, ax=axes[2], label="|Δ⟨S⟩| / common maximum", shrink=0.75)
    axes[2].set_title(f"Difference • relative L2 {error:.2e}")
    for ax in axes:
        ax.set(xlabel="x (µm)", ylabel="y (µm)", aspect="equal")
    fig.suptitle(
        "Same optimized 3D bend • 20 nm • 300 fs • central xy slice • Hann cycle average"
    )
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"flagship_flow.{ext}", dpi=170)
    plt.close(fig)
    (OUT / "flow_comparison.json").write_text(
        json.dumps(
            {
                "relative_l2_poynting": error,
                "passed": True,
                "scope": "Single xy slice through the full 3D field; common scale, not independently normalized",
            },
            indent=2,
        )
        + "\n"
    )
    return error


def accuracy():
    base = read(ROOT / "results/convergence/result.json")["rows"]
    new = read(OUT / "convergence/result.json")
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.9), layout="constrained")
    for label, rows, style in [
        ("FDTDX Yee", base, "s--"),
        ("Warp Yee", [r for r in new["rows"] if r["method"] == "yee"], "o-"),
        ("Warp fourth order", [r for r in new["rows"] if r["method"] == "rk4"], "^-"),
    ]:
        selected = sorted(
            [r for r in rows if r["courant_factor"] == 0.99], key=lambda r: r["dx_nm"]
        )
        axes[0].loglog(
            [r["dx_nm"] for r in selected],
            [r["continuum_relative_error"] for r in selected],
            style,
            label=label,
        )
        time_rows = sorted(
            [r for r in rows if r["cells_per_wavelength"] == 32],
            key=lambda r: r["dt_fs"],
        )
        axes[1].loglog(
            [r["dt_fs"] for r in time_rows],
            [r["temporal_relative_error"] for r in time_rows],
            style,
            label=label,
        )
        if label != "FDTDX Yee":
            axes[2].loglog(
                [r["median_seconds"] * 1e3 for r in selected],
                [r["continuum_relative_error"] for r in selected],
                style,
                label=label,
            )
    axes[0].set(
        xlabel="Grid spacing (nm)",
        ylabel="Relative frequency error vs continuum",
        title="Space + time refinement (CFL 0.99)",
    )
    axes[1].set(
        xlabel="Time step (fs)",
        ylabel="Relative error vs semidiscrete frequency",
        title="Time error at 32 cells/λ",
    )
    axes[2].set(
        xlabel="Warm trajectory runtime (ms)",
        ylabel="Relative frequency error vs continuum",
        title="Warp accuracy per runtime",
    )
    axes[0].set_xticks(
        [12.109, 24.219, 48.438, 96.875, 193.75],
        ["12.1", "24.2", "48.4", "96.9", "193.8"],
    )
    axes[1].set_xticks([0.012, 0.025, 0.05, 0.1], ["0.012", "0.025", "0.05", "0.1"])
    axes[2].set_xticks([2, 5, 10, 20, 50], ["2", "5", "10", "20", "50"])
    for ax in axes:
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.grid(alpha=0.2, which="both")
        ax.legend(fontsize=8)
    fig.suptitle(
        "Measured periodic vacuum Maxwell mode • 20 periods • float32 • fourth order limited by roundoff at fine grids"
    )
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"accuracy.{ext}", dpi=170)
    plt.close(fig)
    return new


def main():
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False})
    rows = performance()
    err = flow()
    has_accuracy = (OUT / "convergence/result.json").exists()
    conv = accuracy() if has_accuracy else None
    table = "\n".join(
        f"| {label} | {r['reference_seconds']:.3f} s | {r['median_seconds']:.3f} s | {r['speedup']:.2f}× |"
        for label, r in zip(
            (
                "192k cells, flux",
                "3M cells, modal",
                "24M cells, flux",
                "24M cells, modal",
            ),
            rows,
        )
    )
    acc_status = (
        f"The smooth periodic test measures order **{conv['orders']['rk4']:.2f}** with Warp’s optional fourth-order propagator versus **{conv['orders']['yee']:.2f}** with Yee. This higher-order path is not yet validated for dielectric interfaces or CPML."
        if conv
        else "A fourth-order periodic accuracy study is in progress; the flagship currently uses matched Yee discretization."
    )
    fine = rows[-1]["measurements"]["modal_transmission"]
    text = f"""# Warp FDTD

Pure NVIDIA Warp 3D electromagnetic solver, compared against pinned, unmodified
FDTDX. The implemented flagship is a silicon 90° waveguide bend with all six E/H
components, CPML, an injected mode, and Hann-window modal detectors.

**Status:** forward correctness gates pass on the saved cases. On the tested
RTX PRO 6000 Blackwell **MIG 1g.24gb** (46 SMs), matched forward runs are
**{min(r["speedup"] for r in rows):.2f}–{max(r["speedup"] for r in rows):.2f}× faster**. Fine-grid bend transmission is **{100 * fine:.4f}%**.
{acc_status}

| Matched workload | FDTDX | Warp | Speedup |
|---|---:|---:|---:|
{table}

![Latest performance and field agreement](results/warp/performance.png)

"""
    if conv:
        text += "![Latest measured accuracy trends](results/warp/accuracy.png)\n\n"
    text += f"""![Flagship power-flow comparison](results/warp/flagship_flow.png)

The flow panels use one common scale; vector-field relative L2 difference is
**{err:.2e}**. These are slices of full 3D simulations. Timing excludes setup,
compilation and host postprocessing; raw repeats and limitations are in
[measurement details](docs/warp.md). No general superiority claim is made for
untested geometries, hardware, boundaries or gradients.

```bash
uv venv .venv-warp
uv pip install --python .venv-warp/bin/python -r requirements-warp.txt
.venv-warp/bin/python -m unittest discover -s tests -v
.venv-warp/bin/python -m warp_fdtd.benchmark
```

[Prioritized issues](https://github.com/alecjacobson/warp-fdtd/issues) ·
[Solver and reproduction](docs/warp.md) ·
[FDTDX baseline methodology](docs/fdtdx-baseline.md) ·
[Raw results](results/warp)
"""
    (ROOT / "README.md").write_text(text)


if __name__ == "__main__":
    main()
