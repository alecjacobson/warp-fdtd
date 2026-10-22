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
        ("3M / 200 fs", "packed_control20_full"),
        ("24M / 300 fs", "packed_control10_full"),
    ]
    rows = [read(OUT / case / "result.json") for label, case in cases]
    assert all(r["passed"] for r in rows)
    gradient_path = OUT / "gradient20/result.json"
    gradient = read(gradient_path) if gradient_path.exists() else None
    if gradient and (gradient["status"] != "complete" or not gradient["passed"]):
        gradient = None
    count = 3 if gradient else 2
    fig, axes = plt.subplots(
        1, count, figsize=(5.25 * count, 3.8), layout="constrained"
    )
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
    axes[0].set_ylim(top=max(r["reference_seconds"] for r in rows) * 1.8)
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
    axes[1].set_title("Mid-z field agreement on the 3D bend")
    axes[1].legend()
    if gradient:
        axes[2].bar(
            [0, 1, 2],
            [
                gradient["exact_fdtdx_seconds"],
                gradient["fast_fdtdx_seconds"],
                gradient["median_seconds"],
            ],
            color=["#dd8750", "#efbd83", "#188b93"],
        )
        axes[2].set_xticks(
            [0, 1, 2], ["FDTDX exact", "FDTDX fast", "Warp exact"], rotation=15
        )
        axes[2].set_ylabel("Gradient evaluation (seconds)")
        axes[2].set_title(
            f"3M gradient • {gradient['speedup_vs_fast']:.2f}× vs fast adjoint"
        )
    fig.suptitle(
        "RTX PRO 6000 Blackwell MIG 1g.24gb • float32 • synchronized warm runs"
    )
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
    timing = read(ROOT / "results/periodic_performance/result.json")["rows"]
    fd = next(r for r in timing if r["cells_per_wavelength"] == 64)
    wa = next(
        r
        for r in new["rows"]
        if r["method"] == "rk4"
        and r["cells_per_wavelength"] == 32
        and r["courant_factor"] == 0.99
    )
    comparison = dict(
        fdtdx_cells_per_wavelength=64,
        warp_cells_per_wavelength=32,
        fdtdx_seconds=fd["median_seconds"],
        warp_seconds=wa["median_seconds"],
        speedup=fd["median_seconds"] / wa["median_seconds"],
        frequency_error_reduction=fd["continuum_relative_error"]
        / wa["continuum_relative_error"],
        warp_amplitude_loss_20_periods=wa["amplitude_loss_20_periods"],
        scope="Periodic smooth vacuum mode; one interior example from complete plotted mesh sweeps, not a bend/interface result",
    )
    (OUT / "periodic_cost_comparison.json").write_text(
        json.dumps(comparison, indent=2) + "\n"
    )
    for r in timing:
        matched = next(
            w
            for w in new["rows"]
            if w["method"] == "yee"
            and w["courant_factor"] == 0.99
            and w["cells_per_wavelength"] == r["cells_per_wavelength"]
        )
        assert r["steps"] == matched["steps"] and r["shape"] == matched["shape"]
        assert np.isclose(r["dt_fs"], matched["dt_fs"], rtol=1e-5, atol=0)
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
        cost_rows = timing if label == "FDTDX Yee" else selected
        color = {
            "FDTDX Yee": "tab:blue",
            "Warp Yee": "tab:orange",
            "Warp fourth order": "tab:green",
        }[label]
        axes[2].loglog(
            [r["median_seconds"] * 1e3 for r in cost_rows],
            [r["continuum_relative_error"] for r in cost_rows],
            style,
            label=label,
            color=color,
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
        title="Measured accuracy per runtime",
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
    table = "\n".join(
        f"| {label} | {r['reference_seconds']:.3f} s | {r['median_seconds']:.3f} s | {r['speedup']:.2f}× |"
        for label, r in zip(
            ("20 nm · 3M cells · 200 fs", "10 nm · 24M cells · 300 fs"), rows
        )
    )
    gradient = read(OUT / "gradient20/result.json")
    assert gradient["status"] == "complete" and gradient["passed"]
    fine = rows[-1]["measurements"]["modal_transmission"]
    text = f"""# Warp FDTD

Pure NVIDIA Warp 3D electromagnetic solver, compared with pinned, unmodified
FDTDX on an optimized silicon 90° waveguide bend. The flagship uses explicit
Yee updates, a uniform grid, CPML, injected modes and Hann-window modal detectors.

**Measured status:** **{min(r["speedup"] for r in rows):.2f}–{max(r["speedup"] for r in rows):.2f}× faster** matched forward runs;
**{100 * fine:.4f}%** fine-grid transmission. Adaptive flagship results are pending.

| Same-input flagship workload | FDTDX | Warp | Speedup |
|---|---:|---:|---:|
{table}

![Flagship performance and numerical agreement](results/warp/performance.png)

Field agreement is against FDTDX, not continuum ground truth. These archived
mesh levels have slightly different source/output positions; controlled physical
convergence is tracked in [#25](https://github.com/alecjacobson/warp-fdtd/issues/25).
The 3M-cell material gradient is **{gradient["speedup_vs_fast"]:.2f}× faster** than
FDTDX's fast adjoint, with **{gradient["relative_l2"]:.2e}** relative gradient error
and **{gradient["history_bytes"] / 1e9:.2f} GB** of stored history.

![Flagship time-averaged power flow](results/warp/flagship_flow.png)

Common-scale central slices of the full 3D bend, 20 nm / 300 fs. Poynting-vector
relative L2 difference: **{err:.2e}**. Measurements use an RTX PRO 6000 Blackwell
**MIG 1g.24gb**, float32, synchronized warm runs; timings exclude compilation,
setup and host postprocessing.

[Reproduction and limitations](docs/warp.md) ·
[Adaptivity progress](docs/adaptivity-progress.md) ·
[Prioritized issues](https://github.com/alecjacobson/warp-fdtd/issues) ·
[Raw results](results/warp)
"""
    (ROOT / "README.md").write_text(text)


if __name__ == "__main__":
    main()
