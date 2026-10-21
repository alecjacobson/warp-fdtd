"""Render cycle-averaged bend power flow from co-located CW phasors."""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

from .common import ROOT, write_json


def main():
    output = ROOT / "results/bend_flow20"
    record = json.loads((output / "result.json").read_text())
    reference = json.loads((ROOT / "results/hann_300fs/result.json").read_text())
    assert record["status"] == "complete"
    assert record["loaded_params_sha256"] == reference["loaded_params_sha256"]
    with np.load(output / "phasors.npz") as saved:
        phasors = saved["phasors"]
        eps = np.squeeze(1 / saved["inv_eps"])
        weights = saved["window_weights"]
        dt = float(saved["dt_s"])
        grid_slice = saved["grid_slice"]
    # FDTDX stores E_SI / eta0 and H_SI. Relative vector field is enough for
    # visualization; no absolute launched-power calibration is asserted here.
    S = 0.5 * np.real(
        np.cross(phasors[:3], np.conj(phasors[3:]), axisa=0, axisb=0, axisc=0)
    )
    assert np.isfinite(S).all()
    dx = record["settings"]["spacing_nm"] / 1000
    x, y = [np.arange(n) * dx for n in S.shape[1:]]
    magnitude = np.linalg.norm(S, axis=0)
    pml = record["settings"]["pml_cells"]
    scale = float(magnitude[pml:-pml, pml:-pml].max())
    low = magnitude < scale * 1e-4
    np.savez_compressed(
        output / "poynting.npz",
        S=S,
        x_um=x,
        y_um=y,
        permittivity=eps,
        magnitude_scale=scale,
    )
    delta = abs(
        record["measurements"][0]["modal_transmission"]
        - reference["measurements"][0]["modal_transmission"]
    )
    assert delta < 1e-6, "Passive flow detector changed the evaluated result"
    checks = {
        "finite_vector_field": True,
        "transmission_change_from_adding_detector": delta,
        "candidate_sha256": record["loaded_params_sha256"],
        "formula": "0.5 * Re(E_phasor cross conjugate(H_phasor)) in FDTDX field units",
        "sampling": "Upstream exact_interpolation: spatial E_z co-location and (H_previous+H_current)/2 time centering; Hann-weighted final eight periods.",
        "window_support_fs": [
            float(t * dt * 1e15) for t in np.flatnonzero(weights > 0)[[0, -1]]
        ],
        "slice_z_um_from_lower_face": float((grid_slice[2, 0] + 0.5) * dx),
        "scope": "Cycle-averaged monochromatic flow, xy projection of a 3D field on one slice. Streamlines are not a full-volume energy budget; color is normalized magnitude, not watts.",
    }
    write_json(output / "flow_checks.json", checks)
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), constrained_layout=True)
    extent = (-dx / 2, x[-1] + dx / 2, -dx / 2, y[-1] + dx / 2)
    im = axes[0].imshow(
        np.sqrt(eps).T, origin="lower", extent=extent, cmap="copper", vmin=1, vmax=3.5
    )
    fig.colorbar(im, ax=axes[0], label="Refractive index")
    axes[0].set_title("Saved optimized bend • material cross section")
    im = axes[1].imshow(
        np.log10(np.maximum(magnitude / scale, 1e-5)).T,
        origin="lower",
        extent=extent,
        cmap="magma",
        vmin=-4,
        vmax=0,
    )
    fig.colorbar(im, ax=axes[1], label="log₁₀(|⟨S⟩| / maximum in active region)")
    axes[1].contour(
        x, y, eps.T, levels=[6.625], colors="cyan", linewidths=0.6, alpha=0.8
    )
    axes[1].streamplot(
        x,
        y,
        np.ma.masked_where(low, S[0]).T,
        np.ma.masked_where(low, S[1]).T,
        density=1.8,
        color="white",
        linewidth=0.55,
        arrowsize=0.8,
    )
    axes[1].set_title("Cycle-averaged Poynting flow • in-plane projection")
    for ax in axes:
        ax.add_patch(
            Rectangle(
                (pml * dx, pml * dx),
                (len(x) - 2 * pml) * dx,
                (len(y) - 2 * pml) * dx,
                fill=False,
                linestyle=":",
                edgecolor="gray",
                linewidth=0.8,
            )
        )
        ax.set(
            xlabel="x from domain lower face (µm)",
            ylabel="y from domain lower face (µm)",
            xlim=(0, 4),
            ylim=(0, 4),
            aspect="equal",
        )
    fig.suptitle(
        "1550 nm • 20 nm grid • z = 0.75 µm • Hann window after settling\nWhite arrows show power-flow direction; dotted rectangle marks the PML interface",
        fontsize=11,
    )
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"poynting_flow.{suffix}", dpi=190)
    plt.close(fig)


if __name__ == "__main__":
    main()
