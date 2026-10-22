"""Offline field-representation study. NOT an adaptive propagation benchmark.

Replace coarse reconstruction by the exact reference inside hypothetical fine
blocks. This optimistic diagnostic cannot certify a mesh or a speedup: coupled
Maxwell evolution, CFL, interfaces and GPU overhead still need independent tests.
"""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
import numpy as np
from scipy.ndimage import distance_transform_edt, map_coordinates

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def reconstruct(field, ratio):
    """Linear reconstruction from coarse knots, including both domain endpoints."""
    knots = [np.unique(np.r_[np.arange(0, n, ratio), n - 1]) for n in field.shape]
    coarse = field[np.ix_(*knots)]
    coordinates = np.meshgrid(
        *[
            np.interp(np.arange(n), k, np.arange(len(k)))
            for n, k in zip(field.shape, knots)
        ],
        indexing="ij",
    )
    return map_coordinates(
        coarse, coordinates, order=1, mode="nearest", prefilter=False
    )


def block_mask(mask, block):
    shape = mask.shape
    padded = tuple(((n + block - 1) // block) * block for n in shape)
    a = np.pad(mask, [(0, p - n) for p, n in zip(padded, shape)])
    selected = a.reshape(
        padded[0] // block, block, padded[1] // block, block, padded[2] // block, block
    ).any(axis=(1, 3, 5))
    expanded = selected.repeat(block, 0).repeat(block, 1).repeat(block, 2)
    return expanded[tuple(slice(n) for n in shape)], int(selected.sum())


def load_volume(directory):
    meta = json.loads((directory / "result.json").read_text())
    shape = meta["sample_shape"]
    if len(shape) != 3 or any(not isinstance(n, int) or n < 1 for n in shape):
        raise ValueError("Invalid volume shape")
    a = np.empty((6, *shape), np.complex64)
    covered = np.zeros(shape[2], dtype=bool)
    for chunk in meta["files"]:
        lo, hi = chunk["z_start"], chunk["z_stop"]
        if not 0 <= lo < hi <= shape[2] or covered[lo:hi].any():
            raise ValueError("Invalid or overlapping volume chunks")
        path = directory / chunk["file"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != chunk["sha256"]:
            raise ValueError(f"Volume checksum mismatch: {path}")
        with np.load(path) as z:
            field = z["phasors"]
            if (
                field.shape != (6, shape[0], shape[1], hi - lo)
                or not np.isfinite(field).all()
            ):
                raise ValueError("Invalid volume chunk data")
            a[..., lo:hi] = field
        covered[lo:hi] = True
    if not covered.all():
        raise ValueError("Incomplete volume chunks")
    return meta, a


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--volume", default="results/warp/volume20")
    p.add_argument("--output", default="results/warp/opportunity20")
    args = p.parse_args()
    directory = Path(args.volume)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    meta, fields = load_volume(directory)
    case = Path(meta["case"])
    raw = case / "inputs.npz"
    if hashlib.sha256(raw.read_bytes()).hexdigest() != meta["inputs_sha256"]:
        raise ValueError("Input checksum mismatch")
    with np.load(raw) as inp:
        eps = 1 / inp["inv_permittivity"][0]
    # First study deliberately uses the native grid: no inferred resampling of materials.
    if (
        list(eps.shape) != meta["sample_shape"]
        or meta["solver_shape"] != meta["sample_shape"]
    ):
        raise ValueError(
            "Opportunity study currently requires native-grid volume phasors"
        )
    axes = [np.asarray(a) for a in meta["axes_m"]]
    active = np.ones(eps.shape, bool)
    for axis, x in enumerate(axes):
        sh = [1, 1, 1]
        sh[axis] = len(x)
        lo = x[0] + 0.2e-6
        hi = x[-1] - 0.2e-6
        active &= ((x >= lo) & (x <= hi)).reshape(sh)
    if not active.any():
        raise ValueError("Domain too small for the declared 200 nm exclusion")
    # Energy weighting is a colocated diagnostic, not a discrete conserved invariant.
    energy = (
        eps * np.sum(np.abs(fields[:3]) ** 2, axis=0)
        + np.sum(np.abs(fields[3:]) ** 2, axis=0)
    ).astype(float)
    total = float(energy[active].sum())
    if not total > 0 or not np.isfinite(total):
        raise ValueError("Invalid field energy")
    concentration = []
    ranked = np.sort(energy[active].ravel())[::-1]
    cumulative = np.cumsum(ranked) / total
    for share in (0.9, 0.99, 0.999, 0.9999):
        concentration.append(
            dict(
                energy_fraction=share,
                active_volume_fraction=float(
                    (np.searchsorted(cumulative, share) + 1) / len(ranked)
                ),
            )
        )
    silicon = eps > 2.3
    interfaces = np.zeros_like(silicon)
    for axis in range(3):
        change = np.abs(np.diff(eps, axis=axis)) > 1e-3
        lo = [slice(None)] * 3
        hi = lo.copy()
        lo[axis] = slice(None, -1)
        hi[axis] = slice(1, None)
        interfaces[tuple(lo)] |= change
        interfaces[tuple(hi)] |= change
    policies = {"silicon": silicon, "all_material_interfaces": silicon | interfaces}
    masks = {}
    for policy, base in policies.items():
        distance = distance_transform_edt(~base, sampling=meta["sample_nm"])
        for buffer_nm in (0, 40, 80, 160, 320):
            for block in (4, 8, 16):
                mask, blocks = block_mask(distance <= buffer_nm, block)
                masks[policy, buffer_nm, block] = (mask, blocks)
    rows = []
    for ratio in (2, 4, 8):
        error = np.zeros(eps.shape, float)
        for c in range(6):
            difference = fields[c] - reconstruct(fields[c], ratio)
            error += np.abs(difference) ** 2 * (eps if c < 3 else 1)
        uniform_error = float(np.sqrt(error[active].sum() / total))
        for (policy, buffer_nm, block), (fine, blocks) in masks.items():
            if block % ratio:
                continue  # Fine blocks must span an integer number of coarse cells.
            f = float(fine.mean())
            remaining = error[active & ~fine].sum()
            # This ignores transition compatibility and retains exact reference in fine blocks.
            row = dict(
                ratio=ratio,
                policy=policy,
                buffer_nm=buffer_nm,
                block_cells=block,
                fine_fraction=f,
                blocks=blocks,
                uniform_coarse_reconstruction_relative_l2=uniform_error,
                mixed_reconstruction_relative_l2=float(np.sqrt(remaining / total)),
                ideal_memory_reduction=1 / (f + (1 - f) / ratio**3),
                ideal_work_reduction_global_step=1 / (f + (1 - f) / ratio**3),
                ideal_work_reduction_lts=1 / (f + (1 - f) / ratio**4),
                allocated_leaf_cell_fraction_with_one_cell_halo=float(
                    blocks * (block + 2) ** 3 / eps.size + (1 - f) / ratio**3
                ),
                memory_reduction_if_covered_coarse_cells_retained=1
                / (f + 1 / ratio**3),
            )
            rows.append(row)
        print(
            f"ratio {ratio}: uniform reconstruction error {uniform_error:.6g}",
            flush=True,
        )
    result = dict(
        status="complete",
        scope="Offline representation only: exact reference retained in fine blocks; no adaptive evolution, interface, CFL, or GPU-speedup claim",
        input_volume_result_sha256=hashlib.sha256(
            (directory / "result.json").read_bytes()
        ).hexdigest(),
        analysis_code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        sample_shape=meta["sample_shape"],
        sample_nm=meta["sample_nm"],
        excluded_boundary_nm=200,
        field_weight="epsilon*|E|^2+|H|^2 at colocated samples; diagnostic not a conserved discrete energy",
        concentration=concentration,
        rows=rows,
    )
    (out / "result.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    fig, axes_plot = plt.subplots(1, 3, figsize=(14, 4), layout="constrained")
    for ratio in (2, 4, 8):
        for policy, style in [("silicon", "o-"), ("all_material_interfaces", "s--")]:
            selected = [
                r
                for r in rows
                if r["ratio"] == ratio
                and r["policy"] == policy
                and r["block_cells"] == 8
            ]
            label = f"{ratio}x coarser; " + (
                "Si buffer" if policy == "silicon" else "all interfaces"
            )
            axes_plot[0].semilogy(
                [100 * r["fine_fraction"] for r in selected],
                [r["mixed_reconstruction_relative_l2"] for r in selected],
                style,
                label=label,
                markersize=4,
            )
            axes_plot[1].loglog(
                [r["ideal_memory_reduction"] for r in selected],
                [r["mixed_reconstruction_relative_l2"] for r in selected],
                style,
                markersize=4,
            )
    axes_plot[0].set(
        xlabel="Full-domain volume kept fine (%)",
        ylabel="Weighted field reconstruction error",
        title="Exact reference inside hypothetical fine blocks",
    )
    axes_plot[0].legend(fontsize=6)
    axes_plot[1].set(
        xlabel="Ideal storage reduction (no interface overhead)",
        ylabel="Weighted field reconstruction error",
        title="Diagnostic tradeoff; not propagation accuracy",
    )
    axes_plot[1].set_xticks([2, 3, 5, 10], ["2", "3", "5", "10"])
    axes_plot[1].xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    axes_plot[2].plot(
        [100 * r["energy_fraction"] for r in concentration],
        [100 * r["active_volume_fraction"] for r in concentration],
        "o-",
    )
    axes_plot[2].set(
        xlabel="Fraction of diagnostic field energy (%)",
        ylabel="Active volume containing that energy (%)",
        title="Energy concentration is not an error estimator",
    )
    for ax in axes_plot:
        ax.grid(alpha=0.2)
    fig.suptitle("Optimized 3D bend: offline mesh-opportunity study")
    fig.savefig(out / "opportunity.png", dpi=150)
    plt.close(fig)
    print(
        json.dumps(
            {
                "concentration": concentration,
                "best_1pct_representation": sorted(
                    [r for r in rows if r["mixed_reconstruction_relative_l2"] <= 0.01],
                    key=lambda r: -r["ideal_memory_reduction"],
                )[:3],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
