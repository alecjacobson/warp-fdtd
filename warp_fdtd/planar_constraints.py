"""Physical composite charge audit for the planar coarse–fine prototype.

The interface dual control volume has coarse tangential width h and extends
h/2 into the coarse side and hf/2 into the fine side. Its top is partitioned
into r*r fine dual faces. Each fine tangential side contains r half-faces.
This is an integrated Gauss flux, not a fitted invariant from a nullspace.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from .planar_interface import PlanarOperator, PlanarStepper


def electric_pillboxes(op):
    c, f, r = op.coarse, op.fine, op.ratio
    nx, ny, nz = c.shape
    rows, cols, values = [], [], []
    epsilon = np.concatenate((c.me / op._e_volumes(c), f.me / op._e_volumes(f)))

    def add(row, g, offset, component, i, j, k, area):
        index = int(g.eids[component][i % g.shape[0], j, k % g.shape[2]]) + offset
        rows.append(row)
        cols.append(index)
        values.append(area * epsilon[index])

    for i, k in np.ndindex(nx, nz):
        row = i * nz + k
        add(row, c, 0, 1, i, ny - 1, k, -(c.h**2))
        for shift, sign in ((0, 1), (-1, -1)):
            add(row, c, 0, 0, i + shift, ny, k, sign * c.h**2 / 2)
            add(row, c, 0, 2, i, ny, k + shift, sign * c.h**2 / 2)
        for a, b in np.ndindex(r, r):
            add(row, f, len(c.me), 1, i * r + a - r // 2, 0, k * r + b - r // 2, f.h**2)
        for a in range(r):
            for shift, sign in ((r // 2, 1), (-(r // 2) - 1, -1)):
                add(
                    row,
                    f,
                    len(c.me),
                    0,
                    i * r + shift,
                    0,
                    k * r + a - r // 2,
                    sign * f.h**2 / 2,
                )
                add(
                    row,
                    f,
                    len(c.me),
                    2,
                    i * r + a - r // 2,
                    0,
                    k * r + shift,
                    sign * f.h**2 / 2,
                )
    return sp.coo_matrix((values, (rows, cols)), shape=(nx * nz, len(epsilon))).tocsr()


def normal_magnetic_fluxes(op):
    """Coarse-face flux minus the sum of its geometrically matching fine faces."""
    c, f, r = op.coarse, op.fine, op.ratio
    rows, cols, values = [], [], []
    for i, k in np.ndindex(c.shape[0], c.shape[2]):
        row = i * c.shape[2] + k
        rows.append(row)
        cols.append(int(c.hids[1][i, -1, k]))
        values.append(c.h**2)
        for a, b in np.ndindex(r, r):
            rows.append(row)
            cols.append(len(c.mh) + int(f.hids[1][i * r + a, 0, k * r + b]))
            values.append(-(f.h**2))
    return sp.coo_matrix(
        (values, (rows, cols)), shape=(c.shape[0] * c.shape[2], len(op.mh))
    ).tocsr()


def magnetic_gradient_probe(op):
    """Can a curl-free fine trace generate coarse normal B?

    Only applicable to the area trace, whose fine tangential DOFs are free.
    Use unequal x/z Fourier modes to avoid accidental symmetry cancellation.
    This tests commuting traces; it does not infer a physical div B from
    differences between a point sample and a face quadrature.
    """
    if op.trace not in ("area", "circulation"):
        raise ValueError("Probe requires independent area-trace fine DOFs")
    c, f = op.coarse, op.fine
    nx, _, nz = f.shape
    x, z = np.meshgrid(np.arange(nx) * f.h, np.arange(nz) * f.h, indexing="ij")
    phi = np.cos(2 * np.pi * x / op.lengths[0]) * np.cos(4 * np.pi * z / op.lengths[2])
    derivatives = {
        0: (np.roll(phi, -1, axis=0) - phi) / f.h,
        2: (np.roll(phi, -1, axis=1) - phi) / f.h,
    }
    e = np.zeros(op.me.shape[0])
    for component in (0, 2):
        for i, k in np.ndindex(nx, nz):
            row = int(f.eids[component][i, 0, k]) + len(c.me)
            start, end = op.p.indptr[row : row + 2]
            if end - start != 1 or op.p.data[start] != 1:
                raise ValueError("Unexpected fine trace map")
            e[op.p.indices[start]] = derivatives[component][i, k]
    derivative = op.h_update @ e
    coarse = derivative[c.hids[1][:, -1, :]]
    fine = derivative[len(c.mh) + f.hids[1][:, 0, :]]
    return {
        "fine_curl_maxabs": float(np.max(np.abs(fine))),
        "coarse_curl_maxabs": float(np.max(np.abs(coarse))),
        "commuting_trace_gate_passed": bool(np.max(np.abs(coarse)) <= 1e-11),
    }


def audit(trace, ratio, contrast):
    op = PlanarOperator(
        nx=4,
        ny_coarse=2,
        nz=4,
        h=0.25,
        ratio=ratio,
        trace=trace,
        epsilon_coarse=contrast,
        epsilon_fine=lambda p: np.where(p[:, 0] < 0.5, contrast, 1.0),
    )
    flux = electric_pillboxes(op)
    rate = flux @ op.p @ op.e_update
    maxabs = float(np.max(np.abs(rate.data), initial=0))
    magnetic_rate = normal_magnetic_fluxes(op) @ op.h_update
    magnetic_maxabs = float(np.max(np.abs(magnetic_rate.data), initial=0))
    result = {
        "trace": trace,
        "ratio": ratio,
        "contrast": contrast,
        "integrated_charge_rate_coefficient_maxabs": maxabs,
        "integrated_charge_rate_frobenius": float(sp.linalg.norm(rate)),
        "electric_pillbox_gate_passed": maxabs <= 1e-11,
        "normal_B_flux_rate_coefficient_maxabs": magnetic_maxabs,
        "normal_B_flux_gate_passed": magnetic_maxabs <= 1e-11,
    }
    if trace in ("area", "circulation"):
        result["magnetic_gradient_probe"] = magnetic_gradient_probe(op)
    return result


def evolution(steps, device, trace="circulation"):
    op = PlanarOperator(
        nx=4,
        ny_coarse=2,
        nz=4,
        h=0.25,
        ratio=3,
        trace=trace,
        epsilon_coarse=12.25,
        epsilon_fine=lambda p: 1 + 99 * (p[:, 0] < 0.5),
    )
    s = PlanarStepper(op, cfl=0.99, device=device, precision="float64")
    rng = np.random.default_rng(135)
    s.set_fields(rng.normal(size=op.me.shape[0]), rng.normal(size=len(op.mh)))
    e, h = s.fields()
    q = electric_pillboxes(op) @ op.p
    initial = q @ e
    bflux = normal_magnetic_fluxes(op)
    initial_bflux = bflux @ h
    s.run(steps)
    en, hn = s.fields()
    # Integrated charge / physical pillbox volume, normalized by initial D RMS.
    volume = op.coarse.h**2 * (op.coarse.h + op.fine.h) / 2
    eps = np.concatenate(
        (op.coarse.me / op._e_volumes(op.coarse), op.fine.me / op._e_volumes(op.fine))
    )
    scale = np.sqrt(np.mean((eps * (op.p @ e)) ** 2))
    return {
        "trace": trace,
        "steps": steps,
        "device": device,
        "precision": "float64",
        "electric_pillbox_divergence_change_rms_L1": float(
            np.sqrt(np.mean(((q @ en - initial) / volume) ** 2)) / scale
        ),
        "normal_B_flux_change_rms_over_H_rms": float(
            np.sqrt(np.mean(((bflux @ hn - initial_bflux) / op.coarse.h**2) ** 2))
            / np.sqrt(np.mean(h * h))
        ),
        "scope": "Planar composite charge and matching normal B flux; no closed patch certification",
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default="cpu")
    p.add_argument("--steps", type=int, default=10000)
    p.add_argument(
        "--output", type=Path, default=Path("results/warp/planar_constraints_cpu")
    )
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    rows = [
        audit(trace, ratio, contrast)
        for ratio in (1, 3, 5)
        for trace in ("published", "area", "circulation")
        for contrast in (1.0, 12.25, 100.0)
    ]
    result = {
        "status": "complete",
        "scope": "Independent physical pillbox and commuting-trace diagnostics",
        "operator_audits": rows,
        "evolution": evolution(a.steps, a.device),
        "source_sha256": {
            f"warp_fdtd/{n}": hashlib.sha256(
                Path(__file__).with_name(n).read_bytes()
            ).hexdigest()
            for n in ("planar_constraints.py", "planar_interface.py")
        },
        "composite_constraints_certified": False,
    }
    (a.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)
    # Known negative controls are retained. Never call these a passing full audit.
    if any(
        not row["electric_pillbox_gate_passed"]
        or not row.get("magnetic_gradient_probe", {}).get(
            "commuting_trace_gate_passed", True
        )
        for row in rows
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
