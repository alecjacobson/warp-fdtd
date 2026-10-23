"""Analytic 3D mode convergence and broadband stability for planar coupling."""

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from .planar_interface import PlanarOperator, PlanarStepper


def mode(coords, components, lengths, t, electric):
    kx, ky, kz = 2 * np.pi / lengths[0], np.pi / lengths[1], 2 * np.pi / lengths[2]
    omega = np.sqrt(kx * kx + ky * ky + kz * kz)
    x, y, z = coords.T
    theta = kx * x + kz * z
    if electric:
        fields = np.array(
            [
                kz * np.sin(ky * y) * np.cos(theta),
                np.zeros_like(x),
                -kx * np.sin(ky * y) * np.cos(theta),
            ]
        )
        fields *= np.cos(omega * t)
    else:
        # H = -curl(E_spatial)/omega * sin(omega*t).
        fields = np.array(
            [
                -kx * ky * np.cos(ky * y) * np.cos(theta),
                -(kx * kx + kz * kz) * np.sin(ky * y) * np.sin(theta),
                -kz * ky * np.cos(ky * y) * np.cos(theta),
            ]
        )
        fields *= -np.sin(omega * t) / omega
    return fields[components, np.arange(len(coords))]


def convergence(n, ratio, device, precision, trace="published"):
    op = PlanarOperator(nx=n, ny_coarse=n // 2, nz=n, h=1 / n, ratio=ratio, trace=trace)
    s = PlanarStepper(op, cfl=0.7, device=device, precision=precision)
    e0 = mode(op.ecoords, op.e_components, op.lengths, 0, True)
    h0 = mode(op.hcoords, op.h_components, op.lengths, -s.dt / 2, False)
    s.set_fields(op.project_e(e0), h0)
    steps = int(round(0.4 / s.dt))
    t = steps * s.dt
    initial_energy = op.energy(*s.fields(), s.dt)
    s.run(steps)
    e, h = s.fields()
    expected = mode(op.ecoords, op.e_components, op.lengths, t, True)
    expected_h = mode(op.hcoords, op.h_components, op.lengths, t - s.dt / 2, False)

    def norm(values, weights):
        return np.sqrt(np.dot(values * weights, values))

    # Reference amplitude is the initial field norm: do not divide by a temporal node.
    denom = norm(e0, op.me_full)
    error = norm(op.p @ e - expected, op.me_full) / denom
    projected_error = norm(op.p @ (e - op.project_e(expected)), op.me_full) / denom
    initial_projection = norm(op.p @ op.project_e(e0) - e0, op.me_full) / denom
    near = np.abs(op.ecoords[:, 1] - 0.5) <= 1 / n + 1e-12
    return {
        "coarse_n": n,
        "ratio": ratio,
        "time": t,
        "steps": steps,
        "dt": s.dt,
        "electric_relative_l2": float(error),
        "electric_error_against_projected_exact": float(projected_error),
        "initial_projection_relative_l2": float(initial_projection),
        "interface_layer_error_over_global_norm": float(
            norm((op.p @ e - expected)[near], op.me_full[near]) / denom
        ),
        "magnetic_error_over_initial_electric_norm": float(
            norm(h - expected_h, op.mh) / denom
        ),
        "energy_relative_change": float(op.energy(e, h, s.dt) / initial_energy - 1),
        "normal_B_jump_rms_over_initial_E_rms": float(
            np.sqrt(np.mean(op.normal_magnetic_jump(h) ** 2))
            / (denom / np.sqrt(np.sum(op.me_full)))
        ),
        "operator_audit": op.audit(),
    }


def stability(steps, cfl, device, precision, contrast, trace="published"):
    op = PlanarOperator(
        nx=2,
        ny_coarse=1,
        nz=2,
        h=0.5,
        ratio=3,
        epsilon_coarse=contrast,
        trace=trace,
        epsilon_fine=lambda p: np.where(p[:, 0] < 0.5, contrast, 1.0),
    )
    s = PlanarStepper(op, cfl=cfl, device=device, precision=precision)
    rng = np.random.default_rng(109)
    # A broadband electric state; magnetic initial state is a discrete curl.
    e = rng.normal(size=op.me.shape[0])
    h = op.h_update @ rng.normal(size=op.me.shape[0])
    h /= np.sqrt(np.dot(h * op.mh, h))
    s.set_fields(e, h)
    e, h = s.fields()
    energy0 = op.energy(e, h, s.dt)
    div0 = op.magnetic_divergence(h)
    checkpoints = []
    done = 0
    for target in np.unique(np.linspace(0, steps, 11, dtype=int))[1:]:
        s.run(int(target) - done)
        done = int(target)
        en, hn = s.fields()
        drift = abs(op.energy(en, hn, s.dt) / energy0 - 1)
        # Fixed L=1, unweighted relative Euclidean norm reported explicitly.
        div = np.linalg.norm(op.magnetic_divergence(hn) - div0) / np.linalg.norm(h)
        checkpoints.append(
            {
                "step": done,
                "modified_energy_relative_drift": float(drift),
                "local_magnetic_divergence_change": float(div),
            }
        )
    return {
        "cfl": cfl,
        "contrast": contrast,
        "steps": steps,
        "checkpoints": checkpoints,
        "scope": "Local block div B only; composite interface div D/normal flux not yet certified",
        "pass_restricted_energy_and_local_B": all(
            c["modified_energy_relative_drift"] <= 1e-4
            and c["local_magnetic_divergence_change"] <= 1e-4
            for c in checkpoints
        ),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--output", type=Path, default=Path("results/warp/planar_interface_cpu")
    )
    p.add_argument("--device", default="cpu")
    p.add_argument("--precision", choices=("float32", "float64"), default="float64")
    p.add_argument("--steps", type=int, default=10000)
    p.add_argument("--trace", choices=("published", "area"), default="published")
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    waves = []
    for ratio in (1, 3):
        rows = []
        for n in (4, 8, 16):
            row = convergence(n, ratio, a.device, a.precision, a.trace)
            rows.append(row)
            print(json.dumps(row), flush=True)
        for previous, row in zip(rows, rows[1:]):
            row["observed_order"] = float(
                np.log(previous["electric_relative_l2"] / row["electric_relative_l2"])
                / np.log(2)
            )
        waves.extend(rows)
    stability_rows = []
    for cfl, contrast in ((0.3, 1.0), (0.9, 12.25), (0.99, 100.0)):
        row = stability(a.steps, cfl, a.device, a.precision, contrast, a.trace)
        stability_rows.append(row)
        print(json.dumps(row), flush=True)
    sources = {
        f"warp_fdtd/{name}": hashlib.sha256(
            Path(__file__).with_name(name).read_bytes()
        ).hexdigest()
        for name in ("planar_interface.py", "planar_validation.py")
    }
    result = {
        "status": "complete",
        "scope": "Planar 3D interface prototype; no local patch edges/corners, ports or CPML",
        "trace": a.trace,
        "device": a.device,
        "precision": a.precision,
        "wave_convergence": waves,
        "stability": stability_rows,
        "source_sha256": sources,
        "elapsed_seconds": time.perf_counter() - start,
        "convergence_gate_passed": all(
            row["observed_order"] >= 1.8 for row in waves if "observed_order" in row
        ),
        "restricted_stability_gate_passed": all(
            row["pass_restricted_energy_and_local_B"] for row in stability_rows
        ),
        "composite_gauss_certified": False,
        "production_ready": False,
    }
    (a.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    if (
        not result["convergence_gate_passed"]
        or not result["restricted_stability_gate_passed"]
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
