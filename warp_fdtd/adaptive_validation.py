"""Analytic and energy checks for the experimental rectilinear control backend."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import warp as wp

from .rectilinear import RectilinearPeriodic, curl_numpy, divergence_numpy


def analytic(s, time_e):
    p = np.array([1.0, -1.0, 0.0]) / np.sqrt(2)
    q = np.cross(np.ones(3) / np.sqrt(3), p)
    omega = 2 * np.pi * np.sqrt(3)
    result = []
    for c in range(6):
        phase = 2 * np.pi * sum(s.coordinates(c))
        result.append(
            p[c] * np.cos(phase) * np.cos(omega * time_e)
            if c < 3
            else q[c - 3] * np.sin(phase) * np.sin(omega * (time_e + 0.5 * s.dt))
        )
    return np.array(result[:3]), np.array(result[3:])


def tiny_operator_check():
    edges = [
        np.array([0, 0.18, 0.55, 1.0]),
        np.array([0, 0.22, 0.61, 1.0]),
        np.array([0, 0.3, 0.65, 1.0]),
    ]
    # Build the CPU curl operators independently of Warp kernels.
    widths = [np.diff(x) for x in edges]
    dual = [0.5 * (w + np.roll(w, 1)) for w in widths]
    shape = (3, 3, 3)
    n = 3 * 27
    basis = np.eye(n).reshape(n, 3, *shape)
    a = np.stack([curl_numpy(v, dual, False).ravel() for v in basis], axis=1)
    b = np.stack([curl_numpy(v, widths, True).ravel() for v in basis], axis=1)
    volumes = []
    for electric in (True, False):
        vv = []
        for c in range(3):
            v = np.ones(shape)
            for axis in range(3):
                sh = [1, 1, 1]
                sh[axis] = 3
                w = widths[axis] if ((axis == c) == electric) else dual[axis]
                v *= w.reshape(sh)
            vv.append(v)
        volumes.append(np.array(vv).ravel())
    me, mh = map(np.diag, volumes)
    residual = float(np.linalg.norm(me @ a - b.T @ mh) / np.linalg.norm(me @ a))
    rows = []
    for cfl in np.linspace(0.1, 0.99, 12):
        dt = float(cfl / np.sqrt(sum(1 / w.min() ** 2 for w in widths)))
        eye = np.eye(n)
        update = np.block([[eye, dt * a], [-dt * b, eye - dt**2 * b @ a]])
        q = np.block([[me, 0.5 * dt * me @ a], [0.5 * dt * a.T @ me, mh]])
        drift = float(np.linalg.norm(update.T @ q @ update - q) / np.linalg.norm(q))
        minimum = float(np.linalg.eigvalsh(q).min())
        radius = float(np.abs(np.linalg.eigvals(update)).max())
        rows.append(
            dict(
                cfl=float(cfl),
                invariant_residual=drift,
                minimum_energy_eigenvalue=minimum,
                spectral_radius=radius,
            )
        )
    passed = residual < 1e-11 and all(
        r["invariant_residual"] < 1e-11
        and r["minimum_energy_eigenvalue"] > 0
        and r["spectral_radius"] < 1 + 1e-10
        for r in rows
    )
    return dict(
        weighted_curl_adjoint_residual=residual,
        rows=rows,
        passed=passed,
        scope="3x3x3 periodic nonuniform lossless operator; not a block-interface or CPML proof",
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", default="results/warp/adaptive_validation")
    p.add_argument("--precision", choices=("float32", "float64"), default="float32")
    p.add_argument("--compensated", action="store_true")
    p.add_argument(
        "--flux",
        action="store_true",
        help="Experimental shared-edge two-float flux storage",
    )
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--long-steps", type=int, default=1000000)
    args = p.parse_args()
    if args.flux and (args.precision != "float32" or args.compensated):
        p.error("--flux is a separate float32 control")
    if args.long_steps < 1:
        p.error("--long-steps must be positive")
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    def solver(edges, cfl):
        if args.flux:
            from .flux_rectilinear import FluxRectilinearPeriodic

            return FluxRectilinearPeriodic(edges, cfl=cfl, device=args.device)
        return RectilinearPeriodic(
            edges,
            cfl=cfl,
            precision=args.precision,
            compensated=args.compensated,
            device=args.device,
        )

    rows = []
    for graded in (False, True):
        for n in (8, 16, 32, 64):
            u = np.linspace(0, 1, n + 1)
            edges = [
                u + 0.45 / (2 * np.pi) * np.sin(2 * np.pi * u) if graded else u
                for _ in range(3)
            ]
            s = solver(edges, cfl=0.5)
            initial = analytic(s, 0)
            s.set_fields(*initial)
            duration = 1.75 / np.sqrt(3)
            steps = round(duration / s.dt)
            s.compile((256, steps % 256))
            started = time.perf_counter()
            s.run(steps)
            elapsed = time.perf_counter() - started
            e, h = s.fields()
            ee, hh = analytic(s, steps * s.dt)
            error = float(
                np.sqrt(
                    (
                        np.sum(s.mass_e * (e - ee) ** 2)
                        + np.sum(s.mass_h * (h - hh) ** 2)
                    )
                    / (np.sum(s.mass_e * ee**2) + np.sum(s.mass_h * hh**2))
                )
            )
            drift = abs(s.energy(e, h) / s.energy(*initial) - 1)
            div0 = divergence_numpy(initial[0], s.dual, False)
            div1 = divergence_numpy(e, s.dual, False)
            gauss = float(
                np.linalg.norm(div1 - div0)
                / np.sqrt(sum(np.sum(a.astype(float) ** 2) for a in initial))
            )
            row = dict(
                graded=graded,
                n=n,
                steps=steps,
                dt=s.dt,
                field_relative_l2=error,
                energy_relative_drift=drift,
                gauss_change_normalized=gauss,
                seconds=elapsed,
            )
            print(json.dumps(row), flush=True)
            rows.append(row)
    orders = {
        str(g): float(
            np.polyfit(
                np.log([1 / r["n"] for r in rows if r["graded"] == g]),
                np.log([r["field_relative_l2"] for r in rows if r["graded"] == g]),
                1,
            )[0]
        )
        for g in (False, True)
    }
    # Broadband high-frequency test, includes discrete Gauss-compatible curls.
    n = 8
    u = np.linspace(0, 1, n + 1)
    edges = [u + 0.45 / (2 * np.pi) * np.sin(2 * np.pi * u)] * 3
    s = solver(edges, cfl=0.9)
    rng = np.random.default_rng(51)
    e = curl_numpy(rng.normal(size=(3, n, n, n)), s.dual, False)
    h = curl_numpy(rng.normal(size=(3, n, n, n)), s.primal, True)
    s.set_fields(e, h)
    initial = s.fields()
    energy = s.energy(*initial)
    checkpoints = []
    done = 0
    # Ten checkpoints capture drift, rather than only comparing two endpoints.
    while done < args.long_steps:
        count = min(max(1, args.long_steps // 10), args.long_steps - done)
        s.run(count)
        done += count
        ef, hf = s.fields()
        checkpoints.append(
            dict(
                step=done,
                energy_relative_drift=abs(s.energy(ef, hf) / energy - 1),
                gauss_e=float(
                    np.linalg.norm(divergence_numpy(ef, s.dual, False))
                    / np.linalg.norm(ef)
                ),
                gauss_h=float(
                    np.linalg.norm(divergence_numpy(hf, s.primal, True))
                    / np.linalg.norm(hf)
                ),
            )
        )
        print("long", json.dumps(checkpoints[-1]), flush=True)
    operator = tiny_operator_check()
    passed = (
        operator["passed"]
        and min(orders.values()) >= 1.8
        and max(r["energy_relative_drift"] for r in rows) < 1e-4
        and max(r["energy_relative_drift"] for r in checkpoints) < 1e-4
        and max(r["gauss_change_normalized"] for r in rows) < 1e-4
        and max(max(r["gauss_e"], r["gauss_h"]) for r in checkpoints) < 1e-4
    )
    result = dict(
        status="complete",
        passed=passed,
        scope="Periodic lossless rectilinear verification only; no dielectric scattering/CPML/block refinement/flagship acceptance",
        device=s.device.name,
        precision=args.precision,
        compensated=args.compensated,
        flux=args.flux,
        gauss_normalization="Unit domain length; analytic change divided by initial combined E/H norm to avoid standing-wave nodes; broadband absolute residual divided by each instantaneous field norm",
        seed=51,
        configuration={
            "graded_edge_formula": "u + 0.45/(2*pi)*sin(2*pi*u)",
            "analytic_cfl": 0.5,
            "long_time_cfl": 0.9,
            "long_steps": args.long_steps,
            "normalized_domain_lengths": [1, 1, 1],
        },
        acceptance={
            "minimum_smooth_order": 1.8,
            "maximum_energy_relative_drift": 1e-4,
            "maximum_normalized_gauss_residual": 1e-4,
        },
        code_sha256={
            name: hashlib.sha256(Path("warp_fdtd", name).read_bytes()).hexdigest()
            for name in (
                "rectilinear.py",
                "adaptive_validation.py",
                "flux_rectilinear.py",
            )
        },
        warp_version=wp.__version__,
        rows=rows,
        orders=orders,
        long_time=checkpoints,
        operator=operator,
    )
    (out / "result.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    if not passed:
        raise AssertionError("Rectilinear validation failed; see result.json")


if __name__ == "__main__":
    main()
