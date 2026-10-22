"""Normal-incidence dielectric pulse scattering against Fresnel energy fractions.

A 3D periodic operator with planar symmetry provides an analytic material test.
The pulse and its reflected/transmitted parts are separated from both periodic
boundaries at measurement time. This is not a slab resonance, oblique-interface,
CPML, or full 3D sphere test.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import warp as wp

from .flux_rectilinear import FluxRectilinearPeriodic
from .rectilinear import RectilinearPeriodic, curl_numpy


def pulse(x):
    return np.exp(-0.5 * ((x - 0.2) / 0.025) ** 2) * np.cos(
        2 * np.pi * (x - 0.2) / 0.05
    )


def evaluate(n, contrast, backend, device, cfl):
    edges = [np.linspace(0, 1, n + 1), np.linspace(0, 0.08, 5), np.linspace(0, 0.08, 5)]
    eps = np.ones((3, n, 4, 4))
    # At the planar interface, tangential E owns half of each dielectric's
    # dual volume. Normal E is on a primal edge wholly in one dielectric.
    eps[:, n // 2 :, :, :] = contrast
    eps[1:, n // 2, :, :] = 0.5 * (1 + contrast)
    cls = FluxRectilinearPeriodic if backend == "flux" else RectilinearPeriodic
    kwargs = {} if backend == "flux" else dict(precision=backend)
    s = cls(edges, eps, cfl=cfl, device=device, **kwargs)
    e, h = np.zeros_like(eps), np.zeros_like(eps)
    e[2] = pulse(s.coordinates(2)[0])
    # E^0, H^(1/2), right-going wave: H_y=-E_z in the incident medium.
    h[1] = -pulse(s.coordinates(4)[0] - 0.5 * s.dt)
    s.set_fields(e, h)
    initial_energy = s.energy(*s.fields())
    steps = round(0.5 / s.dt)
    if s.device.is_cuda:
        s.run(steps)
    else:
        for _ in range(steps):
            s.step()
        wp.synchronize_device(s.device)
    ef, hf = s.fields()
    # Decompose the same modified-energy invariant into separated spatial
    # regions. Residual energy close to the interface/boundaries is reported.
    cross = ef * curl_numpy(hf, s.dual, False) * (s.mass_e / s.epsilon)
    density = 0.5 * (
        s.mass_e * ef.astype(float) ** 2
        + s.mass_h * hf.astype(float) ** 2
        + s.dt * cross
    )
    left = float(np.sum(density[:, : n // 2])) / initial_energy
    right = float(np.sum(density[:, n // 2 :])) / initial_energy
    x = edges[0][:-1]
    # At t~0.5, reflected center~0.3 and transmitted center~0.5+0.2/sqrt(eps).
    # Exclude a deliberately narrow strip; slow high-contrast packets are close
    # to the interface, so this is an observation rather than a passing gate.
    edge_mask = (x < 0.05) | (x > 0.95)
    boundary_fraction = float(np.sum(np.abs(density[:, edge_mask]))) / initial_energy
    exact_r = ((1 - np.sqrt(contrast)) / (1 + np.sqrt(contrast))) ** 2
    return dict(
        n=n,
        shape=list(s.shape),
        steps=steps,
        dt=s.dt,
        contrast=contrast,
        cfl=cfl,
        reflected_fraction=left,
        transmitted_fraction=right,
        analytic_reflection=float(exact_r),
        analytic_transmission=float(1 - exact_r),
        reflection_absolute_error=abs(left - exact_r),
        transmission_absolute_error=abs(right - (1 - exact_r)),
        energy_relative_drift=abs(left + right - 1),
        boundary_energy_fraction=boundary_fraction,
        cells_per_shortest_carrier_wavelength=0.05 * n / np.sqrt(contrast),
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--backend", choices=["flux", "float32", "float64"], default="flux")
    p.add_argument("--sizes", type=int, nargs="+", default=[512, 1024, 2048, 4096])
    p.add_argument("--contrasts", type=float, nargs="+", default=[2.25, 12.25])
    p.add_argument("--cfl", type=float, default=0.7)
    p.add_argument("--output", type=Path, default=Path("results/warp/fresnel_flux"))
    args = p.parse_args()
    if (
        len(args.sizes) < 3
        or any(n < 64 or n % 2 for n in args.sizes)
        or sorted(set(args.sizes)) != args.sizes
    ):
        p.error("At least three strictly increasing even sizes >=64 required")
    if any(not np.isfinite(c) or c <= 1 for c in args.contrasts):
        p.error("Finite dielectric contrasts >1 required")
    rows = []
    for contrast in args.contrasts:
        for n in args.sizes:
            row = evaluate(n, contrast, args.backend, args.device, args.cfl)
            rows.append(row)
            print(json.dumps(row), flush=True)
    # Estimate the slope on the three finest levels; not all coarse meshes
    # resolve the pulse spectrum inside the high-index material.
    orders = {
        str(c): float(
            np.polyfit(
                np.log([1 / r["n"] for r in rows if r["contrast"] == c][-3:]),
                np.log(
                    [
                        r["reflection_absolute_error"]
                        for r in rows
                        if r["contrast"] == c
                    ][-3:]
                ),
                1,
            )[0]
        )
        for c in args.contrasts
    }
    passed = min(orders.values()) >= 1.8 and all(
        r["energy_relative_drift"] < 1e-4 and r["boundary_energy_fraction"] < 1e-6
        for r in rows
    )
    result = dict(
        status="complete",
        passed=passed,
        backend=args.backend,
        device=wp.get_device(args.device).name,
        warp_version=wp.__version__,
        scope="Planar normal-incidence half-space pulse in the 3D periodic operator only.",
        domain_lengths=[1, 0.08, 0.08],
        interface_x=0.5,
        initial_pulse_center=0.2,
        pulse_sigma=0.025,
        carrier_wavelength=0.05,
        target_observation_time=0.5,
        material_rule="Arithmetic dual-volume average for tangential E at the interface; normal E uses its containing medium.",
        acceptance=dict(
            minimum_fine_three_level_order=1.8,
            maximum_energy_drift=1e-4,
            maximum_boundary_energy_fraction=1e-6,
        ),
        code_sha256={
            n: hashlib.sha256(Path("warp_fdtd", n).read_bytes()).hexdigest()
            for n in ["rectilinear.py", "flux_rectilinear.py", "fresnel_validation.py"]
        },
        orders=orders,
        rows=rows,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "result.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    if not passed:
        raise AssertionError("Fresnel test failed; see result.json")


if __name__ == "__main__":
    main()
