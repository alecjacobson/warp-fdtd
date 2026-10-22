"""Broadband Fabry–Perot energy test for a finite dielectric slab.

The analytic Gaussian-weighted transmission uses a convergent Poisson-kernel
series, independently checked against direct spectral quadrature in tests.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import warp as wp

from .flux_rectilinear import FluxRectilinearPeriodic
from .rectilinear import RectilinearPeriodic, curl_numpy


def analytic_transmission(
    contrast, thickness=0.02, sigma=0.06, wavelength=0.05, terms=64
):
    n = np.sqrt(contrast)
    rho = ((1 - n) / (1 + n)) ** 2
    k0 = 2 * np.pi / wavelength
    dc = np.exp(-((sigma * k0) ** 2))
    m = np.arange(1, terms + 1, dtype=float)
    delay = 2 * n * thickness * m
    coherence = (
        np.exp(-(delay**2) / (4 * sigma**2)) * (np.cos(k0 * delay) + dc) / (1 + dc)
    )
    value = (1 - rho) / (1 + rho) * (1 + 2 * np.sum(rho**m * coherence))
    # Geometric tail bound also holds without Gaussian damping.
    bound = 2 * rho ** (terms + 1) / (1 + rho)
    return float(value), float(bound)


def evaluate(n, contrast, backend, device):
    edges = [np.linspace(0, 3, n + 1), np.linspace(0, 0.08, 5), np.linspace(0, 0.08, 5)]
    lo, hi = round(n * 0.4), round(n * 1.22 / 3)
    eps = np.ones((3, n, 4, 4))
    eps[:, lo:hi] = contrast
    eps[1:, lo] = eps[1:, hi] = 0.5 * (1 + contrast)
    cls = FluxRectilinearPeriodic if backend == "flux" else RectilinearPeriodic
    kwargs = {} if backend == "flux" else dict(precision=backend)
    s = cls(edges, eps, cfl=0.7, device=device, **kwargs)

    def pulse(x):
        return np.exp(-0.5 * ((x - 0.5) / 0.06) ** 2) * np.cos(
            2 * np.pi * (x - 0.5) / 0.05
        )

    e, h = np.zeros_like(eps), np.zeros_like(eps)
    e[2] = pulse(s.coordinates(2)[0])
    h[1] = -pulse(s.coordinates(4)[0] - 0.5 * s.dt)
    s.set_fields(e, h)
    initial = s.energy(*s.fields())
    steps = round(1.6 / s.dt)
    if s.device.is_cuda:
        s.run(steps)
    else:
        for _ in range(steps):
            s.step()
        wp.synchronize_device(s.device)
    ef, hf = s.fields()
    cross = ef * curl_numpy(hf, s.dual, False) * (s.mass_e / s.epsilon)
    density = (
        0.5
        * (
            s.mass_e * ef.astype(float) ** 2
            + s.mass_h * hf.astype(float) ** 2
            + s.dt * cross
        )
        / initial
    )
    reflection = float(np.sum(density[:, :lo]))
    transmission = float(np.sum(density[:, hi:]))
    remaining = float(np.sum(np.abs(density[:, lo:hi])))
    x = edges[0][:-1]
    boundary = float(np.sum(np.abs(density[:, (x < 0.05) | (x > 2.95)])))
    exact, bound = analytic_transmission(contrast)
    return dict(
        n=n,
        shape=list(s.shape),
        steps=steps,
        dt=s.dt,
        contrast=contrast,
        reflection=reflection,
        transmission=transmission,
        analytic_transmission=exact,
        analytic_series_tail_bound=bound,
        reflection_absolute_error=abs(reflection - (1 - exact)),
        transmission_absolute_error=abs(transmission - exact),
        remaining_slab_energy_fraction=remaining,
        boundary_energy_fraction=boundary,
        energy_relative_drift=abs(float(np.sum(density)) - 1),
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--backend", choices=["flux", "float32", "float64"], default="flux")
    p.add_argument("--sizes", type=int, nargs="+", default=[1500, 3000, 6000, 12000])
    p.add_argument("--output", type=Path, default=Path("results/warp/slab_flux"))
    args = p.parse_args()
    if (
        len(args.sizes) < 3
        or sorted(set(args.sizes)) != args.sizes
        or any(n < 150 or n % 150 for n in args.sizes)
    ):
        p.error("Three or more increasing sizes, multiples of 150, required")
    rows = []
    for contrast in (2.25, 12.25):
        for n in args.sizes:
            row = evaluate(n, contrast, args.backend, args.device)
            rows.append(row)
            print(json.dumps(row), flush=True)
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
        for c in (2.25, 12.25)
    }
    passed = min(orders.values()) >= 1.8 and all(
        r["energy_relative_drift"] < 1e-4
        and r["boundary_energy_fraction"] < 1e-6
        and r["remaining_slab_energy_fraction"] < 1e-5
        for r in rows
    )
    result = dict(
        status="complete",
        passed=passed,
        backend=args.backend,
        device=wp.get_device(args.device).name,
        warp_version=wp.__version__,
        scope="Gaussian-spectrum integrated normal-incidence slab scattering; not a resolved frequency sweep, oblique test, CPML or Mie validation.",
        domain_lengths=[3, 0.08, 0.08],
        slab_bounds=[1.2, 1.22],
        pulse_center=0.5,
        pulse_sigma=0.06,
        carrier_wavelength=0.05,
        target_observation_time=1.6,
        cfl=0.7,
        acceptance=dict(
            minimum_fine_three_level_order=1.8,
            maximum_energy_drift=1e-4,
            maximum_boundary_energy_fraction=1e-6,
            maximum_remaining_slab_energy_fraction=1e-5,
        ),
        code_sha256={
            n: hashlib.sha256(Path("warp_fdtd", n).read_bytes()).hexdigest()
            for n in ["rectilinear.py", "flux_rectilinear.py", "slab_validation.py"]
        },
        orders=orders,
        rows=rows,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "result.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    if not passed:
        raise AssertionError("Slab validation failed; see result.json")


if __name__ == "__main__":
    main()
