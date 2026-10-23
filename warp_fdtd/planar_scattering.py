"""Normal-incidence vacuum pulse crossing a planar refinement boundary.

Full 3D operator with planar symmetry, not an oblique/scatterer certification.
Exact physical reflection is zero. E/H characteristic separation is time and
space centered; uniform controls report the measured numerical floor.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .planar_interface import PlanarOperator, PlanarStepper


def pulse(y):
    delta = y - 1.5
    return np.exp(-0.5 * (delta / 0.3) ** 2) * np.cos(2 * np.pi * delta)


def run(cells, ratio, trace, device):
    h = 1 / cells
    op = PlanarOperator(nx=2, ny_coarse=3 * cells, nz=2, h=h, ratio=ratio, trace=trace)
    s = PlanarStepper(op, cfl=0.7, device=device, precision="float64")
    initial_e = pulse(op.ecoords[:, 1]) * (op.e_components == 2)
    initial_h = pulse(op.hcoords[:, 1] + s.dt / 2) * (op.h_components == 0)
    s.set_fields(op.project_e(initial_e), initial_h)
    steps = int(round(2.7 / s.dt))
    s.run(steps)
    e, hfield = s.fields()
    physical_e = op.p @ e
    # H^{n} = (H^{n-1/2}+H^{n+1/2})/2, using one passive half-step prediction.
    centered_h = hfield + 0.5 * s.dt * (op.h_update @ e)
    g = op.coarse
    ez = physical_e[g.eids[2]][:, 1:-1, :]
    hx = centered_h[g.hids[0]]
    hx = 0.5 * (hx[:, :-1, :] + hx[:, 1:, :])
    y = np.arange(1, g.shape[1]) * g.h
    mask = (y >= 0.5) & (y <= 2.7)
    backward = 0.5 * (ez[:, mask, :] - hx[:, mask, :])
    reflected = float(np.sum(backward**2) * g.h**3)
    cross_area = (g.shape[0] * g.h) * (g.shape[2] * g.h)
    exact_incident = (
        cross_area
        * np.sqrt(np.pi)
        * 0.3
        * 0.5
        * (1 + np.exp(-((2 * np.pi * 0.3) ** 2)))
    )
    return {
        "coarse_cells_per_wavelength": cells,
        "ratio": ratio,
        "trace": trace,
        "time": steps * s.dt,
        "steps": steps,
        "reflected_energy_fraction": reflected / exact_incident,
        "reflection_gate_1e_5_passed": bool(reflected / exact_incident <= 1e-5),
        "exact_reflection": 0.0,
        "scope": "Normal incidence, planar symmetry, vacuum, float64",
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--output", type=Path, default=Path("results/warp/planar_scattering_cpu")
    )
    p.add_argument("--device", default="cpu")
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    cases = []
    for cells in (20, 40, 80):
        for ratio, trace in ((1, "published"), (3, "published"), (3, "area")):
            row = run(cells, ratio, trace, a.device)
            cases.append(row)
            print(json.dumps(row), flush=True)
    for cells in (20, 40, 80):
        group = [r for r in cases if r["coarse_cells_per_wavelength"] == cells]
        floor = group[0]["reflected_energy_fraction"]
        for row in group:
            row["uniform_control_reflection_floor"] = floor
    hashes = {
        f"warp_fdtd/{n}": hashlib.sha256(
            Path(__file__).with_name(n).read_bytes()
        ).hexdigest()
        for n in ("planar_scattering.py", "planar_interface.py")
    }
    result = {
        "status": "complete",
        "device": a.device,
        "cases": cases,
        "source_sha256": hashes,
        "all_reflection_gates_passed": all(
            r["reflection_gate_1e_5_passed"] for r in cases
        ),
        "scope": "Isolated normal-incidence interface test only; no production certification",
    }
    (a.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    if not result["all_reflection_gates_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
