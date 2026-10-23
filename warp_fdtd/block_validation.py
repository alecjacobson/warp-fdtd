"""Analytic 3D cavity waves through a closed locally refined patch."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .block_mesh import Block, BlockMesh
from .block_operator import BlockOperator
from .planar_interface import PlanarStepper


def mode(coords, components, lengths, t, electric):
    wave = np.pi * np.array([1.0, 2.0, 1.0]) / lengths
    a = np.cross(wave, [0.3, 0.7, 1.0])
    omega = np.linalg.norm(wave)
    theta = coords * wave
    si = np.sin(theta)
    co = np.cos(theta)
    if electric:
        values = np.array(
            [
                a[0] * co[:, 0] * si[:, 1] * si[:, 2],
                a[1] * si[:, 0] * co[:, 1] * si[:, 2],
                a[2] * si[:, 0] * si[:, 1] * co[:, 2],
            ]
        ) * np.cos(omega * t)
    else:
        curl = np.cross(wave, a)
        values = np.array(
            [
                curl[0] * si[:, 0] * co[:, 1] * co[:, 2],
                curl[1] * co[:, 0] * si[:, 1] * co[:, 2],
                curl[2] * co[:, 0] * co[:, 1] * si[:, 2],
            ]
        ) * (-np.sin(omega * t) / omega)
    return values[components, np.arange(len(coords))]


def make_mesh(block_cells):
    mesh = BlockMesh((3, 3, 3), 1.0, cells_per_block=block_cells, ratio=3, max_level=1)
    mesh.split(Block(0, (1, 1, 1)))
    mesh.geometry_leaf_count = len(mesh.leaves)
    mesh.balance()
    return mesh


def convergence(block_cells, device):
    mesh = make_mesh(block_cells)
    op = BlockOperator(mesh)
    s = PlanarStepper(op, device=device)
    e0 = mode(op.ecoords, op.e_components, op.lengths, 0, True)
    h0 = mode(op.hcoords, op.h_components, op.lengths, -s.dt / 2, False)
    s.set_fields(op.project_e(e0), h0)
    e, h = s.fields()
    u0 = op.energy(e, h, s.dt)
    initial_flux = op.normal_flux @ h
    steps = int(round(0.4 / s.dt))
    s.run(steps)
    en, hn = s.fields()
    exact = mode(op.ecoords, op.e_components, op.lengths, steps * s.dt, True)
    denominator = np.dot(e0 * op.me_full, e0)
    diff = op.p @ en - exact
    return {
        "block_cells": block_cells,
        "mesh": mesh.accounting(),
        "audit": op.audit(),
        "steps": steps,
        "time": steps * s.dt,
        "electric_relative_l2": float(
            np.sqrt(np.dot(diff * op.me_full, diff) / denominator)
        ),
        "modified_energy_relative_change": float(op.energy(en, hn, s.dt) / u0 - 1),
        "normal_flux_change_maxabs": float(
            np.max(np.abs(op.normal_flux @ hn - initial_flux), initial=0)
        ),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default="cpu")
    p.add_argument("--output", type=Path, default=Path("results/warp/closed_patch_cpu"))
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    rows = []
    for n in (3, 6, 12):
        row = convergence(n, a.device)
        rows.append(row)
        print(json.dumps(row), flush=True)
    for prev, row in zip(rows, rows[1:]):
        row["observed_order"] = float(
            np.log(prev["electric_relative_l2"] / row["electric_relative_l2"])
            / np.log(2)
        )
    result = {
        "status": "complete",
        "device": a.device,
        "precision": "float64",
        "scope": "Closed 3D refined cavity patch, all components, PEC walls; not a driven guide",
        "cases": rows,
        "second_order_gate_passed": all(r["observed_order"] >= 1.8 for r in rows[1:]),
        "source_sha256": {
            f"warp_fdtd/{n}": hashlib.sha256(
                Path(__file__).with_name(n).read_bytes()
            ).hexdigest()
            for n in (
                "block_operator.py",
                "block_validation.py",
                "block_mesh.py",
                "planar_interface.py",
            )
        },
    }
    (a.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    if not result["second_order_gate_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
