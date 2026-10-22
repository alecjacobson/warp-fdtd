"""Audit physical plane positions before interpreting saved mesh comparisons.

This checks a necessary condition, not reference uncertainty or material parity.
Plane coordinates are the integer Yee-grid anchors used by the exported source
and detectors; individual components still have their usual staggered offsets.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def summarize(case):
    case = Path(case)
    manifest = case / "manifest.json"
    meta = json.loads(manifest.read_text())
    raw = case / "inputs.npz"
    if hashlib.sha256(raw.read_bytes()).hexdigest() != meta["inputs_sha256"]:
        raise ValueError(f"Input checksum mismatch: {case}")
    with np.load(raw) as inp:
        edges = [inp[c + "_edges_m"].astype(float) for c in "xyz"]
    planes = {}
    for obj in meta["objects"]:
        if obj["class"] not in ("ModePlaneSource", "ModeOverlapDetector"):
            continue
        axes = [a for a, (lo, hi) in enumerate(obj["grid_slice"]) if hi - lo == 1]
        if len(axes) != 1:
            raise ValueError(f"Expected plane: {obj['name']}")
        axis = axes[0]
        planes[obj["name"]] = dict(
            axis=axis, anchor_nm=float(edges[axis][obj["grid_slice"][axis][0]] * 1e9)
        )
    return dict(
        case=str(case),
        manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
        spacing_nm=meta["settings"]["spacing_nm"],
        domain_nm=[float((e[-1] - e[0]) * 1e9) for e in edges],
        pml_thickness_nm=meta["settings"]["spacing_nm"] * meta["settings"]["pml_cells"],
        duration_fs=meta["time_steps"] * meta["dt_s"] * 1e15,
        parameter_sha256=meta["parameter_sha256"],
        beta=meta["beta"],
        planes=planes,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--cases",
        nargs="+",
        default=[
            "results/warp_reference/hann_200fs",
            "results/warp_reference/hann_10nm",
        ],
    )
    p.add_argument("--output", default="results/warp/reference_audit/result.json")
    p.add_argument("--require-matched-planes", action="store_true")
    args = p.parse_args()
    rows = [summarize(c) for c in args.cases]
    first = rows[0]
    differences = []
    for r in rows[1:]:
        for name in sorted(set(first["planes"]) | set(r["planes"])):
            a, b = first["planes"].get(name), r["planes"].get(name)
            if (
                a is None
                or b is None
                or a["axis"] != b["axis"]
                or abs(a["anchor_nm"] - b["anchor_nm"]) > 1e-3
            ):
                differences.append(dict(case=r["case"], plane=name, first=a, other=b))
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    result = dict(
        scope="Necessary plane-placement audit only; not a full reference certification",
        matched_planes=not differences,
        rows=rows,
        differences=differences,
    )
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if args.require_matched_planes and differences:
        raise SystemExit(
            "Physical plane mismatch; do not interpret as controlled mesh convergence"
        )


if __name__ == "__main__":
    main()
