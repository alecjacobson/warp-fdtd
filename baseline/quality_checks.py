"""Predeclared acceptance gates for the independently evaluated bend."""

import argparse
import json

import numpy as np

from .common import ROOT, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--window", choices=["rectangular", "hann"], default="hann")
    args = parser.parse_args()
    names = [
        "straight20_calibration",
        "quality_200fs",
        "quality_300fs",
        "quality_400fs",
        "quality_thicker_pml",
        "quality_10nm",
    ]
    records = {}
    missing = []
    paths = {
        name: ROOT
        / "results"
        / (
            (
                "hann_straight20"
                if name == "straight20_calibration"
                else name.replace("quality_", "hann_")
            )
            if args.window == "hann"
            else name
        )
        for name in names
    }
    output = (
        ROOT
        / "results"
        / (
            "quality_checks.json"
            if args.window == "hann"
            else "quality_checks_rectangular.json"
        )
    )
    for name in names:
        path = paths[name] / "result.json"
        if (
            not path.exists()
            or json.loads(path.read_text()).get("status") != "complete"
        ):
            missing.append(paths[name].name)
        else:
            records[name] = json.loads(path.read_text())
    thresholds = {
        "calibration_T_error": 0.01,
        "calibration_R_max": 0.001,
        "time_T_absolute_change": 0.005,
        "pml_T_absolute_change": 0.005,
        "mesh_T_absolute_change": 0.01,
        "target_fine_mesh_T_min": 0.92,
    }
    result = {
        "window": args.window,
        "runs": {n: str(p.relative_to(ROOT)) for n, p in paths.items()},
        "thresholds": thresholds,
        "missing": missing,
        "passed": False,
    }
    if missing:
        duration_names = ["quality_200fs", "quality_300fs", "quality_400fs"]
        if all(n in records for n in duration_names):
            ts = [
                records[n]["measurements"][0]["modal_transmission"]
                for n in duration_names
            ]
            result.update(
                time_T_change=max(ts) - min(ts),
                time_convergence=max(ts) - min(ts) < 0.005,
            )
        write_json(output, result)
        raise SystemExit("Quality evaluation incomplete: " + ", ".join(missing))
    m = {name: r["measurements"][0] for name, r in records.items()}
    hashes = {
        records[n]["loaded_params_sha256"]
        for n in names
        if n != "straight20_calibration"
    }
    assert len(hashes) == 1, "Convergence evaluations used different candidates"
    cal = m["straight20_calibration"]
    ts = [
        m[name]["modal_transmission"]
        for name in ("quality_200fs", "quality_300fs", "quality_400fs")
    ]
    time_change = max(ts) - min(ts)
    pml_change = abs(
        m["quality_thicker_pml"]["modal_transmission"]
        - m["quality_300fs"]["modal_transmission"]
    )
    mesh_change = abs(
        m["quality_10nm"]["modal_transmission"]
        - m["quality_300fs"]["modal_transmission"]
    )
    with np.load(paths["quality_300fs"] / "arrays.npz") as a:
        coarse = a["inv_eps"].copy()
    with np.load(paths["quality_thicker_pml"] / "arrays.npz") as a:
        padded = a["inv_eps"][..., 10:-10, 10:-10].copy()
    with np.load(paths["quality_10nm"] / "arrays.npz") as a:
        fine = a["inv_eps"][..., ::2, ::2].copy()
    geometry_passed = bool(
        np.allclose(coarse, padded, atol=1e-6) and np.allclose(coarse, fine, atol=1e-6)
    )
    checks = {
        "straight_calibration": abs(cal["modal_transmission"] - 1) < 0.01
        and cal["modal_reflection"] < 0.001,
        "time_convergence": time_change < 0.005,
        "pml_convergence": pml_change < 0.005,
        "mesh_convergence": mesh_change < 0.01,
        "same_physical_material_cross_section": geometry_passed,
        "fine_mesh_quality_target": m["quality_10nm"]["modal_transmission"] >= 0.92,
    }
    result.update(
        checks=checks,
        passed=all(checks.values()),
        time_T_change=time_change,
        pml_T_change=pml_change,
        mesh_T_change=mesh_change,
        fine_mesh_T=m["quality_10nm"]["modal_transmission"],
        fine_mesh_R=m["quality_10nm"]["modal_reflection"],
        candidate_sha256=next(iter(hashes)),
    )
    write_json(output, result)
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit(
            "At least one design-quality gate failed; inspect quality_checks.json"
        )


if __name__ == "__main__":
    main()
