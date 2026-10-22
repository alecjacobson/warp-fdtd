"""Export exact material/source/CPML inputs for a later engine comparison."""

import argparse
import hashlib
import json
from pathlib import Path

import fdtdx as f
import jax
import jax.numpy as jnp
import numpy as np

from .common import ROOT, environment, write_json
from .scene import Settings, build
from .reference_protocol import audit_placed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=ROOT / "results/hann_10nm")
    parser.add_argument("--output", type=Path, default=ROOT / "results/warp_reference")
    args = parser.parse_args()
    record = json.loads((args.run / "result.json").read_text())
    assert record["status"] == "complete"
    settings = Settings(**record["settings"])
    manifest = {
        "environment": environment(),
        "settings": record["settings"],
        "reference_result": str(args.run / "result.json"),
        "reference_result_sha256": hashlib.sha256(
            (args.run / "result.json").read_bytes()
        ).hexdigest(),
        "reference_arrays": str(args.run / "arrays.npz"),
        "reference_arrays_sha256": hashlib.sha256(
            (args.run / "arrays.npz").read_bytes()
        ).hexdigest(),
        "baseline_median_seconds": None
        if record["arguments"].get("reference_only")
        else record["median_seconds"],
        "baseline_timing_scope": record.get(
            "timing_scope", "warm synchronized evaluations"
        ),
        "reference_measurements": record["measurements"][0],
        "convention": "FDTDX normalized fields, Yee staggering and CPML; use the pinned source/update conventions.",
    }
    gradient_path = args.run / "gradients.npz"
    if gradient_path.exists():
        manifest["reference_gradients"] = str(gradient_path)
        manifest["reference_gradients_sha256"] = hashlib.sha256(
            gradient_path.read_bytes()
        ).hexdigest()
    objects, arrays, params, config, _ = build(settings)
    manifest["physical_contract"] = audit_placed(settings, objects, config)
    if record["arguments"].get("load"):
        param_path = Path(record["arguments"]["load"])
        assert (
            hashlib.sha256(param_path.read_bytes()).hexdigest()
            == record["loaded_params_sha256"]
        )
        with np.load(param_path) as saved:
            leaves, treedef = jax.tree.flatten(params)
            params = jax.tree.unflatten(
                treedef, [jnp.asarray(saved[f"p{i}"]) for i in range(len(leaves))]
            )
    else:
        assert record["arguments"].get("init", "random") == "random", (
            "Nonrandom initializers require a saved candidate"
        )
    arrays, objects, _ = f.apply_params(
        arrays,
        objects,
        params,
        jax.random.PRNGKey(settings.seed),
        beta=record["arguments"]["beta"],
    )
    payload = {
        "inv_permittivity": np.asarray(arrays.inv_permittivities),
        "inv_permeability": np.asarray(arrays.inv_permeabilities),
    }
    payload.update(
        {
            f"parameters/p{i}": np.asarray(v)
            for i, v in enumerate(jax.tree.leaves(params))
        }
    )
    for axis in "xyz":
        payload[f"{axis}_edges_m"] = np.asarray(
            getattr(config.resolved_grid, f"{axis}_edges")
        )
    manifest.update(
        dt_s=config.time_step_duration,
        time_steps=config.time_steps_total,
        parameter_sha256=record.get("loaded_params_sha256"),
        beta=record["arguments"]["beta"],
    )
    manifest["objects"] = []
    for obj in [*objects.sources, *objects.pml_objects, *objects.detectors]:
        meta = {
            "name": obj.name,
            "class": type(obj).__name__,
            "grid_slice": [[s.start, s.stop] for s in obj.grid_slice],
        }
        for attr in (
            "direction",
            "propagation_axis",
            "horizontal_axis",
            "vertical_axis",
            "mode_index",
            "scale_factor",
            "static_amplitude_factor",
            "alpha_start",
            "alpha_end",
            "alpha_order",
            "sigma_start",
            "sigma_end",
            "sigma_order",
            "kappa_start",
            "kappa_end",
            "kappa_order",
        ):
            if hasattr(obj, attr):
                value = getattr(obj, attr)
                meta[attr] = (
                    value
                    if value is None or isinstance(value, (str, bool, int, float))
                    else float(value)
                )
        for attr in (
            "_E",
            "_H",
            "_neff",
            "_time_offset_E",
            "_time_offset_H",
            "_time_step_to_on_idx",
            "pml_a_E",
            "pml_b_E",
            "inv_kappa_E",
            "pml_a_H",
            "pml_b_H",
            "inv_kappa_H",
            "_mode_E",
            "_mode_H",
            "_mode_neff",
            "_time_step_to_arr_idx",
            "_window_at_time_step_arr",
        ):
            value = getattr(obj, attr, None)
            if isinstance(value, jax.Array):
                payload[f"{obj.name}/{attr}"] = np.asarray(value)
        if hasattr(obj, "switch"):
            meta["fixed_sample_steps"] = obj.switch.fixed_on_time_steps
        if hasattr(obj, "_window_sum"):
            meta["window_sum"] = float(obj._window_sum)
        if hasattr(obj, "temporal_profile"):
            meta["temporal_profile"] = type(obj.temporal_profile).__name__
            meta["frequency_hz"] = float(obj.wave_character.get_frequency())
            meta["phase_shift"] = float(obj.wave_character.phase_shift)
            meta["profile_phase_shift"] = float(
                getattr(obj.temporal_profile, "phase_shift", 0.0)
            )
            meta["num_startup_periods"] = int(
                getattr(obj.temporal_profile, "num_startup_periods", 0)
            )
            meta["waveform_implementation"] = (
                "upstream/fdtdx/src/fdtdx/objects/sources/tfsf.py"
            )
        manifest["objects"].append(meta)
    # Confirm the exported material matches the evaluated cross section.
    with np.load(args.run / "arrays.npz") as evaluated:
        manifest["field_reference_scope"] = (
            "full_3d" if evaluated["E"].ndim == 4 else "mid_z_cross_section"
        )
        material = payload["inv_permittivity"]
        if evaluated["inv_eps"].ndim < material.ndim:
            material = material[..., material.shape[-1] // 2]
        np.testing.assert_allclose(
            material,
            evaluated["inv_eps"],
            rtol=0,
            atol=1e-6,
        )
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / "inputs.npz"
    np.savez_compressed(path, **payload)
    manifest["inputs_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest["arrays"] = {
        name: {"shape": list(value.shape), "dtype": str(value.dtype)}
        for name, value in payload.items()
    }
    write_json(args.output / "manifest.json", manifest)
    print(f"Exported {len(payload)} arrays to {path}")


if __name__ == "__main__":
    main()
