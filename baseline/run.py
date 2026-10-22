"""Synchronized forward and optimization benchmarks, with persistent evidence."""

import argparse
from contextlib import nullcontext
from dataclasses import asdict
import json
import hashlib
import pickle
import statistics
import time
import sys
import traceback
from pathlib import Path

import fdtdx as f
import jax
import jax.numpy as jnp
import numpy as np
import optax
from fdtdx.fdtd.fdtd import custom_fdtd_forward

from .common import environment, finite_tree, memory, timed, write_json
from .scene import Settings, build, fluxes
from .reference_protocol import audit_placed


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=["forward", "optimize", "gradient"])
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--spacing-nm", type=float, default=20)
    p.add_argument("--voxel-nm", type=float, default=20)
    p.add_argument(
        "--reference-protocol",
        choices=("legacy", "fixed_physical_v1"),
        default="legacy",
    )
    p.add_argument("--courant-factor", type=float, default=0.99)
    p.add_argument("--modal-window-fs", type=float, default=40.0)
    p.add_argument("--time-fs", type=float, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--backend", choices=["gpu", "cpu"], default="gpu")
    p.add_argument(
        "--gradient",
        choices=["none", "paper", "compressed", "exact", "checkpointed", "adjoint"],
        default="none",
    )
    p.add_argument(
        "--parameterization", choices=["smooth", "binary", "brush"], default="smooth"
    )
    p.add_argument("--exact-interpolation", action="store_true")
    p.add_argument("--pml-cells", type=int, default=10)
    p.add_argument("--domain-um", type=float, default=4)
    p.add_argument("--reps", type=int, default=3)
    p.add_argument(
        "--reference-only",
        action="store_true",
        help="One field evaluation for convergence; no warmed throughput claim",
    )
    p.add_argument("--epochs", type=int, default=250)
    p.add_argument("--beta", type=float, default=1.0)
    p.add_argument("--load", type=Path)
    p.add_argument("--resume", type=Path)
    p.add_argument("--finite-difference", action="store_true")
    p.add_argument("--quality-monitors", action="store_true")
    p.add_argument(
        "--poynting-plane",
        action="store_true",
        help="Record co-located E/H phasors on the central xy plane",
    )
    p.add_argument(
        "--modal-window", choices=["rectangular", "hann"], default="rectangular"
    )
    p.add_argument("--objective", choices=["flux", "modal"], default="flux")
    p.add_argument("--init", choices=["random", "ring"], default="random")
    p.add_argument("--fixed-beta", action="store_true")
    p.add_argument("--precision", choices=["float32", "float64"], default="float32")
    p.add_argument(
        "--max-steps",
        type=int,
        help="Forward-only capacity stress prefix; no transmission claim",
    )
    p.add_argument("--scene", choices=["bend", "straight"], default="bend")
    p.add_argument(
        "--save-full-fields",
        action="store_true",
        help="Save full E/H/material arrays for numerical parity checks",
    )
    p.add_argument("--source-profile", choices=["current", "paper"], default="current")
    p.add_argument(
        "--trace",
        action="store_true",
        help="Profile warmup only; timed repetitions remain unprofiled",
    )
    p.add_argument(
        "--pad-domain",
        action="store_true",
        help="Extend outer domain when thickening PML, preserving active geometry",
    )
    args = p.parse_args()
    if args.reference_only and (
        args.mode != "forward" or args.max_steps is not None or args.trace
    ):
        p.error("--reference-only requires an untraced full forward trajectory")
    if args.reps < 1 or args.epochs < 1:
        p.error("reps and epochs must be positive")
    if args.poynting_plane and (args.scene != "bend" or args.mode != "forward"):
        p.error("--poynting-plane currently requires forward mode with the bend scene")
    if args.mode != "forward" and args.gradient == "none":
        p.error("gradient/optimize require --gradient")
    if args.max_steps is not None and (
        args.mode != "forward" or args.max_steps < 1 or args.gradient != "none"
    ):
        p.error(
            "--max-steps requires positive steps and forward mode without gradients"
        )
    if args.load and args.resume:
        p.error("--load and --resume are mutually exclusive")
    if args.objective == "modal":
        args.quality_monitors = True
    if args.gradient == "adjoint" and args.objective != "modal":
        p.error("Frequency adjoint requires the phasor/modal objective")
    if args.precision == "float64":
        jax.config.update("jax_enable_x64", True)
    args.output.mkdir(parents=True, exist_ok=True)
    env = environment()
    assert all(
        d.platform == ("gpu" if args.backend == "gpu" else "cpu") for d in jax.devices()
    ), "Backend mismatch"
    s = Settings(**{k: getattr(args, k) for k in Settings.__dataclass_fields__})
    result = {
        "settings": asdict(s),
        "environment": env,
        "mode": args.mode,
        "status": "setting_up",
        "timing_scope": "single reference evaluation; not warmed throughput"
        if args.reference_only
        else "warm synchronized evaluations after one warmup",
        "arguments": {
            k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
        },
    }
    write_json(args.output / "result.json", result)
    start = time.perf_counter()
    objects, arrays, params, config, _ = build(s)
    result["physical_contract"] = audit_placed(s, objects, config)
    jax.block_until_ready((arrays, params))
    result.update(
        setup_seconds=time.perf_counter() - start,
        shape=list(arrays.fields.E.shape[1:]),
        time_steps=min(
            args.max_steps or config.time_steps_total, config.time_steps_total
        ),
        configured_time_steps=config.time_steps_total,
        physical_evaluation=args.max_steps is None
        or args.max_steps >= config.time_steps_total,
        parameters=sum(x.size for x in jax.tree.leaves(params)),
    )
    result["dtypes"] = {
        "E": str(arrays.fields.E.dtype),
        "H": str(arrays.fields.H.dtype),
        "inverse_permittivity": str(arrays.inv_permittivities.dtype),
        "detectors": {
            name: {k: str(v.dtype) for k, v in state.items()}
            for name, state in arrays.detector_states.items()
        },
    }
    if args.init == "ring":
        if s.parameterization != "smooth":
            raise ValueError(
                "Ring initialization currently uses the smooth parameterization"
            )

        def ring(x):
            xx = (jnp.arange(x.shape[0]) + 0.5) * s.voxel_nm / 1000
            yy = (jnp.arange(x.shape[1]) + 0.5) * s.voxel_nm / 1000
            radius = jnp.sqrt(xx[:, None] ** 2 + (yy[None, :] - 1.6) ** 2)
            return ((radius >= 0.6) & (radius <= 1.0)).astype(jnp.float32)[..., None]

        params = jax.tree.map(ring, params)
    if args.load:
        result["loaded_params_sha256"] = hashlib.sha256(
            args.load.read_bytes()
        ).hexdigest()
        with np.load(args.load) as saved:
            leaves, treedef = jax.tree.flatten(params)
            restored = [jnp.asarray(saved[f"p{i}"]) for i in range(len(leaves))]
            if any(x.shape != y.shape for x, y in zip(restored, leaves)):
                raise ValueError("Parameter shape mismatch")
            params = jax.tree.unflatten(treedef, restored)
    if args.precision == "float64":
        params = jax.tree.map(lambda x: x.astype(jnp.float64), params)
    key = jax.random.PRNGKey(s.seed)
    if config.gradient_config is not None:
        result["resolved_reversible_checkpoints"] = (
            config.gradient_config.num_checkpoints_reversible
        )

    def objective(params, arrays, beta):
        arrays, ob, _ = f.apply_params(arrays, objects, params, key, beta=beta)
        if args.max_steps is not None:
            _, arrays = custom_fdtd_forward(
                arrays=arrays,
                objects=ob,
                config=config,
                key=key,
                reset_container=True,
                record_detectors=True,
                start_time=0,
                end_time=min(args.max_steps, config.time_steps_total),
                show_progress=False,
            )
        else:
            _, arrays = f.run_fdtd(arrays, ob, config, key, show_progress=False)
        if not result["physical_evaluation"]:
            energy = jnp.mean(arrays.fields.E**2) + jnp.mean(arrays.fields.H**2)
            return energy, (arrays, {"mean_squared_field": energy})
        info = fluxes(arrays, ob)
        return -info[
            "modal_transmission" if args.objective == "modal" else "transmission"
        ], (arrays, info)

    fn = (
        objective
        if args.mode == "forward"
        else jax.value_and_grad(objective, has_aux=True)
    )
    beta = jnp.asarray(args.beta, dtype=jnp.float32)
    start = time.perf_counter()
    executable = (
        jax.jit(fn, donate_argnames=["arrays"]).lower(params, arrays, beta).compile()
    )
    result["compile_seconds"] = time.perf_counter() - start
    analysis = executable.memory_analysis()
    result["executable_memory"] = {
        k: getattr(analysis, k)
        for k in (
            "argument_size_in_bytes",
            "output_size_in_bytes",
            "alias_size_in_bytes",
            "temp_size_in_bytes",
        )
    }
    result["status"] = "running"
    write_json(args.output / "result.json", result)
    print(
        json.dumps(
            {
                k: result[k]
                for k in ("shape", "time_steps", "compile_seconds", "executable_memory")
            }
        ),
        flush=True,
    )

    if args.mode == "optimize":
        run_contract = {
            "settings": asdict(s),
            "epochs": args.epochs,
            "objective": args.objective,
            "fixed_beta": args.fixed_beta,
            "beta": args.beta,
        }
        schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.0005,
            peak_value=0.005,
            end_value=0.0005,
            warmup_steps=min(10, args.epochs // 3),
            decay_steps=max(2, round(0.9 * args.epochs)),
        )
        optimizer = optax.nadam(schedule)
        state = optimizer.init(params)
        start_epoch = 0
        if args.resume:
            # Only load locally generated checkpoints; pickle is not an interchange format.
            with args.resume.open("rb") as file:
                checkpoint = pickle.load(file)
            # New optional settings use their original defaults for older checkpoints.
            checkpoint["contract"]["settings"] = asdict(
                Settings(**checkpoint["contract"]["settings"])
            )
            if checkpoint["contract"] != run_contract:
                raise ValueError("Resume settings/schedule differ from checkpoint")
            params, state = jax.device_put(
                (checkpoint["params"], checkpoint["optimizer"])
            )
            start_epoch = checkpoint["next_epoch"]
        history = []
        if start_epoch:
            history = json.loads((args.output / "history.json").read_text())[
                :start_epoch
            ]
        for epoch in range(start_epoch, args.epochs):
            beta = jnp.asarray(
                0.1 + 49.9 * epoch / args.epochs
                if s.parameterization == "smooth" and not args.fixed_beta
                else args.beta,
                jnp.float32,
            )
            step_start = time.perf_counter()
            ((loss, (arrays, info)), grads), seconds = timed(
                executable, params, arrays, beta
            )
            if not finite_tree((loss, info, grads)):
                raise FloatingPointError(
                    f"Nonfinite objective/gradient at epoch {epoch}"
                )
            evaluated_params = params
            updates, state = optimizer.update(grads, state, params)
            params = jax.tree.map(
                lambda x: jnp.clip(x, 0, 1), optax.apply_updates(params, updates)
            )
            jax.block_until_ready((params, state))
            row = {
                "epoch": epoch,
                "beta": float(beta),
                "gradient_seconds": seconds,
                "step_seconds": time.perf_counter() - step_start,
                "grad_norm": float(optax.global_norm(grads)),
                "objective": float(-loss),
                **{k: float(v) for k, v in info.items()},
            }
            history.append(row)
            print(json.dumps(row), flush=True)
            # Save the actual evaluated candidate, not the subsequent unevaluated update.
            if row["objective"] >= max(r["objective"] for r in history):
                np.savez(
                    args.output / "best_params.npz",
                    **{
                        f"p{i}": np.asarray(x)
                        for i, x in enumerate(jax.tree.leaves(evaluated_params))
                    },
                )
                write_json(args.output / "best.json", row)
            if epoch % 5 == 0 or epoch == args.epochs - 1:
                write_json(args.output / "history.json", history)
                with (args.output / "checkpoint.tmp").open("wb") as file:
                    pickle.dump(
                        {
                            "params": jax.device_get(params),
                            "optimizer": jax.device_get(state),
                            "contract": run_contract,
                            "next_epoch": epoch + 1,
                        },
                        file,
                    )
                (args.output / "checkpoint.tmp").replace(args.output / "checkpoint.pkl")
        result["history"] = history
        np.savez(
            args.output / "final_params.npz",
            **{f"p{i}": np.asarray(x) for i, x in enumerate(jax.tree.leaves(params))},
        )
    else:
        times, values = [], []
        for rep in range(1 if args.reference_only else args.reps + 1):
            trace_context = (
                jax.profiler.trace(
                    str(args.output / "profiler"), create_perfetto_link=False
                )
                if args.trace and rep == 0
                else nullcontext()
            )
            with trace_context:
                out, seconds = timed(executable, params, arrays, beta)
            if args.mode == "forward":
                loss, (arrays, info) = out
            else:
                (loss, (arrays, info)), grads = out
                if not finite_tree(grads):
                    raise FloatingPointError("Nonfinite gradient")
                result["gradient_norm"] = float(optax.global_norm(grads))
            if not finite_tree((loss, info, arrays.fields)):
                raise FloatingPointError("Nonfinite field or measurement")
            if "input_flux" in info and float(info["input_flux"]) <= 0:
                raise ValueError("Nonpositive input flux")
            row = {k: float(v) for k, v in info.items()}
            print(json.dumps({"rep": rep, "seconds": seconds, **row}), flush=True)
            if rep == 0 and not args.reference_only:
                result["warmup_seconds"] = seconds
            else:
                times.append(seconds)
                values.append(row)
        result.update(
            run_seconds=times,
            median_seconds=statistics.median(times),
            measurements=values,
            gcups=np.prod(result["shape"]).item()
            * result["time_steps"]
            / statistics.median(times)
            / 1e9,
        )
        if args.trace:
            result["trace_directory"] = str(args.output / "profiler")
        # Retain compact cross sections and detector traces for CPU/GPU comparisons and figures.
        if args.poynting_plane:
            flow = next(o for o in objects.detectors if o.name == "flow")
            np.savez_compressed(
                args.output / "phasors.npz",
                phasors=np.asarray(arrays.detector_states["flow"]["phasor"])[
                    0, 0, ..., 0
                ],
                inv_eps=np.asarray(
                    arrays.inv_permittivities[..., flow.grid_slice[2].start]
                ),
                grid_slice=np.asarray(flow.grid_slice_tuple),
                dt_s=config.time_step_duration,
                window_weights=np.asarray(flow._window_at_time_step_arr),
            )
        mid = arrays.fields.E.shape[3] // 2
        np.savez_compressed(
            args.output / "arrays.npz",
            E=np.asarray(
                arrays.fields.E
                if args.save_full_fields
                else arrays.fields.E[:, :, :, mid]
            ),
            H=np.asarray(
                arrays.fields.H
                if args.save_full_fields
                else arrays.fields.H[:, :, :, mid]
            ),
            inv_eps=np.asarray(
                arrays.inv_permittivities
                if args.save_full_fields
                else arrays.inv_permittivities[..., mid]
            ),
            input=np.asarray(arrays.detector_states["input"]["poynting_flux"]),
            output=np.asarray(arrays.detector_states["output"]["poynting_flux"]),
        )
        if args.mode == "gradient":
            np.savez(
                args.output / "gradients.npz",
                **{
                    f"g{i}": np.asarray(x) for i, x in enumerate(jax.tree.leaves(grads))
                },
            )
            if args.finite_difference:
                if s.parameterization != "smooth":
                    raise ValueError(
                        "Finite differences cannot validate straight-through binary gradients"
                    )
                forward = (
                    jax.jit(objective, donate_argnames=["arrays"])
                    .lower(params, arrays, beta)
                    .compile()
                )
                norm = optax.global_norm(grads)
                if float(norm) == 0:
                    raise ValueError("Zero gradient cannot define a check direction")
                direction = jax.tree.map(lambda g: g / norm, grads)
                checks = []
                for h in (0.1, 0.03, 0.01, 0.003, 0.001):
                    plus = jax.tree.map(lambda x, d: x + h * d, params, direction)
                    minus = jax.tree.map(lambda x, d: x - h * d, params, direction)
                    (vp, (arrays, _)), _ = timed(forward, plus, arrays, beta)
                    (vm, (arrays, _)), _ = timed(forward, minus, arrays, beta)
                    fd = float((vp - vm) / (2 * h))
                    checks.append(
                        {
                            "h": h,
                            "finite_difference": fd,
                            "autodiff": float(norm),
                            "relative_error": abs(fd - float(norm))
                            / max(abs(fd), float(norm), 1e-12),
                        }
                    )
                result["finite_difference"] = checks
                print(json.dumps({"finite_difference": checks}), flush=True)
    result.update(status="complete", memory_stats=memory())
    write_json(args.output / "result.json", result)


if __name__ == "__main__":
    try:
        main()
    except (Exception, KeyboardInterrupt):
        if "--output" in sys.argv:
            path = Path(sys.argv[sys.argv.index("--output") + 1]) / "result.json"
            record = json.loads(path.read_text()) if path.exists() else {}
            record.update(status="failed", error=traceback.format_exc())
            write_json(path, record)
        raise
