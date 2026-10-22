"""Matched warm timing for the periodic accuracy fixture, with the same point probe."""

import json
from pathlib import Path
import time
import fdtdx as f
from fdtdx.fdtd.forward import forward
import jax
import jax.numpy as jnp
import numpy as np
from .common import environment, timed, finite_tree, write_json
from .convergence import LENGTH, PERIOD, fit_frequency


def measure(n):
    config = f.SimulationConfig(
        time=20 * PERIOD,
        grid=f.UniformGrid(spacing=LENGTH / n),
        courant_factor=0.99,
        dtype=jnp.float32,
        backend="gpu",
    )
    volume = f.SimulationVolume(name="volume", partial_grid_shape=(4, 4, n))
    bounds, constraints = f.boundary_objects_from_config(
        f.BoundaryConfig.from_uniform_bound(boundary_type="periodic"), volume
    )
    key = jax.random.PRNGKey(0)
    objects, arrays, params, config, _ = f.place_objects(
        [volume, *bounds.values()], config, constraints, key
    )
    arrays, objects, _ = f.apply_params(arrays, objects, params, key)
    initial = np.cos(2 * np.pi * np.arange(n, dtype=np.float32) / n)
    arrays = arrays.aset(
        "fields->E", arrays.fields.E.at[0].set(jnp.asarray(initial)[None, None, :])
    )

    def evolve(arr):
        def step(state, unused):
            state = forward(
                state,
                config,
                objects,
                key,
                record_detectors=False,
                record_boundaries=False,
                simulate_boundaries=True,
                inject_sources=False,
            )
            return state, state[1].fields.E[0, 0, 0, 0]

        return jax.lax.scan(
            step, (jnp.asarray(0, jnp.int32), arr), None, length=config.time_steps_total
        )

    start = time.perf_counter()
    exe = jax.jit(evolve).lower(arrays).compile()
    comp = time.perf_counter() - start
    _, warm = timed(exe, arrays)
    seconds = []
    for _ in range(3):
        result, elapsed = timed(exe, arrays)
        seconds.append(elapsed)
    (_, final), signal = result
    signal = np.asarray(signal)
    ratio, residual = fit_frequency(signal, config.time_step_duration)
    predicted = (
        2
        / config.time_step_duration
        * np.arcsin(
            config.time_step_duration * f.constants.c / (LENGTH / n) * np.sin(np.pi / n)
        )
        / (2 * np.pi / PERIOD)
    )
    row = dict(
        method="fdtdx_yee",
        cells_per_wavelength=n,
        courant_factor=0.99,
        shape=[4, 4, n],
        steps=config.time_steps_total,
        dt_fs=config.time_step_duration * 1e15,
        dx_nm=LENGTH / n * 1e9,
        compile_seconds=comp,
        warmup_seconds=warm,
        seconds=seconds,
        median_seconds=float(np.median(seconds)),
        measured_frequency_ratio=ratio,
        continuum_relative_error=abs(ratio - 1),
        prediction_residual=abs(ratio / predicted - 1),
        fit_relative_rms=residual,
        finite_fields=finite_tree(final.fields),
    )
    assert (
        row["finite_fields"] and row["prediction_residual"] < 2e-6 and residual < 1e-4
    ), row
    return row, signal


def main():
    out = Path("results/periodic_performance")
    out.mkdir(parents=True, exist_ok=True)
    record = dict(
        status="running",
        environment=environment(),
        scope="4x4xN periodic vacuum Maxwell mode, initial Ex=cos(2*pi*z/N), H=0, 20 periods, identical point probe E_x[0,0,0] each step. Warm timing excludes compile/first execution, includes final fields and probe trajectory.",
        rows=[],
    )
    signals = {}
    for n in (8, 16, 32, 64, 128):
        row, signal = measure(n)
        record["rows"].append(row)
        signals[str(n)] = signal
        print(json.dumps(row), flush=True)
        write_json(out / "result.json", record)
        jax.clear_caches()
    record["status"] = "complete"
    record["passed"] = True
    write_json(out / "result.json", record)
    np.savez_compressed(out / "signals.npz", **signals)


if __name__ == "__main__":
    main()
