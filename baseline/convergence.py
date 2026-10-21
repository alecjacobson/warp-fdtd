"""Measure Yee dispersion on an analytic periodic Maxwell mode using upstream steps."""

import json
import time

import fdtdx as f
from fdtdx.fdtd.forward import forward
import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import numpy as np
from scipy.optimize import minimize_scalar

from .common import ROOT, environment, finite_tree, timed, write_json

LENGTH = 1550e-9
PERIOD = LENGTH / f.constants.c


def fit_frequency(signal, dt):
    # Fit amplitude and phase as nuisance parameters. No numerical-dispersion
    # prediction is used to select the fitted frequency.
    t = np.arange(1, len(signal) + 1, dtype=float) * dt / PERIOD
    signal = signal.astype(float)

    def loss(ratio):
        phase = 2 * np.pi * ratio * t
        basis = np.column_stack((np.cos(phase), np.sin(phase), np.ones_like(t)))
        coeff = np.linalg.lstsq(basis, signal, rcond=None)[0]
        return np.mean((basis @ coeff - signal) ** 2)

    search = np.linspace(0.9, 1.1, 101)
    best = int(np.argmin([loss(r) for r in search]))
    assert 0 < best < len(search) - 1, "Frequency outside fit range"
    fit = minimize_scalar(
        loss,
        bounds=(search[best - 1], search[best + 1]),
        method="bounded",
        options={"xatol": 1e-13},
    )
    assert fit.success
    return float(fit.x), float(np.sqrt(fit.fun / np.mean(signal**2)))


def measure(cells, courant):
    dx = LENGTH / cells
    config = f.SimulationConfig(
        time=20 * PERIOD,
        grid=f.UniformGrid(spacing=dx),
        courant_factor=courant,
        dtype=jnp.float32,
        backend="gpu",
    )
    volume = f.SimulationVolume(name="volume", partial_grid_shape=(4, 4, cells))
    bounds, constraints = f.boundary_objects_from_config(
        f.BoundaryConfig.from_uniform_bound(boundary_type="periodic"), volume
    )
    key = jax.random.PRNGKey(0)
    objects, arrays, params, config, _ = f.place_objects(
        [volume, *bounds.values()], config, constraints, key
    )
    arrays, objects, _ = f.apply_params(arrays, objects, params, key)
    mode = jnp.cos(2 * jnp.pi * jnp.arange(cells) / cells)
    arrays = arrays.aset("fields->E", arrays.fields.E.at[0].set(mode[None, None, :]))

    # E is a single spatial Fourier mode, H starts at zero: a standing wave.
    # A sine/cosine fit absorbs the leapfrog initial half-step phase.
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
            amplitude = 2 * jnp.mean(state[1].fields.E[0] * mode[None, None, :])
            return state, amplitude

        return jax.lax.scan(
            step, (jnp.asarray(0, jnp.int32), arr), None, length=config.time_steps_total
        )

    start = time.perf_counter()
    executable = jax.jit(evolve).lower(arrays).compile()
    compilation = time.perf_counter() - start
    ((_, final), signal), seconds = timed(executable, arrays)
    assert finite_tree(final.fields)
    signal = np.asarray(signal)
    dt = config.time_step_duration
    ratio, residual = fit_frequency(signal, dt)
    omega = 2 * np.pi / PERIOD
    omega_space = 2 * f.constants.c / dx * np.sin(np.pi / cells)
    omega_discrete = 2 / dt * np.arcsin(omega_space * dt / 2)
    predicted = omega_discrete / omega
    row = dict(
        cells_per_wavelength=cells,
        shape=list(final.fields.E.shape[1:]),
        dx_nm=dx * 1e9,
        dt_fs=dt * 1e15,
        courant_factor=courant,
        steps=config.time_steps_total,
        compile_seconds=compilation,
        execution_seconds=seconds,
        measured_frequency_ratio=ratio,
        continuum_relative_error=abs(ratio - 1),
        semidiscrete_frequency_ratio=omega_space / omega,
        temporal_relative_error=abs(ratio * omega / omega_space - 1),
        predicted_frequency_ratio=predicted,
        relative_error_vs_discrete_prediction=abs(ratio / predicted - 1),
        fit_relative_rms=residual,
        finite_fields=True,
    )
    assert residual < 1e-4, row
    assert row["relative_error_vs_discrete_prediction"] < 2e-6, row
    return row, signal


def plot(rows, output):
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), constrained_layout=True)
    for cfl in (0.99, 0.5, 0.25):
        r = sorted(
            (r for r in rows if r["courant_factor"] == cfl), key=lambda r: r["dx_nm"]
        )
        axes[0].loglog(
            [x["dx_nm"] for x in r],
            [x["continuum_relative_error"] for x in r],
            "o-",
            label=f"CFL factor {cfl}",
        )
    r = sorted(
        (r for r in rows if r["cells_per_wavelength"] == 32), key=lambda r: r["dt_fs"]
    )
    dt = np.array([x["dt_fs"] for x in r])
    err = np.array([x["temporal_relative_error"] for x in r])
    axes[1].loglog(
        dt, err, "o-", color="#007f86", label="Measured vs exact semi-discrete mode"
    )
    axes[1].loglog(
        dt, err[-1] * (dt / dt[-1]) ** 2, "--", color="gray", label="Second-order guide"
    )
    axes[2].semilogx(
        dt, [100 * x["continuum_relative_error"] for x in r], "o-", color="#df7852"
    )
    axes[2].axhline(
        100 * abs(r[0]["semidiscrete_frequency_ratio"] - 1),
        ls="--",
        color="gray",
        label="Spatial-error floor as Δt → 0",
    )
    axes[0].set(
        xlabel="Grid spacing Δx (nm)",
        ylabel="Relative frequency error",
        title="Space + time refinement vs continuum",
    )
    axes[1].set(
        xlabel="Time step Δt (fs)",
        ylabel="Relative frequency error",
        title="Time error isolated at 32 cells/λ",
    )
    axes[2].set(
        xlabel="Time step Δt (fs)",
        ylabel="Continuum frequency error (%)",
        title="Smaller Δt need not reduce total error",
    )
    ticks = sorted({r["dx_nm"] for r in rows})
    axes[0].set_xticks(ticks, [f"{x:.1f}" for x in ticks])
    for ax in axes[1:]:
        ax.set_xticks([0.012, 0.025, 0.05, 0.1], ["0.012", "0.025", "0.05", "0.1"])
    for ax in axes:
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.grid(alpha=0.2, which="both")
        ax.legend(fontsize=8)
    fig.suptitle(
        "Measured FDTDX GPU propagation • vacuum periodic Maxwell mode • λ = 1550 nm • float32"
    )
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"accuracy_trends.{suffix}", dpi=180)
    plt.close(fig)


def main():
    output = ROOT / "results/convergence"
    output.mkdir(parents=True, exist_ok=True)
    result = {
        "environment": environment(),
        "status": "running",
        "rows": [],
        "reference": "Analytic continuum vacuum periodic Maxwell mode; omega=c*k. Temporal-only reference is exact-in-time centered spatial discretization, omega_h=2*c*sin(k*dx/2)/dx.",
        "scope": "Homogeneous periodic mode, no PML, sources, material interfaces or inverse design. Actual upstream forward() steps, not just a plotted dispersion formula. Execution time is diagnostic, not a performance benchmark.",
    }
    assert jax.devices()[0].platform == "gpu"
    cases = [(n, c) for n in (8, 16, 32, 64, 128) for c in (0.99, 0.5, 0.25)]
    cases += [(32, c) for c in (0.7, 0.35, 0.125)]
    signals = {}
    for n, c in cases:
        row, signal = measure(n, c)
        result["rows"].append(row)
        signals[f"n{n}_cfl{c}"] = signal
        write_json(output / "result.json", result)
        print(json.dumps(row), flush=True)
        jax.clear_caches()
    slopes = {}
    for c in (0.99, 0.5, 0.25):
        r = [
            r
            for r in result["rows"]
            if r["courant_factor"] == c and r["cells_per_wavelength"] >= 16
        ]
        slopes[str(c)] = float(
            np.polyfit(
                np.log([x["dx_nm"] for x in r]),
                np.log([x["continuum_relative_error"] for x in r]),
                1,
            )[0]
        )
    r = [r for r in result["rows"] if r["cells_per_wavelength"] == 32]
    temporal_order = float(
        np.polyfit(
            np.log([x["dt_fs"] for x in r]),
            np.log([x["temporal_relative_error"] for x in r]),
            1,
        )[0]
    )
    result.update(
        status="complete",
        spatial_orders=slopes,
        temporal_order=temporal_order,
        passed=all(1.9 < v < 2.1 for v in (*slopes.values(), temporal_order)),
    )
    write_json(output / "result.json", result)
    np.savez_compressed(output / "mode_traces.npz", **signals)
    plot(result["rows"], output)
    assert result["passed"], result


if __name__ == "__main__":
    main()
