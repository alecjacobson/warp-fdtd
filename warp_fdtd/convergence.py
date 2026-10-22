"""Measure numerical frequency, not just the dispersion formula, on the GPU."""

import json
from pathlib import Path
import time
import numpy as np
from scipy.optimize import minimize_scalar, least_squares
from .periodic import Periodic

LENGTH = 1550e-9
PERIOD = LENGTH / 299792458.0


def fit_frequency(signal, dt, damped=False):
    t = np.arange(1, len(signal) + 1, dtype=float) * dt / PERIOD
    signal = signal.astype(float)

    def loss(ratio):
        phase = 2 * np.pi * ratio * t
        basis = np.column_stack((np.cos(phase), np.sin(phase), np.ones_like(t)))
        coeff = np.linalg.lstsq(basis, signal, rcond=None)[0]
        return np.mean((basis @ coeff - signal) ** 2)

    search = np.linspace(0.9, 1.1, 101)
    i = int(np.argmin([loss(r) for r in search]))
    assert 0 < i < len(search) - 1
    fit = minimize_scalar(
        loss,
        bounds=(search[i - 1], search[i + 1]),
        method="bounded",
        options={"xatol": 1e-13},
    )
    assert fit.success
    if damped:

        def residual(v):
            phase = 2 * np.pi * v[0] * t
            envelope = np.exp(-v[1] * t)
            basis = np.column_stack(
                (envelope * np.cos(phase), envelope * np.sin(phase), np.ones_like(t))
            )
            return basis @ np.linalg.lstsq(basis, signal, rcond=None)[0] - signal

        joint = least_squares(
            residual,
            [fit.x, 0.0],
            bounds=([0.9, -0.1], [1.1, 1.0]),
            xtol=1e-13,
            ftol=1e-13,
            gtol=1e-13,
        )
        assert joint.success
        return (
            float(joint.x[0]),
            float(np.sqrt(np.mean(joint.fun**2) / np.mean(signal**2))),
            float(joint.x[1]),
        )
    return float(fit.x), float(np.sqrt(fit.fun / np.mean(signal**2))), 0.0


def measure(n, cfl, method):
    s = Periodic(n, cfl, method)
    compilation = s.compile()
    times = []
    for r in range(4):
        s.reset()
        start = time.perf_counter()
        s.run()
        elapsed = time.perf_counter() - start
        if r:
            times.append(elapsed)
    signal = s.signal.numpy()
    ratio, residual, decay = fit_frequency(signal, s.dt, damped=method == "rk4")
    omega = 2 * np.pi / PERIOD
    symbol = np.sin(np.pi / n)
    if method == "rk4":
        symbol = 1.125 * symbol - np.sin(3 * np.pi / n) / 24
    space = 2 * 299792458.0 / (LENGTH / n) * symbol
    if method == "yee":
        predicted = 2 / s.dt * np.arcsin(space * s.dt / 2) / omega
    else:
        z = 1j * space * s.dt
        predicted = np.angle(1 + z + z**2 / 2 + z**3 / 6 + z**4 / 24) / s.dt / omega
    row = dict(
        amplitude_loss_20_periods=float(1 - np.exp(-20 * decay)),
        method=method,
        cells_per_wavelength=n,
        courant_factor=cfl,
        shape=s.shape,
        steps=s.steps,
        dx_nm=LENGTH / n * 1e9,
        dt_fs=s.dt * 1e15,
        compile_seconds=compilation,
        seconds=times,
        median_seconds=float(np.median(times)),
        measured_frequency_ratio=ratio,
        continuum_relative_error=abs(ratio - 1),
        temporal_relative_error=abs(ratio * omega / space - 1),
        predicted_frequency_ratio=float(predicted),
        prediction_residual=abs(ratio / predicted - 1),
        fit_relative_rms=residual,
        finite_fields=bool(np.isfinite(s.state.numpy()).all()),
    )
    assert (
        row["finite_fields"] and row["prediction_residual"] < 2e-6 and residual < 1e-4
    ), row
    return row, signal


def main():
    out = Path("results/warp/convergence")
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    signals = {}
    cases = [(n, c) for n in (8, 16, 32, 64, 128) for c in (0.99, 0.5, 0.25)] + [
        (32, c) for c in (0.7, 0.35, 0.125)
    ]
    for method in ("yee", "rk4"):
        for n, c in cases:
            row, signal = measure(n, c, method)
            rows.append(row)
            signals[f"{method}_{n}_{c}"] = signal
            print(json.dumps(row), flush=True)
            (out / "result.json").write_text(
                json.dumps(dict(status="running", rows=rows), indent=2)
            )
    orders = {}
    for method in ("yee", "rk4"):
        r = [
            x
            for x in rows
            if x["method"] == method
            and x["courant_factor"] == 0.99
            and 8 <= x["cells_per_wavelength"] <= 32
        ]
        orders[method] = float(
            np.polyfit(
                np.log([x["dx_nm"] for x in r]),
                np.log([x["continuum_relative_error"] for x in r]),
                1,
            )[0]
        )
    result = dict(
        status="complete",
        rows=rows,
        orders=orders,
        scope="Smooth homogeneous periodic Maxwell mode, 4x4xN grid, 20 optical periods; fourth-order path is not validated for dielectric interfaces or PML",
        passed=1.9 < orders["yee"] < 2.1 and 3.8 < orders["rk4"] < 4.2,
    )
    (out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    np.savez_compressed(out / "signals.npz", **signals)
    assert result["passed"], orders


if __name__ == "__main__":
    main()
