"""Fixed-input 3D FDTD solver using only Warp for GPU computation.

NumPy loads immutable input bundles and prepares source coefficients on the host.
No fdtdx or JAX module is imported by this package.
"""

from pathlib import Path
import hashlib
import json
import time
import numpy as np
import warp as wp
from . import kernels as k


class Solver:
    def __init__(self, case, device="cuda:0", graph_steps=64):
        self.path = Path(case)
        self.meta = json.loads((self.path / "manifest.json").read_text())
        raw = self.path / "inputs.npz"
        if hashlib.sha256(raw.read_bytes()).hexdigest() != self.meta["inputs_sha256"]:
            raise ValueError("Input bundle checksum mismatch")
        with np.load(raw) as data:
            self.data = {n: data[n] for n in data.files}
        d, m = self.data, self.meta
        eps = d["inv_permittivity"]
        if (
            eps.ndim != 4
            or eps.shape[0] != 1
            or not np.all(np.isfinite(eps))
            or np.any(eps <= 0)
        ):
            raise ValueError("Only finite positive isotropic permittivity is supported")
        if d["inv_permeability"].ndim != 0 or d["inv_permeability"] != 1:
            raise ValueError("Only relative permeability 1 is supported")
        if m["settings"].get("precision", "float32") != "float32":
            raise ValueError("Only float32 field bundles supported")
        dx = m["settings"]["spacing_nm"] * 1e-9
        for axis in "xyz":
            if not np.allclose(np.diff(d[f"{axis}_edges_m"]), dx, rtol=1e-4, atol=0):
                raise ValueError("Only uniform cubic grids supported")
        self.shape = eps.shape[1:]
        self.n = int(np.prod(self.shape))
        self.steps = m["time_steps"]
        self.device = wp.get_device(device)
        self.graph_steps = graph_steps
        if graph_steps < 1:
            raise ValueError("graph_steps must be positive")
        self.courant = 0.99 / np.sqrt(3)
        self.E = wp.zeros((3, self.n), dtype=float, device=self.device)
        self.H = wp.zeros_like(self.E)
        self.hprev = wp.zeros_like(self.H)
        self.eps = wp.array(eps.reshape(-1), dtype=float, device=self.device)
        self.psi_e = wp.zeros((6, self.n), dtype=float, device=self.device)
        self.psi_h = wp.zeros_like(self.psi_e)
        self.step = wp.zeros(1, dtype=int, device=self.device)
        self.coeffs = {}
        for field in "EH":
            coeff = np.zeros((3, max(self.shape), 3), np.float32)
            coeff[:, :, 1:] = 1
            covered = np.zeros((3, max(self.shape)), bool)
            for obj in m["objects"]:
                if obj["class"] != "PerfectlyMatchedLayer":
                    continue
                name = obj["name"]
                dims = d[f"{name}/pml_a_{field}"].shape
                axis = int(np.argmax(dims))
                lo, hi = obj["grid_slice"][axis]
                if covered[axis, lo:hi].any():
                    raise ValueError("Overlapping CPML slabs unsupported")
                covered[axis, lo:hi] = True
                for j, key in enumerate(
                    (f"pml_a_{field}", f"pml_b_{field}", f"inv_kappa_{field}")
                ):
                    coeff[axis, lo:hi, j] = d[f"{name}/{key}"].ravel()
            self.coeffs[field] = wp.array(coeff, dtype=wp.vec3, device=self.device)
        sources = [o for o in m["objects"] if o["class"] == "ModePlaneSource"]
        if len(sources) != 1:
            raise ValueError("Exactly one exported plane mode source required")
        src = sources[0]
        if (
            src["direction"] != "+"
            or src["propagation_axis"] != 0
            or src["temporal_profile"] != "SingleFrequencyProfile"
        ):
            raise ValueError("Only +x single-frequency source supported")
        self.source_x = src["grid_slice"][0][0]
        if src["grid_slice"] != [
            [self.source_x, self.source_x + 1],
            [0, self.shape[1]],
            [0, self.shape[2]],
        ]:
            raise ValueError("Source must span complete yz plane")
        self.sources, self.waves = {}, {}
        name = src["name"]
        for field in "EH":
            profile = d[f"{name}/_{field}"]
            if np.iscomplexobj(profile):
                raise ValueError("Complex source profiles unsupported")
            self.sources[field] = wp.array(
                profile.reshape(3, -1), dtype=float, device=self.device
            )
            offsets = d[f"{name}/_time_offset_{field}"].reshape(3, -1)
            if not np.all(offsets == offsets[:, :1]):
                raise ValueError("Spatially varying source delays unsupported")
            # Match float32 source arithmetic, including the separate multiply/divide.
            half_step = np.float32(0.5 if field == "E" else 0.0)
            t = (
                np.arange(self.steps, dtype=np.float32)[None, :]
                + half_step
                + offsets[:, :1]
            ) * np.float32(m["dt_s"])
            period = np.float32(1 / src["frequency_hz"])
            phase = (
                np.float32(2 * np.pi) * t / period
                + np.float32(src["phase_shift"])
                + np.float32(src["profile_phase_shift"])
            )
            ramp = np.clip(
                t / np.float32(src["num_startup_periods"] / src["frequency_hz"]), 0, 1
            )
            wave = ramp * np.cos(phase) * np.float32(src["static_amplitude_factor"])
            self.waves[field] = wp.array(wave, dtype=float, device=self.device)
        self.monitors = []
        self.save_h_from = self.steps
        self.graphs = {}

    def reset(self):
        for a in (self.E, self.H, self.hprev, self.psi_e, self.psi_h, self.step):
            a.zero_()
        for monitor in self.monitors:
            monitor.reset()
        wp.synchronize_device(self.device)

    def launch_step(self):
        common = [
            self.hprev,
            self.save_h_from,
            *self.shape,
            self.source_x,
            self.courant,
            0,
        ]
        wp.launch(
            k.update_e,
            self.n,
            inputs=[
                self.E,
                self.H,
                self.eps,
                self.psi_e,
                self.coeffs["E"],
                self.sources["H"],
                self.waves["H"],
                self.step,
                *common,
            ],
            device=self.device,
        )
        wp.launch(
            k.update_h,
            self.n,
            inputs=[
                self.H,
                self.E,
                self.eps,
                self.psi_h,
                self.coeffs["H"],
                self.sources["E"],
                self.waves["E"],
                self.step,
                *common,
            ],
            device=self.device,
        )
        for monitor in self.monitors:
            monitor.launch(self)
        wp.launch(k.advance, 1, inputs=[self.step], device=self.device)

    def compile(self):
        start = time.perf_counter()
        self.launch_step()
        wp.synchronize_device(self.device)
        self.reset()
        for length in {self.graph_steps, self.steps % self.graph_steps} - {0}:
            with wp.ScopedCapture(device=self.device) as capture:
                for _ in range(length):
                    self.launch_step()
            self.graphs[length] = capture.graph
        self.reset()
        return time.perf_counter() - start

    def run(self):
        for _ in range(self.steps // self.graph_steps):
            wp.capture_launch(self.graphs[self.graph_steps])
        tail = self.steps % self.graph_steps
        if tail:
            wp.capture_launch(self.graphs[tail])
        wp.synchronize_device(self.device)

    def fields(self):
        return {
            name: arr.numpy().reshape(3, *self.shape)
            for name, arr in [("E", self.E), ("H", self.H)]
        }
