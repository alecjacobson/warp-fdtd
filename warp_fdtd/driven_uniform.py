"""Direct structured Yee baseline for the identical driven dielectric guide."""

import numpy as np
import warp as wp
from .driven_blocks import advance

wp.set_module_options({"enable_backward": False, "fast_math": False})


@wp.func
def e_index(c: int, p: wp.vec3i, n: wp.vec3i):
    dims = n + wp.vec3i(1, 1, 1)
    dims[c] -= 1
    offset = int(0)
    if c >= 1:
        offset += n[0] * (n[1] + 1) * (n[2] + 1)
    if c == 2:
        offset += (n[0] + 1) * n[1] * (n[2] + 1)
    return offset + (p[0] * dims[1] + p[1]) * dims[2] + p[2]


@wp.func
def h_index(c: int, p: wp.vec3i, n: wp.vec3i):
    dims = n
    dims[c] += 1
    offset = int(0)
    if c >= 1:
        offset += (n[0] + 1) * n[1] * n[2]
    if c == 2:
        offset += n[0] * (n[1] + 1) * n[2]
    return offset + (p[0] * dims[1] + p[1]) * dims[2] + p[2]


@wp.func
def sigma(p: wp.vec3, lengths: wp.vec3, width: float, rate: float):
    result = float(0.0)
    for a in range(3):
        depth = wp.max(0.0, 1.0 - wp.min(p[a], lengths[a] - p[a]) / width)
        result += rate * depth * depth * depth
    return result


@wp.func
def epsilon(
    p: wp.vec3,
    h: float,
    lengths: wp.vec3,
    core: wp.vec2,
    eps_core: float,
    eps_clad: float,
):
    f = float(1.0)
    for a in range(1, 3):
        lo = lengths[a] * 0.5 - core[a - 1] * 0.5
        hi = lengths[a] * 0.5 + core[a - 1] * 0.5
        f *= wp.clamp(
            (wp.min(p[a] + h * 0.5, hi) - wp.max(p[a] - h * 0.5, lo)) / h, 0.0, 1.0
        )
    return eps_clad + (eps_core - eps_clad) * f


@wp.func
def epsilon_normal(
    p: wp.vec3,
    h: float,
    lengths: wp.vec3,
    core: wp.vec2,
    eps_core: float,
    eps_clad: float,
    component: int,
):
    # Harmonic normal / arithmetic tangential averaging for aligned interfaces.
    # Sequential mixing at rectangular corners is a control, not full-tensor
    # perturbative subpixel smoothing for arbitrary geometry.
    fraction = wp.vec2(0.0, 0.0)
    for a in range(1, 3):
        lo = lengths[a] * 0.5 - core[a - 1] * 0.5
        hi = lengths[a] * 0.5 + core[a - 1] * 0.5
        fraction[a - 1] = wp.clamp(
            (wp.min(p[a] + h * 0.5, hi) - wp.max(p[a] - h * 0.5, lo)) / h, 0.0, 1.0
        )
    if component == 0:
        return eps_clad + (eps_core - eps_clad) * fraction[0] * fraction[1]
    normal = component - 1
    other = 1 - normal
    inside = eps_clad + (eps_core - eps_clad) * fraction[other]
    return 1.0 / ((1.0 - fraction[normal]) / eps_clad + fraction[normal] / inside)


@wp.kernel
def magnetic(
    e: wp.array(dtype=float),
    hfield: wp.array(dtype=float),
    n: wp.vec3i,
    h: float,
    dt: float,
    lengths: wp.vec3,
    sponge_width: float,
    sponge_rate: float,
):
    i = wp.tid()
    first = (n[0] + 1) * n[1] * n[2]
    second = n[0] * (n[1] + 1) * n[2]
    c = int(0)
    j = i
    if i >= first + second:
        c = 2
        j = i - first - second
    elif i >= first:
        c = 1
        j = i - first
    dims = n
    dims[c] += 1
    p = wp.vec3i(j // (dims[1] * dims[2]), (j // dims[2]) % dims[1], j % dims[2])
    a = (c + 1) % 3
    b = (c + 2) % 3
    pa = p
    pa[a] += 1
    pb = p
    pb[b] += 1
    curl = (
        e[e_index(b, pa, n)]
        - e[e_index(b, p, n)]
        - e[e_index(a, pb, n)]
        + e[e_index(a, p, n)]
    )
    xyz = (wp.vec3(float(p[0]), float(p[1]), float(p[2])) + wp.vec3(0.5, 0.5, 0.5)) * h
    xyz[c] -= 0.5 * h
    q = 0.5 * dt * sigma(xyz, lengths, sponge_width, sponge_rate)
    hfield[i] = ((1.0 - q) * hfield[i] - dt / h * curl) / (1.0 + q)


@wp.kernel
def electric(
    hfield: wp.array(dtype=float),
    e: wp.array(dtype=float),
    n: wp.vec3i,
    h: float,
    dt: float,
    lengths: wp.vec3,
    core: wp.vec2,
    eps_core: float,
    eps_clad: float,
    sponge_width: float,
    sponge_rate: float,
    src: wp.vec3,
    src_sigma: wp.vec3,
    wave: wp.array(dtype=float),
    counter: wp.array(dtype=int),
    normal_average: int,
):
    i = wp.tid()
    first = n[0] * (n[1] + 1) * (n[2] + 1)
    second = (n[0] + 1) * n[1] * (n[2] + 1)
    c = int(0)
    j = i
    if i >= first + second:
        c = 2
        j = i - first - second
    elif i >= first:
        c = 1
        j = i - first
    dims = n + wp.vec3i(1, 1, 1)
    dims[c] -= 1
    p = wp.vec3i(j // (dims[1] * dims[2]), (j // dims[2]) % dims[1], j % dims[2])
    a = (c + 1) % 3
    b = (c + 2) % 3
    if p[a] > 0 and p[a] < n[a] and p[b] > 0 and p[b] < n[b]:
        pa = p
        pa[a] -= 1
        pb = p
        pb[b] -= 1
        curl = (
            hfield[h_index(b, p, n)]
            - hfield[h_index(b, pa, n)]
            - hfield[h_index(a, p, n)]
            + hfield[h_index(a, pb, n)]
        )
        xyz = wp.vec3(float(p[0]), float(p[1]), float(p[2])) * h
        xyz[c] += 0.5 * h
        ep = epsilon(xyz, h, lengths, core, eps_core, eps_clad)
        if normal_average == 1:
            ep = epsilon_normal(xyz, h, lengths, core, eps_core, eps_clad, c)
        current = float(0.0)
        if c == 1:
            d = wp.cw_div(xyz - src, src_sigma)
            current = wp.exp(-0.5 * wp.dot(d, d)) * wave[counter[0]]
        q = 0.5 * dt * sigma(xyz, lengths, sponge_width, sponge_rate)
        e[i] = ((1.0 - q) * e[i] + dt * (curl / h + current) / ep) / (1.0 + q)


@wp.kernel
def magnetic_xyz(
    e: wp.array(dtype=float),
    hfield: wp.array(dtype=float),
    n: wp.vec3i,
    h: float,
    dt: float,
    lengths: wp.vec3,
    sponge_width: float,
    sponge_rate: float,
):
    ix, iy, iz = wp.tid()
    p = wp.vec3i(ix, iy, iz)
    for c in range(3):
        a = (c + 1) % 3
        b = (c + 2) % 3
        if p[c] <= n[c] and p[a] < n[a] and p[b] < n[b]:
            i = h_index(c, p, n)
            pa = p
            pa[a] += 1
            pb = p
            pb[b] += 1
            curl = (
                e[e_index(b, pa, n)]
                - e[e_index(b, p, n)]
                - e[e_index(a, pb, n)]
                + e[e_index(a, p, n)]
            )
            xyz = (
                wp.vec3(float(p[0]), float(p[1]), float(p[2])) + wp.vec3(0.5, 0.5, 0.5)
            ) * h
            xyz[c] -= 0.5 * h
            q = 0.5 * dt * sigma(xyz, lengths, sponge_width, sponge_rate)
            hfield[i] = ((1.0 - q) * hfield[i] - dt / h * curl) / (1.0 + q)


@wp.kernel
def electric_xyz(
    hfield: wp.array(dtype=float),
    e: wp.array(dtype=float),
    n: wp.vec3i,
    h: float,
    dt: float,
    lengths: wp.vec3,
    core: wp.vec2,
    eps_core: float,
    eps_clad: float,
    sponge_width: float,
    sponge_rate: float,
    src: wp.vec3,
    src_sigma: wp.vec3,
    wave: wp.array(dtype=float),
    counter: wp.array(dtype=int),
    normal_average: int,
):
    ix, iy, iz = wp.tid()
    p = wp.vec3i(ix, iy, iz)
    for c in range(3):
        a = (c + 1) % 3
        b = (c + 2) % 3
        if p[c] < n[c] and p[a] > 0 and p[a] < n[a] and p[b] > 0 and p[b] < n[b]:
            i = e_index(c, p, n)
            pa = p
            pa[a] -= 1
            pb = p
            pb[b] -= 1
            curl = (
                hfield[h_index(b, p, n)]
                - hfield[h_index(b, pa, n)]
                - hfield[h_index(a, p, n)]
                + hfield[h_index(a, pb, n)]
            )
            xyz = wp.vec3(float(p[0]), float(p[1]), float(p[2])) * h
            xyz[c] += 0.5 * h
            ep = epsilon(xyz, h, lengths, core, eps_core, eps_clad)
            if normal_average == 1:
                ep = epsilon_normal(xyz, h, lengths, core, eps_core, eps_clad, c)
            current = float(0.0)
            if c == 1:
                d = wp.cw_div(xyz - src, src_sigma)
                current = wp.exp(-0.5 * wp.dot(d, d)) * wave[counter[0]]
            q = 0.5 * dt * sigma(xyz, lengths, sponge_width, sponge_rate)
            e[i] = ((1.0 - q) * e[i] + dt * (curl / h + current) / ep) / (1.0 + q)


class DrivenUniform:
    def __init__(
        self,
        problem,
        spacing,
        steps,
        cfl=0.7,
        device="cuda:0",
        layout="xyz",
        normal_average=False,
    ):
        if layout not in ("xyz", "flat"):
            raise ValueError("layout must be xyz or flat")
        self.layout = layout
        self.normal_average = int(normal_average)
        self.problem = problem
        self.hspacing = spacing
        self.shape = tuple(int(round(x / spacing)) for x in problem.lengths)
        if any(
            abs(n * spacing - x) > 1e-10 for n, x in zip(self.shape, problem.lengths)
        ):
            raise ValueError("Spacing must divide domain")
        self.n = wp.vec3i(*self.shape)
        self.device = wp.get_device(device)
        self.dt = cfl * spacing * np.sqrt(problem.clad_epsilon) / np.sqrt(3)
        self.eshapes = [
            tuple(n if a == c else n + 1 for a, n in enumerate(self.shape))
            for c in range(3)
        ]
        self.hshapes = [
            tuple(n + 1 if a == c else n for a, n in enumerate(self.shape))
            for c in range(3)
        ]
        self.e = wp.zeros(
            sum(int(np.prod(s)) for s in self.eshapes), device=self.device
        )
        self.h = wp.zeros(
            sum(int(np.prod(s)) for s in self.hshapes), device=self.device
        )
        self.counter = wp.zeros(1, dtype=int, device=self.device)
        self.wave = wp.array(
            problem.waveform((np.arange(steps + 64) + 0.5) * self.dt).astype(
                np.float32
            ),
            device=self.device,
        )
        self.arrays = [self.e, self.h, self.counter, self.wave]
        self.graph = None
        self.graph_steps = 32

    @property
    def owned_bytes(self):
        return sum(int(a.capacity) for a in self.arrays)

    def reset(self):
        self.e.zero_()
        self.h.zero_()
        self.counter.zero_()

    def magnetic_step(self):
        p = self.problem
        wp.launch(
            magnetic_xyz if self.layout == "xyz" else magnetic,
            tuple(n + 1 for n in self.shape) if self.layout == "xyz" else len(self.h),
            inputs=[
                self.e,
                self.h,
                self.n,
                self.hspacing,
                self.dt,
                wp.vec3(*p.lengths),
                p.sponge_width,
                p.sponge_rate,
            ],
            device=self.device,
        )

    def step(self):
        p = self.problem
        self.magnetic_step()
        wp.launch(
            electric_xyz if self.layout == "xyz" else electric,
            tuple(n + 1 for n in self.shape) if self.layout == "xyz" else len(self.e),
            inputs=[
                self.h,
                self.e,
                self.n,
                self.hspacing,
                self.dt,
                wp.vec3(*p.lengths),
                wp.vec2(p.core_width, p.core_height),
                p.core_epsilon,
                p.clad_epsilon,
                p.sponge_width,
                p.sponge_rate,
                wp.vec3(p.source_x, p.width / 2, p.height / 2),
                wp.vec3(
                    p.source_sigma_x,
                    p.source_sigma_transverse,
                    p.source_sigma_transverse,
                ),
                self.wave,
                self.counter,
                self.normal_average,
            ],
            device=self.device,
        )
        wp.launch(advance, 1, inputs=[self.counter], device=self.device)

    def compile(self):
        self.step()
        wp.synchronize_device(self.device)
        self.reset()
        if self.device.is_cuda:
            with wp.ScopedCapture(device=self.device) as capture:
                for _ in range(self.graph_steps):
                    self.step()
            self.graph = capture.graph
        self.reset()

    def run(self, steps):
        if self.graph is None:
            for _ in range(steps):
                self.step()
        else:
            for _ in range(steps // self.graph_steps):
                wp.capture_launch(self.graph)
            for _ in range(steps % self.graph_steps):
                self.step()
        wp.synchronize_device(self.device)
