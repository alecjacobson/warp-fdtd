"""Pure-Warp structured block curls with sparse electric trace/mass operations."""

import numpy as np
import warp as wp

wp.set_module_options({"enable_backward": False, "fast_math": False})


@wp.func
def e_index(block: int, c: int, p: wp.vec3i, n: int):
    dims = wp.vec3i(n + 1, n + 1, n + 1)
    dims[c] = n
    count = n * (n + 1) * (n + 1)
    return block * 3 * count + c * count + (p[0] * dims[1] + p[1]) * dims[2] + p[2]


@wp.func
def h_index(block: int, c: int, p: wp.vec3i, n: int):
    dims = wp.vec3i(n, n, n)
    dims[c] = n + 1
    count = (n + 1) * n * n
    return block * 3 * count + c * count + (p[0] * dims[1] + p[1]) * dims[2] + p[2]


@wp.kernel
def advance(counter: wp.array(dtype=int)):
    counter[0] += 1


def make_kernels(double=False):
    dtype = wp.float64 if double else wp.float32

    @wp.kernel
    def expand(
        offset: wp.array(dtype=int),
        index: wp.array(dtype=int),
        value: wp.array(dtype=dtype),
        e: wp.array(dtype=dtype),
        local: wp.array(dtype=dtype),
    ):
        i = wp.tid()
        v = dtype(0.0)
        for j in range(offset[i], offset[i + 1]):
            v += value[j] * e[index[j]]
        local[i] = v

    @wp.kernel
    def magnetic(
        e: wp.array(dtype=dtype),
        h: wp.array(dtype=dtype),
        widths: wp.array(dtype=dtype),
        damping: wp.array(dtype=dtype),
        n: int,
        dt: dtype,
    ):
        i = wp.tid()
        count = (n + 1) * n * n
        block = i // (3 * count)
        c = (i // count) % 3
        dims = wp.vec3i(n, n, n)
        dims[c] = n + 1
        j = i % count
        p = wp.vec3i(j // (dims[1] * dims[2]), (j // dims[2]) % dims[1], j % dims[2])
        a = (c + 1) % 3
        b = (c + 2) % 3
        pa = p
        pa[a] += 1
        pb = p
        pb[b] += 1
        curl = (
            e[e_index(block, b, pa, n)]
            - e[e_index(block, b, p, n)]
            - e[e_index(block, a, pb, n)]
            + e[e_index(block, a, p, n)]
        )
        q = dt * damping[i] * dtype(0.5)
        h[i] = ((dtype(1.0) - q) * h[i] - dt / widths[block] * curl) / (dtype(1.0) + q)

    @wp.kernel
    def weak_curl(
        h: wp.array(dtype=dtype),
        rhs: wp.array(dtype=dtype),
        widths: wp.array(dtype=dtype),
        n: int,
    ):
        i = wp.tid()
        count = n * (n + 1) * (n + 1)
        block = i // (3 * count)
        c = (i // count) % 3
        dims = wp.vec3i(n + 1, n + 1, n + 1)
        dims[c] = n
        j = i % count
        p = wp.vec3i(j // (dims[1] * dims[2]), (j // dims[2]) % dims[1], j % dims[2])
        a = (c + 1) % 3
        b = (c + 2) % 3
        va = dtype(0.0)
        vb = dtype(0.0)
        if p[a] < n:
            va += h[h_index(block, b, p, n)]
        if p[a] > 0:
            pm = p
            pm[a] -= 1
            va -= h[h_index(block, b, pm, n)]
        if p[b] < n:
            vb += h[h_index(block, a, p, n)]
        if p[b] > 0:
            pm = p
            pm[b] -= 1
            vb -= h[h_index(block, a, pm, n)]
        wa = dtype(1.0)
        wb = dtype(1.0)
        if p[b] == 0 or p[b] == n:
            wa = dtype(0.5)
        if p[a] == 0 or p[a] == n:
            wb = dtype(0.5)
        rhs[i] = widths[block] * widths[block] * (wa * va - wb * vb)

    @wp.kernel
    def electric(
        offset: wp.array(dtype=int),
        index: wp.array(dtype=int),
        value: wp.array(dtype=dtype),
        rhs: wp.array(dtype=dtype),
        e: wp.array(dtype=dtype),
        damping: wp.array(dtype=dtype),
        source: wp.array(dtype=dtype),
        wave: wp.array(dtype=dtype),
        counter: wp.array(dtype=int),
        dt: dtype,
    ):
        i = wp.tid()
        v = dtype(0.0)
        for j in range(offset[i], offset[i + 1]):
            v += value[j] * rhs[index[j]]
        q = dt * damping[i] * dtype(0.5)
        e[i] = ((dtype(1.0) - q) * e[i] + dt * (v + wave[counter[0]] * source[i])) / (
            dtype(1.0) + q
        )

    return expand, magnetic, weak_curl, electric


K32 = make_kernels()
K64 = make_kernels(True)


class DrivenBlocks:
    def __init__(
        self,
        op,
        problem=None,
        steps=1,
        cfl=0.7,
        device="cuda:0",
        precision="float32",
        coupling="factorized",
    ):
        self.operator = op
        self.problem = problem
        self.device = wp.get_device(device)
        self.precision = precision
        self.dtype = wp.float64 if precision == "float64" else wp.float32
        self.dt = cfl * op.fine_h * np.sqrt(op.min_epsilon) / np.sqrt(3)
        self.kernels = K64 if precision == "float64" else K32
        self.arrays = []

        def array(x, dtype=None):
            a = wp.array(
                np.asarray(x, dtype=np.int32 if dtype is int else precision),
                device=self.device,
            )
            self.arrays.append(a)
            return a

        def zeros(n):
            return array(np.zeros(n))

        def csr(matrix):
            return (
                array(matrix.indptr, int),
                array(matrix.indices, int),
                array(matrix.data),
            )

        self.p = csr(op.p)
        if coupling not in ("factorized", "fused"):
            raise ValueError("coupling must be factorized or fused")
        self.coupling = coupling
        self.gather = csr(op.me_inverse if coupling == "factorized" else op.gather_e)
        self.restrict = csr(op.p.T.tocsr()) if coupling == "factorized" else None
        self.master_rhs = zeros(op.me.shape[0]) if coupling == "factorized" else None
        self.e = zeros(op.me.shape[0])
        self.h = zeros(len(op.mh))
        self.local_e = zeros(len(op.me_full))
        self.rhs = zeros(len(op.me_full))
        self.widths = array([g.h for g in op.regions])
        self.counter = array([0], int)
        self.n = op.mesh.cells_per_block
        if problem is None:
            sigma_e = np.zeros(len(self.e))
            sigma_h = np.zeros(len(self.h))
            source = np.zeros(len(self.e))
            wave = np.zeros(steps + 64)
        else:
            sigma_e = problem.damping(op.master_coords)
            sigma_h = problem.damping(op.hcoords)
            volumes = np.concatenate(
                [g.me / problem.sample(g.ecoords, g.h) for g in op.regions]
            )
            source = op.gather_e @ (
                volumes * problem.source(op.ecoords, op.e_components)
            )
            wave = problem.waveform((np.arange(steps + 64) + 0.5) * self.dt)
        self.sigma_e = array(sigma_e)
        self.sigma_h = array(sigma_h)
        self.source = array(source)
        self.wave = array(wave)
        self.graph = None
        self.graph_steps = 32

    @property
    def owned_bytes(self):
        return sum(int(a.capacity) for a in self.arrays)

    def reset(self):
        for a in (
            self.e,
            self.h,
            self.local_e,
            self.rhs,
            self.counter,
            self.master_rhs,
        ):
            if a is None:
                continue
            a.zero_()

    def step(self):
        expand, magnetic, weak, electric = self.kernels
        wp.launch(
            expand,
            len(self.local_e),
            inputs=[*self.p, self.e, self.local_e],
            device=self.device,
        )
        wp.launch(
            magnetic,
            len(self.h),
            inputs=[self.local_e, self.h, self.widths, self.sigma_h, self.n, self.dt],
            device=self.device,
        )
        wp.launch(
            weak,
            len(self.rhs),
            inputs=[self.h, self.rhs, self.widths, self.n],
            device=self.device,
        )
        rhs = self.rhs
        if self.restrict is not None:
            wp.launch(
                expand,
                len(self.e),
                inputs=[*self.restrict, self.rhs, self.master_rhs],
                device=self.device,
            )
            rhs = self.master_rhs
        wp.launch(
            electric,
            len(self.e),
            inputs=[
                *self.gather,
                rhs,
                self.e,
                self.sigma_e,
                self.source,
                self.wave,
                self.counter,
                self.dt,
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

    def fields(self):
        return self.e.numpy().copy(), self.h.numpy().copy()
