"""Pure Warp Yee half-steps, CPML and source injection in normalized units."""

import warp as wp

wp.set_module_options({"enable_backward": False, "fast_math": False})


@wp.func
def sample(
    f: wp.array2d(dtype=float),
    x: int,
    y: int,
    z: int,
    nx: int,
    ny: int,
    nz: int,
    periodic: int,
):
    if periodic != 0:
        x = (x + nx) % nx
        y = (y + ny) % ny
        z = (z + nz) % nz
    if x < 0 or y < 0 or z < 0 or x >= nx or y >= ny or z >= nz:
        return wp.vec3(0.0)
    i = (x * ny + y) * nz + z
    return wp.vec3(f[0, i], f[1, i], f[2, i])


@wp.func
def cpml(
    d: float,
    psi: wp.array2d(dtype=float),
    coeff: wp.array2d(dtype=wp.vec3),
    axis: int,
    position: int,
    slot: int,
    i: int,
):
    c = coeff[axis, position]
    if c[0] != 0.0:
        p = c[1] * psi[slot, i] + c[0] * d
        psi[slot, i] = p
        return c[2] * d + p
    return c[2] * d


def make_update(electric):
    @wp.kernel
    def update(
        f: wp.array2d(dtype=float),
        g: wp.array2d(dtype=float),
        inv_eps: wp.array(dtype=float),
        psi: wp.array2d(dtype=float),
        coeff: wp.array2d(dtype=wp.vec3),
        source: wp.array2d(dtype=float),
        waves: wp.array2d(dtype=float),
        step: wp.array(dtype=int),
        hprev: wp.array2d(dtype=float),
        save_h_from: int,
        nx: int,
        ny: int,
        nz: int,
        source_x: int,
        courant: float,
        periodic: int,
    ):
        i = wp.tid()
        z = i % nz
        y = (i // nz) % ny
        x = i // (ny * nz)
        v = sample(g, x, y, z, nx, ny, nz, periodic)
        if wp.static(electric):
            dx = v - sample(g, x - 1, y, z, nx, ny, nz, periodic)
            dy = v - sample(g, x, y - 1, z, nx, ny, nz, periodic)
            dz = v - sample(g, x, y, z - 1, nx, ny, nz, periodic)
        else:
            dx = sample(g, x + 1, y, z, nx, ny, nz, periodic) - v
            dy = sample(g, x, y + 1, z, nx, ny, nz, periodic) - v
            dz = sample(g, x, y, z + 1, nx, ny, nz, periodic) - v
        dxz = cpml(dx[2], psi, coeff, 0, x, 0, i)
        dxy = cpml(dx[1], psi, coeff, 0, x, 1, i)
        dyx = cpml(dy[0], psi, coeff, 1, y, 2, i)
        dyz = cpml(dy[2], psi, coeff, 1, y, 3, i)
        dzy = cpml(dz[1], psi, coeff, 2, z, 4, i)
        dzx = cpml(dz[0], psi, coeff, 2, z, 5, i)
        curl = wp.vec3(dyz - dzy, dzx - dxz, dxy - dyx)
        old = wp.vec3(f[0, i], f[1, i], f[2, i])
        t = step[0]
        if wp.static(electric):
            value = old + courant * curl * inv_eps[i]
            if x == source_x:
                s = y * nz + z
                value[1] += source[2, s] * waves[2, t] * courant * inv_eps[i]
                value[2] -= source[1, s] * waves[1, t] * courant * inv_eps[i]
        else:
            value = old - courant * curl
            if t >= save_h_from:
                hprev[0, i] = old[0]
                hprev[1, i] = old[1]
                hprev[2, i] = old[2]
            if x == source_x:
                s = y * nz + z
                value[1] -= source[2, s] * waves[2, t] * courant
                value[2] += source[1, s] * waves[1, t] * courant
        f[0, i] = value[0]
        f[1, i] = value[1]
        f[2, i] = value[2]

    return update


update_e = make_update(True)
update_h = make_update(False)


@wp.kernel
def advance(step: wp.array(dtype=int)):
    step[0] += 1
