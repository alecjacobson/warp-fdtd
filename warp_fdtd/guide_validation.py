"""Full-field finiteness checks without copying large grids to the CPU."""

import warp as wp


@wp.kernel
def count_nonfinite(field: wp.array(dtype=float), count: wp.array(dtype=int)):
    if not wp.isfinite(field[wp.tid()]):
        wp.atomic_add(count, 0, 1)


def assert_finite(solver):
    count = wp.zeros(1, dtype=int, device=solver.device)
    for field in (solver.e, solver.h):
        wp.launch(
            count_nonfinite, len(field), inputs=[field, count], device=solver.device
        )
    n = int(count.numpy()[0])
    if n:
        raise FloatingPointError(f"{n} nonfinite field values")
