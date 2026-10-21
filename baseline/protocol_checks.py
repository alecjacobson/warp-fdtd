"""Regression checks for the historical example's source/material conventions."""

import jax
import jax.numpy as jnp
import numpy as np

from .common import ROOT, write_json
from .scene import PaperCWProfile, Settings, build


def main():
    np.testing.assert_allclose(
        np.asarray(
            PaperCWProfile().get_amplitude(jnp.array([0.0, 0.25, 0.5, 1.0]), 1.0)
        ),
        [1, 0, -1, 1],
        atol=1e-6,
    )
    checks = {"abrupt_cosine": True}
    for parameterization in ("binary", "brush"):
        objects, _, params, _, _ = build(
            Settings(
                backend="cpu",
                spacing_nm=50,
                voxel_nm=50,
                time_fs=50,
                source_profile="paper",
                parameterization=parameterization,
            )
        )
        device = objects.devices[0]
        x = params[device.name]
        # Both phases must agree that 0 means silicon (index 1), 1 means air (index 0).
        np.testing.assert_array_equal(
            np.asarray(device(jnp.zeros_like(x))), np.ones(x.shape)
        )
        np.testing.assert_array_equal(
            np.asarray(device(jnp.ones_like(x))), np.zeros(x.shape)
        )
        gradient = jax.grad(lambda x: jnp.sum(device(x)))(jnp.full_like(x, 0.3))
        assert np.all(np.isfinite(gradient)) and np.all(np.asarray(gradient) > 0), (
            "Historical STE orientation changed"
        )
        checks[parameterization + "_orientation_and_ste"] = True
    write_json(ROOT / "results/protocol_checks.json", checks)
    print(checks)


if __name__ == "__main__":
    main()
