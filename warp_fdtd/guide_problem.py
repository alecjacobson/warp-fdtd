"""Fixed 3D pulsed dielectric channel benchmark in normalized c=mu=1 units.

Lengths are micrometers; one time unit is one micrometer/c. The matched E/H
conductivity layer is a sponge, not CPML. A localized impressed current excites
both guided and radiating fields; no pure-mode or modal transmission claim.
"""

from dataclasses import asdict, dataclass
import numpy as np
from .block_mesh import BlockMesh, Box, GuidePolicy


@dataclass(frozen=True)
class GuideProblem:
    length: float = 4.8
    width: float = 3.2
    height: float = 3.2
    core_width: float = 0.4
    core_height: float = 0.4
    core_epsilon: float = 12.25
    clad_epsilon: float = 2.25
    wavelength: float = 1.55
    source_x: float = 0.8
    source_sigma_x: float = 0.10
    source_sigma_transverse: float = 0.12
    pulse_center: float = 3.0
    pulse_sigma: float = 1.0
    duration: float = 14.0
    sponge_width: float = 0.4
    sponge_rate: float = 10.0

    @property
    def lengths(self):
        return (self.length, self.width, self.height)

    def sample(self, coords, h):
        """Aligned-box volume fraction; identical dual averaging on every grid."""
        fractions = []
        for axis, width, center in (
            (1, self.core_width, self.width / 2),
            (2, self.core_height, self.height / 2),
        ):
            fractions.append(
                np.clip(
                    (
                        np.minimum(coords[:, axis] + h / 2, center + width / 2)
                        - np.maximum(coords[:, axis] - h / 2, center - width / 2)
                    )
                    / h,
                    0,
                    1,
                )
            )
        return (
            self.clad_epsilon
            + (self.core_epsilon - self.clad_epsilon) * fractions[0] * fractions[1]
        )

    def source(self, coords, components):
        delta = (coords - [self.source_x, self.width / 2, self.height / 2]) / [
            self.source_sigma_x,
            self.source_sigma_transverse,
            self.source_sigma_transverse,
        ]
        return np.exp(-0.5 * np.sum(delta**2, axis=1)) * (components == 1)

    def damping(self, coords):
        distance = np.minimum(coords, np.array(self.lengths) - coords)
        depth = np.maximum(0, 1 - distance / self.sponge_width)
        return self.sponge_rate * np.sum(depth**3, axis=1)

    def waveform(self, t):
        return np.exp(
            -0.5 * ((t - self.pulse_center) / self.pulse_sigma) ** 2
        ) * np.sin(2 * np.pi * t / self.wavelength)

    def parameters(self):
        return asdict(self)

    def mesh(self, finest=0.05, levels=2, block_cells=4, buffer=0.1):
        root_width = finest * block_cells * 2**levels
        shape = tuple(int(round(v / root_width)) for v in self.lengths)
        if any(abs(n * root_width - v) > 1e-10 for n, v in zip(shape, self.lengths)):
            raise ValueError("Domain must contain an integer number of root blocks")
        box = Box(
            (
                0,
                self.width / 2 - self.core_width / 2,
                self.height / 2 - self.core_height / 2,
            ),
            (
                self.length,
                self.width / 2 + self.core_width / 2,
                self.height / 2 + self.core_height / 2,
            ),
        )
        policy = GuidePolicy(
            (box,),
            finest,
            finest,
            root_width / block_cells,
            guide_buffer=buffer,
            growth=0.7,
        )
        return BlockMesh(
            shape, root_width, cells_per_block=block_cells, ratio=2, max_level=levels
        ).generate(policy)
