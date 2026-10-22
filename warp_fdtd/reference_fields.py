"""Incident-mode normalization for comparing fields on different physical meshes."""

import numpy as np


def incident_normalization(phasors, mode_e, mode_h, axis, area_weights, spacing_m):
    """Match the exported overlap and fix the eigenmode's arbitrary complex phase.

    Uniform FDTDX modes have unit *grid-weighted* power: their overlap omits
    the common dx^2 factor. Dividing by dx converts that convention to unit
    area-weighted incident power. The mode's dominant integrated E component
    fixes its phase independently of the propagated output field.
    """
    integrand = (
        np.cross(mode_e, phasors[3:].conj(), axis=0)
        + np.cross(phasors[:3].conj(), mode_h, axis=0)
    )[axis]
    alpha = np.sum(integrand * area_weights) / 4
    integrated_e = np.sum(mode_e * area_weights, axis=tuple(range(1, mode_e.ndim)))
    component = int(np.argmax(np.abs(integrated_e)))
    anchor = integrated_e[component]
    if (
        not np.isfinite(alpha)
        or abs(alpha) == 0
        or not np.isfinite(anchor)
        or abs(anchor) == 0
    ):
        raise ValueError("Degenerate incoming mode normalization")
    phase = anchor / abs(anchor)
    divisor = complex(np.conj(alpha) * phase * spacing_m)
    return {
        "overlap_conjugate_amplitude": [float(alpha.real), float(alpha.imag)],
        "canonical_mode_component": component,
        "canonical_mode_phase": [float(phase.real), float(phase.imag)],
        "field_divisor": [divisor.real, divisor.imag],
        "grid_weighted_incident_power": float(abs(alpha) ** 2),
        "area_weighted_incident_power": float(abs(alpha) ** 2 * spacing_m**2),
        "scope": "Normalize at the fixed incoming port only; no fit to output or reference fields",
    }
