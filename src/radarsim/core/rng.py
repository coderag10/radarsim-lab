from __future__ import annotations

import numpy as np


class SeededRNG:
    """Owns the single seeded RNG stream for a simulation run.

    Every stochastic component (target motion noise, radar measurement
    noise, particle filter resampling, ...) must draw from a generator
    obtained here rather than calling `numpy.random` directly, so a
    run is fully reproducible from its seed alone. See "Determinism"
    in docs/ARCHITECTURE.md.
    """

    def __init__(self, seed: int) -> None:
        raise NotImplementedError

    def spawn(self, name: str) -> np.random.Generator:
        """Return an independent, reproducible sub-stream for `name`.

        Using named sub-streams (e.g. "targets", "radar",
        "particle_filter") keeps components decorrelated even though
        they all trace back to one top-level seed.
        """
        raise NotImplementedError
