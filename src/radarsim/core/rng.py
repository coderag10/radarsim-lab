from __future__ import annotations

import hashlib

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
        self._seed = seed
        self._streams: dict[str, np.random.Generator] = {}

    def spawn(self, name: str) -> np.random.Generator:
        """Return an independent, reproducible sub-stream for `name`.

        Using named sub-streams (e.g. "targets", "radar",
        "particle_filter") keeps components decorrelated even though
        they all trace back to one top-level seed. The mapping from
        name to sub-stream is order-independent (derived from a hash
        of `name`, not call order) and calls are cached, so repeated
        `spawn(name)` calls return the same generator, continuing its
        state rather than resetting it.
        """
        if name not in self._streams:
            name_hash = int.from_bytes(hashlib.sha256(name.encode()).digest()[:4], "big")
            seed_sequence = np.random.SeedSequence(entropy=[self._seed, name_hash])
            self._streams[name] = np.random.Generator(np.random.PCG64(seed_sequence))
        return self._streams[name]
