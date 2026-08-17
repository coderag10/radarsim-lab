from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True, eq=False)
class GroundTruth:
    """True kinematic state of one simulated target at one timestep.

    Owned by `radarsim.core` / `radarsim.targets`. Algorithm modules
    (`detection`, `tracking`, `fusion`) must never import this type —
    only `radarsim.radar` (to generate measurements) and
    `radarsim.metrics` (to score estimates against truth) may. See
    docs/ARCHITECTURE.md.

    `eq=False`: NumPy array fields make the default dataclass equality
    ambiguous (`array == array` returns an array, not a bool). Compare
    fields explicitly (e.g. via `numpy.testing`) where needed.
    """

    timestamp: float
    target_id: str
    position: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray
    rcs: float
