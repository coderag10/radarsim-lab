from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True, eq=False)
class RadarMeasurement:
    """A single noisy radar observation, as produced by `radarsim.radar`.

    This is the boundary type: everything downstream (`detection`,
    `tracking`, `fusion`) sees only measurements like this, never the
    `GroundTruth` that generated them.

    `eq=False`: see `radarsim.types.state.GroundTruth` docstring.
    """

    timestamp: float
    sensor_id: str
    range: float
    radial_velocity: float
    angle: float
    covariance: np.ndarray
