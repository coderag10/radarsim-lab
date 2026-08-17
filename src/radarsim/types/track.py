from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto

import numpy as np


class TrackStatus(Enum):
    """Lifecycle state of a track, as maintained by the track manager."""

    TENTATIVE = auto()
    ACTIVE = auto()
    LOST = auto()


@dataclass(frozen=True, slots=True, eq=False)
class TrackEstimate:
    """A tracker's current belief about one target's state.

    Produced by `radarsim.tracking` (and refined by `radarsim.fusion`).
    Only `radarsim.metrics` is allowed to compare a `TrackEstimate`
    against the `GroundTruth` it's estimating.

    `eq=False`: see `radarsim.types.state.GroundTruth` docstring.
    """

    track_id: str
    timestamp: float
    state: np.ndarray
    covariance: np.ndarray
    status: TrackStatus
