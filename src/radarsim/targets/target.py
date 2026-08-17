from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from radarsim.targets.motion_models import MotionModel
from radarsim.targets.rcs import RCSModel
from radarsim.types import GroundTruth


@dataclass
class Target:
    """A single simulated target: current kinematic state plus the
    motion/RCS models that evolve it.

    Owns mutable state on purpose (`core.world.World` steps a
    collection of these each tick) — `snapshot`/`step` hand out
    immutable `GroundTruth` copies so callers can't corrupt that state
    by mutating a returned array.
    """

    target_id: str
    position: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray
    motion_model: MotionModel
    rcs_model: RCSModel

    def snapshot(self, timestamp: float) -> GroundTruth:
        """Return the current state as a `GroundTruth`, without advancing it."""
        return GroundTruth(
            timestamp=timestamp,
            target_id=self.target_id,
            position=self.position.copy(),
            velocity=self.velocity.copy(),
            acceleration=self.acceleration.copy(),
            rcs=self.rcs_model.rcs(aspect_angle=0.0),
        )

    def step(self, dt: float, timestamp: float) -> GroundTruth:
        """Advance state by one timestep via `motion_model` and return the new `GroundTruth`."""
        self.position, self.velocity, self.acceleration = self.motion_model.step(
            self.position, self.velocity, self.acceleration, dt
        )
        return self.snapshot(timestamp)
