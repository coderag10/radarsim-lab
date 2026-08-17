from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from radarsim.core.clock import SimulationClock
from radarsim.core.rng import SeededRNG
from radarsim.io.scenario import ScenarioConfig
from radarsim.targets.motion_models import (
    ConstantAccelerationModel,
    ConstantVelocityModel,
    MotionModel,
)
from radarsim.targets.rcs import ConstantRCS
from radarsim.targets.target import Target
from radarsim.types import GroundTruth

_MOTION_MODELS: dict[str, type[MotionModel]] = {
    "constant_velocity": ConstantVelocityModel,
    "constant_acceleration": ConstantAccelerationModel,
}


@dataclass
class World:
    """Owns the simulation clock, RNG, and the set of live targets.

    `core` is the one module allowed to depend on both `io` (for
    `ScenarioConfig`) and `targets` (to build `Target` instances) --
    see the module table in docs/ARCHITECTURE.md. Advancing the world
    by one timestep produces the `GroundTruth` stream that
    `radarsim.radar` will observe once it exists.
    """

    clock: SimulationClock
    rng: SeededRNG
    targets: list[Target] = field(default_factory=list)

    @classmethod
    def from_scenario(cls, config: ScenarioConfig) -> World:
        """Build a `World` from a parsed scenario: clock, RNG, and one `Target` per spec."""
        clock = SimulationClock(timestep=config.timestep, duration=config.duration)
        rng = SeededRNG(seed=config.seed)

        targets: list[Target] = []
        for spec in config.targets:
            model_cls = _MOTION_MODELS.get(spec.model)
            if model_cls is None:
                raise ValueError(
                    f"Unknown motion model {spec.model!r} for target {spec.id!r}; "
                    f"expected one of {sorted(_MOTION_MODELS)}"
                )
            targets.append(
                Target(
                    target_id=spec.id,
                    position=np.array(spec.position, dtype=float),
                    velocity=np.array(spec.velocity, dtype=float),
                    acceleration=np.array(spec.acceleration, dtype=float),
                    motion_model=model_cls(),
                    rcs_model=ConstantRCS(spec.rcs),
                )
            )

        return cls(clock=clock, rng=rng, targets=targets)

    def step(self) -> list[GroundTruth]:
        """Advance every target by one timestep, returning their new ground truth states."""
        timestamp = self.clock.step()
        return [target.step(self.clock.timestep, timestamp) for target in self.targets]

    def run(self) -> list[list[GroundTruth]]:
        """Run the simulation to completion.

        Returns one entry per timestep, starting with the t=0 initial
        state (before any stepping) so downstream consumers (radar,
        metrics) have a true starting point, not just post-step samples.
        """
        initial = [target.snapshot(self.clock.time) for target in self.targets]
        history = [initial]
        while not self.clock.is_finished():
            history.append(self.step())
        return history
