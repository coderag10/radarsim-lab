from __future__ import annotations

from dataclasses import dataclass

from radarsim.core.clock import SimulationClock
from radarsim.core.rng import SeededRNG
from radarsim.types import GroundTruth


@dataclass
class World:
    """Owns the simulation clock, RNG, and the set of live targets.

    Advancing the world by one timestep produces the `GroundTruth`
    stream that `radarsim.radar` observes.
    """

    clock: SimulationClock
    rng: SeededRNG

    def step(self) -> list[GroundTruth]:
        """Advance every target by one timestep, returning their new ground truth states."""
        raise NotImplementedError

    def run(self) -> list[list[GroundTruth]]:
        """Run the simulation to completion, returning ground truth states per timestep."""
        raise NotImplementedError
