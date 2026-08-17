from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SimulationClock:
    """Owns simulated time: current time, timestep, and total duration.

    All stochastic/kinematic components step in lockstep with this
    clock rather than tracking their own notion of time.
    """

    timestep: float
    duration: float
    time: float = 0.0

    def step(self) -> float:
        """Advance simulated time by one `timestep` and return the new time."""
        self.time += self.timestep
        return self.time

    def is_finished(self) -> bool:
        """Return True once `time` has reached `duration`."""
        return self.time >= self.duration - 1e-9

    def reset(self) -> None:
        """Reset `time` to zero."""
        self.time = 0.0
