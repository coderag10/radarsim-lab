from __future__ import annotations

from pydantic import BaseModel


class ScenarioSummary(BaseModel):
    """Scenario metadata without running it."""

    path: str
    duration: float
    timestep: float
    seed: int
    num_targets: int
