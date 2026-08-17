from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class ScenarioConfig:
    """Parsed contents of a scenario YAML file: simulation settings + target definitions.

    See scenarios/ for example files (schema finalized in Phase 1).
    """

    duration: float
    timestep: float
    seed: int


def load_scenario(path: Path) -> ScenarioConfig:
    """Load and validate a scenario YAML file into a `ScenarioConfig`."""
    raise NotImplementedError
