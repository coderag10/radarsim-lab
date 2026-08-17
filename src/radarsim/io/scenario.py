from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

# Kept as plain strings (not imported motion-model classes) so `io` stays
# a pure data layer -- `core.world.World.from_scenario` is what actually
# maps these names to `radarsim.targets` classes.
_VALID_MODELS = frozenset({"constant_velocity", "constant_acceleration"})


@dataclass(frozen=True, slots=True)
class TargetSpec:
    """One target's initial state and motion model, as declared in a scenario YAML file."""

    id: str
    model: str
    position: tuple[float, ...]
    velocity: tuple[float, ...]
    acceleration: tuple[float, ...]
    rcs: float


@dataclass(frozen=True, slots=True)
class ScenarioConfig:
    """Parsed contents of a scenario YAML file: simulation settings + target definitions.

    See scenarios/ for example files.
    """

    duration: float
    timestep: float
    seed: int
    targets: tuple[TargetSpec, ...]


def load_scenario(path: Path) -> ScenarioConfig:
    """Load and validate a scenario YAML file into a `ScenarioConfig`.

    Raises `ValueError` (not a bare `KeyError`/`TypeError`) with a
    specific message for each missing or malformed field -- scenario
    files are user-authored input, so this is the pipeline's one real
    validation boundary (see docs/ARCHITECTURE.md).
    """
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: scenario file must be a YAML mapping")

    simulation = raw.get("simulation")
    if not isinstance(simulation, dict):
        raise ValueError(f"{path}: missing or invalid 'simulation' block")

    duration = _require_number(simulation, "duration", path, "simulation")
    timestep = _require_number(simulation, "timestep", path, "simulation")
    seed = simulation.get("seed")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError(f"{path}: 'simulation.seed' must be an integer")

    raw_targets = raw.get("targets")
    if not isinstance(raw_targets, list) or not raw_targets:
        raise ValueError(f"{path}: 'targets' must be a non-empty list")

    targets = tuple(_parse_target(spec, index, path) for index, spec in enumerate(raw_targets))

    return ScenarioConfig(duration=duration, timestep=timestep, seed=seed, targets=targets)


def _require_number(block: dict[str, Any], key: str, path: Path, block_name: str) -> float:
    value = block.get(key)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{path}: '{block_name}.{key}' must be a number")
    return float(value)


def _require_vector(spec: dict[str, Any], key: str, path: Path, index: int) -> tuple[float, ...]:
    value = spec.get(key)
    if not isinstance(value, list) or not value:
        raise ValueError(f"{path}: targets[{index}].{key} must be a non-empty list of numbers")
    if not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in value):
        raise ValueError(f"{path}: targets[{index}].{key} must contain only numbers")
    return tuple(float(v) for v in value)


def _parse_target(spec: Any, index: int, path: Path) -> TargetSpec:
    if not isinstance(spec, dict):
        raise ValueError(f"{path}: targets[{index}] must be a mapping")

    target_id = spec.get("id")
    if not isinstance(target_id, str) or not target_id:
        raise ValueError(f"{path}: targets[{index}].id must be a non-empty string")

    model = spec.get("model")
    if model not in _VALID_MODELS:
        raise ValueError(
            f"{path}: targets[{index}].model must be one of "
            f"{sorted(_VALID_MODELS)}, got {model!r}"
        )

    position = _require_vector(spec, "position", path, index)
    velocity = _require_vector(spec, "velocity", path, index)
    if len(position) != len(velocity):
        raise ValueError(
            f"{path}: targets[{index}] position/velocity must have the same dimensionality"
        )

    if "acceleration" in spec:
        acceleration = _require_vector(spec, "acceleration", path, index)
    else:
        acceleration = tuple(0.0 for _ in position)
    if len(acceleration) != len(position):
        raise ValueError(
            f"{path}: targets[{index}].acceleration must match position dimensionality"
        )

    rcs = spec.get("rcs", 1.0)
    if not isinstance(rcs, (int, float)) or isinstance(rcs, bool):
        raise ValueError(f"{path}: targets[{index}].rcs must be a number")

    return TargetSpec(
        id=target_id,
        model=model,
        position=position,
        velocity=velocity,
        acceleration=acceleration,
        rcs=float(rcs),
    )
