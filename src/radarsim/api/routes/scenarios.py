from __future__ import annotations

import os
from pathlib import Path

from fastapi import APIRouter, HTTPException

from radarsim.api.schemas.scenario import ScenarioSummary
from radarsim.io.scenario import load_scenario

router = APIRouter(prefix="/scenarios", tags=["scenarios"])


def _scenarios_root() -> Path:
    """Resolved fresh per call (not cached) so `RADARSIM_SCENARIOS_DIR`
    can be pointed elsewhere per-request in tests, or operationally."""
    return Path(os.environ.get("RADARSIM_SCENARIOS_DIR", "scenarios"))


def _summarize(path: Path) -> ScenarioSummary:
    config = load_scenario(path)
    return ScenarioSummary(
        path=path.as_posix(),
        duration=config.duration,
        timestep=config.timestep,
        seed=config.seed,
        num_targets=len(config.targets),
    )


@router.get("", response_model=list[ScenarioSummary])
def list_scenarios() -> list[ScenarioSummary]:
    """List every scenario under the scenarios root.

    A file that fails to parse is skipped rather than failing the
    whole listing -- one malformed scenario shouldn't take down
    discovery of every other one.
    """
    summaries = []
    for path in sorted(_scenarios_root().rglob("*.yaml")):
        try:
            summaries.append(_summarize(path))
        except ValueError:
            continue
    return summaries


@router.get("/{scenario_path:path}", response_model=ScenarioSummary)
def get_scenario(scenario_path: str) -> ScenarioSummary:
    try:
        return _summarize(Path(scenario_path))
    except OSError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
