"""Pydantic request/response models for the HTTP API.

This is the one place in this codebase pydantic belongs -- validating
untrusted HTTP input/output at the API boundary, not the simulation
inner loop (see docs/ARCHITECTURE.md "Data contracts").
"""

from radarsim.api.schemas.run import (
    MetricsOut,
    RunRequest,
    RunResponse,
    SensorInfo,
    TrackOut,
)
from radarsim.api.schemas.scenario import ScenarioSummary

__all__ = [
    "MetricsOut",
    "RunRequest",
    "RunResponse",
    "ScenarioSummary",
    "SensorInfo",
    "TrackOut",
]
