from __future__ import annotations

from typing import Protocol

from radarsim.types import Detection, TrackEstimate


class AssociationStrategy(Protocol):
    """Common interface for assigning detections to existing tracks."""

    def associate(
        self, tracks: list[TrackEstimate], detections: list[Detection]
    ) -> dict[str, Detection]:
        """Return a mapping of track_id -> assigned Detection for this timestep."""
        ...


class NearestNeighbor:
    """Assign each track its closest unclaimed detection, subject to a gating threshold."""

    def __init__(self, gate_threshold: float) -> None:
        raise NotImplementedError

    def associate(
        self, tracks: list[TrackEstimate], detections: list[Detection]
    ) -> dict[str, Detection]:
        raise NotImplementedError
