from __future__ import annotations

from typing import Protocol

import numpy as np

from radarsim.tracking.measurement_conversion import polar_to_cartesian_measurement
from radarsim.types import Detection, TrackEstimate


class AssociationStrategy(Protocol):
    """Common interface for assigning detections to existing tracks."""

    def associate(
        self, tracks: list[TrackEstimate], detections: list[Detection]
    ) -> dict[str, Detection]:
        """Return a mapping of track_id -> assigned Detection for this timestep."""
        ...


class NearestNeighbor:
    """Assign each track its closest unclaimed detection, subject to a gating threshold.

    Greedy, not an optimal assignment (no Hungarian algorithm): tracks
    are matched in list order, each claiming its nearest still-unclaimed
    detection. Distance is plain Euclidean, on positions converted from
    each detection's polar measurement via `polar_to_cartesian_measurement`
    -- `sensor_position` is required for that conversion to land in
    world-frame coordinates, so this instance is tied to one sensor.
    """

    def __init__(self, gate_threshold: float, sensor_position: np.ndarray) -> None:
        self.gate_threshold = gate_threshold
        self.sensor_position = sensor_position

    def associate(
        self, tracks: list[TrackEstimate], detections: list[Detection]
    ) -> dict[str, Detection]:
        detection_positions = [
            polar_to_cartesian_measurement(detection.measurement, self.sensor_position)[0]
            for detection in detections
        ]
        claimed: set[int] = set()
        assignment: dict[str, Detection] = {}

        for track in tracks:
            track_position = track.state[:2]
            best_index: int | None = None
            best_distance = self.gate_threshold

            for index, position in enumerate(detection_positions):
                if index in claimed:
                    continue
                distance = float(np.linalg.norm(position - track_position))
                if distance <= best_distance:
                    best_distance = distance
                    best_index = index

            if best_index is not None:
                assignment[track.track_id] = detections[best_index]
                claimed.add(best_index)

        return assignment
