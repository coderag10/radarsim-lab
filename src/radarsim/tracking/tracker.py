from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from radarsim.tracking.association import AssociationStrategy
from radarsim.tracking.filters.kalman import KalmanFilter
from radarsim.tracking.measurement_conversion import polar_to_cartesian_measurement
from radarsim.types import Detection, TrackEstimate, TrackStatus


@dataclass
class _TrackState:
    """Internal per-track filter state. `TrackEstimate` is the public,
    immutable snapshot handed out by `Tracker.step`; this is the mutable
    working state the tracker updates in place between snapshots.
    """

    track_id: str
    state: np.ndarray
    covariance: np.ndarray
    status: TrackStatus
    hits: int = 0
    misses: int = 0


class Tracker:
    """Owns the set of live tracks across timesteps.

    Each `step` predicts every track forward, associates incoming
    detections against those predictions, updates matched tracks,
    ages unmatched tracks toward `LOST`, and spawns new `TENTATIVE`
    tracks from leftover detections.

    Assumes one fixed timestep baked into `kalman_filter`'s transition
    matrix (matching `SimulationClock`'s fixed-timestep design) -- it
    takes no `dt` itself, staying decoupled from simulation timing.
    Processes one sensor's detections at a time; combining multiple
    sensors is `radarsim.fusion`'s job, not this class's.
    """

    def __init__(
        self,
        kalman_filter: KalmanFilter,
        association: AssociationStrategy,
        confirm_hits: int = 3,
        max_misses: int = 3,
        initial_velocity_variance: float = 100.0,
    ) -> None:
        self.kalman_filter = kalman_filter
        self.association = association
        self.confirm_hits = confirm_hits
        self.max_misses = max_misses
        self.initial_velocity_variance = initial_velocity_variance
        self._tracks: dict[str, _TrackState] = {}
        self._next_id = 0

    def step(self, timestamp: float, detections: list[Detection]) -> list[TrackEstimate]:
        self._tracks = {
            track_id: track
            for track_id, track in self._tracks.items()
            if track.status != TrackStatus.LOST
        }

        for track in self._tracks.values():
            track.state, track.covariance = self.kalman_filter.predict(
                track.state, track.covariance
            )

        predicted_estimates = [
            self._to_estimate(track, timestamp) for track in self._tracks.values()
        ]
        assignment = self.association.associate(predicted_estimates, detections)

        for track_id, track in self._tracks.items():
            detection = assignment.get(track_id)
            if detection is None:
                track.misses += 1
                if track.misses >= self.max_misses:
                    track.status = TrackStatus.LOST
                continue

            position, position_covariance = polar_to_cartesian_measurement(detection.measurement)
            track.state, track.covariance = self.kalman_filter.update(
                track.state, track.covariance, position, measurement_noise=position_covariance
            )
            track.hits += 1
            track.misses = 0
            if track.status == TrackStatus.TENTATIVE and track.hits >= self.confirm_hits:
                track.status = TrackStatus.ACTIVE

        assigned_detections = set(assignment.values())
        for detection in detections:
            if detection in assigned_detections:
                continue
            self._spawn_track(detection)

        return [self._to_estimate(track, timestamp) for track in self._tracks.values()]

    def _spawn_track(self, detection: Detection) -> None:
        position, position_covariance = polar_to_cartesian_measurement(detection.measurement)
        covariance = np.zeros((4, 4))
        covariance[:2, :2] = position_covariance
        covariance[2, 2] = self.initial_velocity_variance
        covariance[3, 3] = self.initial_velocity_variance

        track_id = f"track-{self._next_id}"
        self._next_id += 1
        initial_status = TrackStatus.ACTIVE if 1 >= self.confirm_hits else TrackStatus.TENTATIVE
        self._tracks[track_id] = _TrackState(
            track_id=track_id,
            state=np.array([position[0], position[1], 0.0, 0.0]),
            covariance=covariance,
            status=initial_status,
            hits=1,
            misses=0,
        )

    @staticmethod
    def _to_estimate(track: _TrackState, timestamp: float) -> TrackEstimate:
        return TrackEstimate(
            track_id=track.track_id,
            timestamp=timestamp,
            state=track.state.copy(),
            covariance=track.covariance.copy(),
            status=track.status,
        )
