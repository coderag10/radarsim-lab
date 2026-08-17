import numpy as np

from radarsim.tracking.association import NearestNeighbor
from radarsim.tracking.filters.kalman import KalmanFilter
from radarsim.tracking.prediction import constant_velocity_transition
from radarsim.tracking.tracker import Tracker
from radarsim.types import Detection, RadarMeasurement, TrackStatus


def _detection(range_: float, angle: float = 0.0, timestamp: float = 0.0) -> Detection:
    measurement = RadarMeasurement(
        timestamp=timestamp,
        sensor_id="radar-1",
        range=range_,
        radial_velocity=0.0,
        angle=angle,
        covariance=np.diag([0.1, 0.1, 0.001]),
        snr=20.0,
    )
    return Detection(timestamp=timestamp, measurement=measurement, confidence=0.9, snr=20.0)


def _make_tracker(confirm_hits: int = 3, max_misses: int = 3) -> Tracker:
    kalman_filter = KalmanFilter(
        transition=constant_velocity_transition(dt=1.0),
        observation=np.eye(2, 4),
        process_noise=np.eye(4) * 0.01,
        measurement_noise=np.eye(2),  # overridden per-call by Tracker
    )
    return Tracker(
        kalman_filter=kalman_filter,
        association=NearestNeighbor(gate_threshold=5.0),
        confirm_hits=confirm_hits,
        max_misses=max_misses,
    )


def test_spawns_tentative_track_on_first_detection() -> None:
    tracker = _make_tracker(confirm_hits=3)

    tracks = tracker.step(0.0, [_detection(range_=10.0)])

    assert len(tracks) == 1
    assert tracks[0].status == TrackStatus.TENTATIVE
    np.testing.assert_allclose(tracks[0].state[:2], [10.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(tracks[0].state[2:], [0.0, 0.0])


def test_promotes_to_active_after_confirm_hits() -> None:
    tracker = _make_tracker(confirm_hits=3)

    tracks = tracker.step(0.0, [_detection(range_=10.0, timestamp=0.0)])
    assert tracks[0].status == TrackStatus.TENTATIVE

    tracks = tracker.step(1.0, [_detection(range_=10.0, timestamp=1.0)])
    assert tracks[0].status == TrackStatus.TENTATIVE

    tracks = tracker.step(2.0, [_detection(range_=10.0, timestamp=2.0)])
    assert tracks[0].status == TrackStatus.ACTIVE


def test_single_hit_confirmation_promotes_immediately_on_spawn() -> None:
    tracker = _make_tracker(confirm_hits=1)

    tracks = tracker.step(0.0, [_detection(range_=10.0)])

    assert tracks[0].status == TrackStatus.ACTIVE


def test_degrades_to_lost_after_max_misses_and_is_pruned() -> None:
    tracker = _make_tracker(confirm_hits=1, max_misses=2)

    tracks = tracker.step(0.0, [_detection(range_=10.0, timestamp=0.0)])
    assert tracks[0].status == TrackStatus.ACTIVE
    track_id = tracks[0].track_id

    tracks = tracker.step(1.0, [])
    assert tracks[0].status == TrackStatus.ACTIVE

    tracks = tracker.step(2.0, [])
    assert len(tracks) == 1
    assert tracks[0].status == TrackStatus.LOST
    assert tracks[0].track_id == track_id

    tracks = tracker.step(3.0, [])
    assert tracks == []


def test_hit_resets_miss_counter() -> None:
    tracker = _make_tracker(confirm_hits=1, max_misses=2)

    tracker.step(0.0, [_detection(range_=10.0, timestamp=0.0)])
    tracker.step(1.0, [])  # miss 1
    tracks = tracker.step(2.0, [_detection(range_=10.0, timestamp=2.0)])  # hit resets misses
    assert tracks[0].status == TrackStatus.ACTIVE

    tracks = tracker.step(3.0, [])  # miss 1 again, not miss 2
    assert tracks[0].status == TrackStatus.ACTIVE
