import numpy as np

from radarsim.tracking.association import NearestNeighbor
from radarsim.types import Detection, RadarMeasurement, TrackEstimate, TrackStatus


def _detection(range_: float, angle: float = 0.0) -> Detection:
    measurement = RadarMeasurement(
        timestamp=0.0,
        sensor_id="radar-1",
        range=range_,
        radial_velocity=0.0,
        angle=angle,
        covariance=np.diag([0.1, 0.1, 0.001]),
        snr=20.0,
    )
    return Detection(timestamp=0.0, measurement=measurement, confidence=0.9, snr=20.0)


def _track(track_id: str, x: float, y: float) -> TrackEstimate:
    return TrackEstimate(
        track_id=track_id,
        timestamp=0.0,
        state=np.array([x, y, 0.0, 0.0]),
        covariance=np.eye(4),
        status=TrackStatus.ACTIVE,
    )


def test_nearest_unclaimed_detection_within_gate_is_chosen() -> None:
    track = _track("t1", x=10.0, y=0.0)
    near = _detection(range_=10.1)
    far = _detection(range_=15.0)

    association = NearestNeighbor(gate_threshold=1.0, sensor_position=np.zeros(2))
    result = association.associate([track], [near, far])

    assert result["t1"] is near


def test_nothing_chosen_outside_gate() -> None:
    track = _track("t1", x=10.0, y=0.0)
    far = _detection(range_=50.0)

    association = NearestNeighbor(gate_threshold=1.0, sensor_position=np.zeros(2))
    result = association.associate([track], [far])

    assert result == {}


def test_greedy_claiming_first_track_wins_shared_detection() -> None:
    track_a = _track("a", x=10.0, y=0.0)
    track_b = _track("b", x=10.5, y=0.0)
    detection = _detection(range_=10.0)

    association = NearestNeighbor(gate_threshold=5.0, sensor_position=np.zeros(2))
    result = association.associate([track_a, track_b], [detection])

    assert result == {"a": detection}
