import numpy as np
import pytest

from radarsim.metrics.metrics import detection_probability, false_alarm_rate, position_rmse
from radarsim.types import GroundTruth, TrackEstimate, TrackStatus


def _estimate(track_id: str, x: float, y: float) -> TrackEstimate:
    return TrackEstimate(
        track_id=track_id,
        timestamp=0.0,
        state=np.array([x, y, 0.0, 0.0]),
        covariance=np.eye(4),
        status=TrackStatus.ACTIVE,
    )


def _truth(target_id: str, x: float, y: float) -> GroundTruth:
    return GroundTruth(
        timestamp=0.0,
        target_id=target_id,
        position=np.array([x, y]),
        velocity=np.zeros(2),
        acceleration=np.zeros(2),
        rcs=1.0,
    )


def test_position_rmse_uses_optimal_not_index_order_matching() -> None:
    estimates = [_estimate("track-0", 0.0, 0.0), _estimate("track-1", 10.0, 0.0)]
    truth = [_truth("target-A", 10.0, 0.0), _truth("target-B", 0.0, 0.0)]  # swapped order

    rmse = position_rmse(estimates, truth)

    assert rmse == pytest.approx(0.0, abs=1e-9)


def test_position_rmse_hand_computed() -> None:
    estimates = [_estimate("track-0", 1.0, 0.0), _estimate("track-1", 0.0, 3.0)]
    truth = [_truth("target-A", 0.0, 0.0), _truth("target-B", 0.0, 0.0)]

    rmse = position_rmse(estimates, truth)

    expected = np.sqrt((1.0**2 + 3.0**2) / 2)
    assert rmse == pytest.approx(expected)


def test_position_rmse_empty_estimates_raises() -> None:
    with pytest.raises(ValueError, match="estimate"):
        position_rmse([], [_truth("target-A", 0.0, 0.0)])


def test_position_rmse_empty_truth_raises() -> None:
    with pytest.raises(ValueError, match="ground truth"):
        position_rmse([_estimate("track-0", 0.0, 0.0)], [])


def test_detection_probability_boundary_values() -> None:
    assert detection_probability(0, 5) == 0.0
    assert detection_probability(5, 5) == 1.0
    assert detection_probability(3, 5) == pytest.approx(0.6)


def test_detection_probability_invalid_num_targets_raises() -> None:
    with pytest.raises(ValueError, match="num_targets"):
        detection_probability(0, 0)


def test_detection_probability_num_detected_out_of_range_raises() -> None:
    with pytest.raises(ValueError, match="num_detected"):
        detection_probability(6, 5)
    with pytest.raises(ValueError, match="num_detected"):
        detection_probability(-1, 5)


def test_false_alarm_rate_boundary_values() -> None:
    assert false_alarm_rate(0, 5) == 0.0
    assert false_alarm_rate(5, 5) == 1.0
    assert false_alarm_rate(2, 5) == pytest.approx(0.4)


def test_false_alarm_rate_invalid_total_raises() -> None:
    with pytest.raises(ValueError, match="num_total_detections"):
        false_alarm_rate(0, 0)


def test_false_alarm_rate_num_false_out_of_range_raises() -> None:
    with pytest.raises(ValueError, match="num_false_detections"):
        false_alarm_rate(6, 5)
    with pytest.raises(ValueError, match="num_false_detections"):
        false_alarm_rate(-1, 5)
