import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from radarsim.detection.detectors import ThresholdDetector
from radarsim.types import RadarMeasurement


def _measurement(
    snr: float, timestamp: float = 1.0, sensor_id: str = "radar-1"
) -> RadarMeasurement:
    return RadarMeasurement(
        timestamp=timestamp,
        sensor_id=sensor_id,
        range=10.0,
        radial_velocity=0.0,
        angle=0.0,
        covariance=np.zeros((3, 3)),
        snr=snr,
    )


def test_measurements_above_threshold_are_kept_below_are_dropped() -> None:
    detector = ThresholdDetector(snr_threshold_db=10.0)
    low = _measurement(snr=5.0)
    high = _measurement(snr=15.0)

    detections = detector.detect([low, high])

    assert len(detections) == 1
    assert detections[0].measurement is high


def test_confidence_is_half_at_exact_threshold() -> None:
    detector = ThresholdDetector(snr_threshold_db=10.0)
    detections = detector.detect([_measurement(snr=10.0)])
    assert detections[0].confidence == pytest.approx(0.5)


def test_confidence_increases_monotonically_with_snr() -> None:
    detector = ThresholdDetector(snr_threshold_db=10.0)
    snrs = [10.0, 12.0, 15.0, 20.0, 30.0]
    confidences = [detector.detect([_measurement(snr=s)])[0].confidence for s in snrs]
    assert confidences == sorted(confidences)
    assert len(set(confidences)) == len(confidences)


def test_detection_fields_pass_through_from_measurement() -> None:
    detector = ThresholdDetector(snr_threshold_db=0.0)
    measurement = _measurement(snr=20.0, timestamp=3.5)

    detection = detector.detect([measurement])[0]

    assert detection.timestamp == 3.5
    assert detection.measurement is measurement
    assert detection.snr == 20.0


def test_non_positive_confidence_scale_raises_value_error() -> None:
    with pytest.raises(ValueError, match="confidence_scale_db"):
        ThresholdDetector(snr_threshold_db=10.0, confidence_scale_db=0.0)


@given(
    threshold=st.floats(min_value=-50.0, max_value=50.0, allow_nan=False, allow_infinity=False),
    scale=st.floats(min_value=0.1, max_value=50.0, allow_nan=False, allow_infinity=False),
    snr=st.floats(min_value=-100.0, max_value=200.0, allow_nan=False, allow_infinity=False),
)
def test_confidence_always_in_unit_interval(threshold: float, scale: float, snr: float) -> None:
    # Bounds are closed, not open: for extreme (snr - threshold) / scale ratios the
    # logistic saturates to exactly 0.0/1.0 in float64, even though the true
    # mathematical range is the open interval (0, 1).
    detector = ThresholdDetector(snr_threshold_db=threshold, confidence_scale_db=scale)
    detections = detector.detect([_measurement(snr=snr)])
    if snr >= threshold:
        assert len(detections) == 1
        assert 0.0 <= detections[0].confidence <= 1.0
    else:
        assert detections == []
