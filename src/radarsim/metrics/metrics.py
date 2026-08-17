from __future__ import annotations

from radarsim.types import GroundTruth, TrackEstimate


def position_rmse(estimates: list[TrackEstimate], truth: list[GroundTruth]) -> float:
    """Root-mean-square position error between matched estimates and ground truth."""
    raise NotImplementedError


def detection_probability(num_detected: int, num_targets: int) -> float:
    """Fraction of true targets that produced at least one detection."""
    raise NotImplementedError


def false_alarm_rate(num_false_detections: int, num_total_detections: int) -> float:
    """Fraction of detections that do not correspond to any true target."""
    raise NotImplementedError
