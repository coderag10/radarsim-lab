from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

from radarsim.types import GroundTruth, TrackEstimate


def position_rmse(estimates: list[TrackEstimate], truth: list[GroundTruth]) -> float:
    """Root-mean-square position error between matched estimates and ground truth.

    Track IDs have no natural correspondence to target IDs, so
    estimates and truth are matched via optimal (minimum total
    distance) bipartite assignment, not naive index alignment. Assumes
    `TrackEstimate.state[:2]` is Cartesian position, matching the
    convention `radarsim.tracking.Tracker` establishes.

    Only scores matched pairs -- unlike an OSPA-style metric, this
    doesn't penalize a cardinality mismatch (missed targets, spurious
    tracks); see `detection_probability`/`false_alarm_rate` for those.
    """
    if not estimates or not truth:
        raise ValueError("position_rmse requires at least one estimate and one ground truth target")

    cost_matrix = np.array(
        [
            [float(np.linalg.norm(estimate.state[:2] - target.position)) for target in truth]
            for estimate in estimates
        ]
    )
    row_indices, col_indices = linear_sum_assignment(cost_matrix)
    squared_errors = cost_matrix[row_indices, col_indices] ** 2
    return float(np.sqrt(np.mean(squared_errors)))


def detection_probability(num_detected: int, num_targets: int) -> float:
    """Fraction of true targets that produced at least one detection."""
    if num_targets <= 0:
        raise ValueError(f"num_targets must be positive, got {num_targets}")
    if not 0 <= num_detected <= num_targets:
        raise ValueError(
            f"num_detected ({num_detected}) must be between 0 and num_targets ({num_targets})"
        )
    return num_detected / num_targets


def false_alarm_rate(num_false_detections: int, num_total_detections: int) -> float:
    """Fraction of detections that do not correspond to any true target."""
    if num_total_detections <= 0:
        raise ValueError(f"num_total_detections must be positive, got {num_total_detections}")
    if not 0 <= num_false_detections <= num_total_detections:
        raise ValueError(
            f"num_false_detections ({num_false_detections}) must be between 0 and "
            f"num_total_detections ({num_total_detections})"
        )
    return num_false_detections / num_total_detections
