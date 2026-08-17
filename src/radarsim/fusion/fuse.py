from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

from radarsim.types import TrackEstimate, TrackStatus

_STATUS_RANK = {TrackStatus.LOST: 0, TrackStatus.TENTATIVE: 1, TrackStatus.ACTIVE: 2}


def fuse_estimates(estimates: list[TrackEstimate]) -> TrackEstimate:
    """Combine multiple sensors' estimates of the same target into one fused estimate.

    Inverse-covariance-weighted average -- the maximum-likelihood
    combination of independent Gaussian estimates. `fused_covariance`
    is never less certain than any single input's (this is a general
    property of information fusion, not specific to any numbers here).
    Assumes every estimate already refers to the same target (see
    `match_tracks` for pairing tracks across sensors first) and that
    all `timestamp`s agree -- fusing estimates from different moments
    would silently average over time, which is a real caller mistake,
    not something to paper over.
    """
    if not estimates:
        raise ValueError("fuse_estimates requires at least one estimate")
    if len(estimates) == 1:
        return estimates[0]

    timestamp = estimates[0].timestamp
    if any(estimate.timestamp != timestamp for estimate in estimates):
        raise ValueError("cannot fuse estimates with different timestamps")

    information_matrices = [np.linalg.inv(estimate.covariance) for estimate in estimates]
    fused_information = sum(information_matrices)
    fused_covariance = np.linalg.inv(fused_information)
    weighted_states = sum(
        information @ estimate.state
        for information, estimate in zip(information_matrices, estimates, strict=True)
    )
    fused_state = fused_covariance @ weighted_states

    return TrackEstimate(
        track_id=f"fused({','.join(estimate.track_id for estimate in estimates)})",
        timestamp=timestamp,
        state=fused_state,
        covariance=fused_covariance,
        status=max((estimate.status for estimate in estimates), key=lambda s: _STATUS_RANK[s]),
    )


def match_tracks(
    tracks_a: list[TrackEstimate], tracks_b: list[TrackEstimate], gate_threshold: float
) -> list[tuple[TrackEstimate, TrackEstimate]]:
    """Pair tracks from two independent sensors that likely observe the same target.

    Optimal (minimum total distance) bipartite matching via
    `scipy.optimize.linear_sum_assignment` on Cartesian position
    distance (`TrackEstimate.state[:2]`, matching the tracking
    module's convention), restricted to pairs within `gate_threshold`
    -- a track only one sensor currently sees is left unmatched, not
    force-paired with something implausible.
    """
    if not tracks_a or not tracks_b:
        return []

    cost_matrix = np.array(
        [
            [float(np.linalg.norm(a.state[:2] - b.state[:2])) for b in tracks_b]
            for a in tracks_a
        ]
    )
    row_indices, col_indices = linear_sum_assignment(cost_matrix)

    return [
        (tracks_a[row], tracks_b[col])
        for row, col in zip(row_indices, col_indices, strict=True)
        if cost_matrix[row, col] <= gate_threshold
    ]
