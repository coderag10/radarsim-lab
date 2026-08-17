from __future__ import annotations

from radarsim.types import TrackEstimate


def fuse_estimates(estimates: list[TrackEstimate]) -> TrackEstimate:
    """Combine multiple sensors' estimates of the same target into one fused estimate.

    e.g. covariance-weighted average for independent Gaussian estimates.
    """
    raise NotImplementedError
