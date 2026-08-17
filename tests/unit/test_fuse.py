import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from radarsim.fusion.fuse import fuse_estimates
from radarsim.types import TrackEstimate, TrackStatus


def _estimate(
    track_id: str,
    state: list[float],
    covariance: list[list[float]],
    timestamp: float = 0.0,
    status: TrackStatus = TrackStatus.ACTIVE,
) -> TrackEstimate:
    return TrackEstimate(
        track_id=track_id,
        timestamp=timestamp,
        state=np.array(state, dtype=float),
        covariance=np.array(covariance, dtype=float),
        status=status,
    )


def test_fuse_two_estimates_hand_computed() -> None:
    a = _estimate("a", [10.0], [[4.0]])
    b = _estimate("b", [12.0], [[1.0]])

    fused = fuse_estimates([a, b])

    np.testing.assert_allclose(fused.state, [11.6])
    np.testing.assert_allclose(fused.covariance, [[0.8]])
    assert fused.track_id == "fused(a,b)"
    assert fused.timestamp == 0.0


def test_fuse_single_estimate_returned_unchanged() -> None:
    a = _estimate("a", [1.0, 2.0], [[1.0, 0.0], [0.0, 1.0]])
    assert fuse_estimates([a]) is a


def test_fuse_empty_list_raises_value_error() -> None:
    with pytest.raises(ValueError, match="at least one"):
        fuse_estimates([])


def test_fuse_mismatched_timestamps_raises_value_error() -> None:
    a = _estimate("a", [0.0], [[1.0]], timestamp=1.0)
    b = _estimate("b", [0.0], [[1.0]], timestamp=2.0)
    with pytest.raises(ValueError, match="timestamp"):
        fuse_estimates([a, b])


@pytest.mark.parametrize(
    ("statuses", "expected"),
    [
        ((TrackStatus.ACTIVE, TrackStatus.TENTATIVE), TrackStatus.ACTIVE),
        ((TrackStatus.TENTATIVE, TrackStatus.LOST), TrackStatus.TENTATIVE),
        ((TrackStatus.LOST, TrackStatus.LOST), TrackStatus.LOST),
    ],
)
def test_fuse_status_takes_most_confirmed(
    statuses: tuple[TrackStatus, TrackStatus], expected: TrackStatus
) -> None:
    estimates = [
        _estimate(f"t{i}", [0.0], [[1.0]], status=status) for i, status in enumerate(statuses)
    ]
    assert fuse_estimates(estimates).status == expected


_mat2 = arrays(
    dtype=np.float64,
    shape=(2, 2),
    elements=st.floats(min_value=-3.0, max_value=3.0, allow_nan=False, allow_infinity=False),
)
_vec2 = arrays(
    dtype=np.float64,
    shape=2,
    elements=st.floats(min_value=-50.0, max_value=50.0, allow_nan=False, allow_infinity=False),
)


@given(state_a=_vec2, state_b=_vec2, seed_a=_mat2, seed_b=_mat2)
def test_fused_covariance_trace_never_exceeds_either_input(
    state_a: np.ndarray, state_b: np.ndarray, seed_a: np.ndarray, seed_b: np.ndarray
) -> None:
    covariance_a = seed_a @ seed_a.T + np.eye(2) * 0.5
    covariance_b = seed_b @ seed_b.T + np.eye(2) * 0.5
    a = _estimate("a", list(state_a), covariance_a.tolist())
    b = _estimate("b", list(state_b), covariance_b.tolist())

    fused = fuse_estimates([a, b])

    assert np.trace(fused.covariance) <= min(np.trace(covariance_a), np.trace(covariance_b)) + 1e-6
