import numpy as np

from radarsim.fusion.fuse import match_tracks
from radarsim.types import TrackEstimate, TrackStatus


def _track(track_id: str, x: float, y: float) -> TrackEstimate:
    return TrackEstimate(
        track_id=track_id,
        timestamp=0.0,
        state=np.array([x, y, 0.0, 0.0]),
        covariance=np.eye(4),
        status=TrackStatus.ACTIVE,
    )


def test_match_tracks_pairs_nearest_within_gate() -> None:
    tracks_a = [_track("a0", 0.0, 0.0), _track("a1", 10.0, 0.0)]
    tracks_b = [_track("b0", 10.2, 0.0), _track("b1", 0.1, 0.0)]

    pairs = match_tracks(tracks_a, tracks_b, gate_threshold=1.0)

    pair_ids = {(a.track_id, b.track_id) for a, b in pairs}
    assert pair_ids == {("a0", "b1"), ("a1", "b0")}


def test_match_tracks_leaves_unmatched_target_unpaired() -> None:
    tracks_a = [_track("a0", 0.0, 0.0)]
    tracks_b = [_track("b0", 0.1, 0.0), _track("b1", 100.0, 100.0)]

    pairs = match_tracks(tracks_a, tracks_b, gate_threshold=1.0)

    assert len(pairs) == 1
    assert pairs[0][0].track_id == "a0"
    assert pairs[0][1].track_id == "b0"


def test_match_tracks_empty_input_returns_no_matches() -> None:
    assert match_tracks([], [_track("b0", 0.0, 0.0)], gate_threshold=1.0) == []
    assert match_tracks([_track("a0", 0.0, 0.0)], [], gate_threshold=1.0) == []


def test_match_tracks_outside_gate_returns_no_matches() -> None:
    tracks_a = [_track("a0", 0.0, 0.0)]
    tracks_b = [_track("b0", 100.0, 0.0)]

    assert match_tracks(tracks_a, tracks_b, gate_threshold=1.0) == []
