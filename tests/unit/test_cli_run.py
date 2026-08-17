import json
from pathlib import Path
from typing import Any

import numpy as np

from radarsim.cli.run import format_json, format_table, run_scenario
from radarsim.types import TrackStatus

_SCENARIO = Path("scenarios/basic/two_targets.yaml")


def _run(**overrides: Any) -> Any:
    defaults: dict[str, Any] = {
        "radar_position": np.zeros(2),
        "noise_std": np.array([0.3, 0.1, 0.01]),
        "reference_rcs": 5.0,
        "reference_range": 30.0,
        "reference_snr_db": 20.0,
        "snr_threshold_db": 0.0,
    }
    defaults.update(overrides)
    return run_scenario(_SCENARIO, **defaults)


def test_run_scenario_default_produces_two_active_tracks() -> None:
    result = _run()

    active = [track for track in result.tracks if track.status == TrackStatus.ACTIVE]
    assert len(active) == 2
    assert result.detection_probability == 1.0
    assert result.position_rmse is not None
    assert result.position_rmse < 2.0
    assert result.num_targets == 2
    assert result.seed == 42
    assert len(result.ground_truth) == 2
    assert {truth.target_id for truth in result.ground_truth} == {"target-A", "target-B"}


def test_run_scenario_high_threshold_produces_no_tracks_and_none_rmse() -> None:
    result = _run(snr_threshold_db=1000.0)

    assert result.tracks == []
    assert result.detection_probability == 0.0
    assert result.position_rmse is None


def test_format_table_contains_track_ids_and_statuses() -> None:
    result = _run()
    table = format_table(result)

    for track in result.tracks:
        assert track.track_id in table
        assert track.status.name in table
    for truth in result.ground_truth:
        assert truth.target_id in table
    assert "Detection probability" in table


def test_format_table_handles_no_tracks() -> None:
    result = _run(snr_threshold_db=1000.0)
    table = format_table(result)
    assert "N/A" in table


def test_format_json_round_trips() -> None:
    result = _run()
    payload = json.loads(format_json(result))

    assert payload["seed"] == 42
    assert payload["metrics"]["detection_probability"] == 1.0
    assert len(payload["tracks"]) == len(result.tracks)
    assert payload["tracks"][0]["track_id"] == result.tracks[0].track_id
    assert len(payload["ground_truth"]) == len(result.ground_truth)
    assert payload["ground_truth"][0]["target_id"] == result.ground_truth[0].target_id
