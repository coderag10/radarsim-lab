import json
from pathlib import Path

import pytest

from radarsim.cli.main import build_parser, main


def test_build_parser_defaults() -> None:
    parser = build_parser()
    args = parser.parse_args(["scenarios/basic/two_targets.yaml"])

    assert args.scenario == Path("scenarios/basic/two_targets.yaml")
    assert args.radar_x == 0.0
    assert args.radar_y == 0.0
    assert list(args.noise_std) == [0.3, 0.1, 0.01]
    assert args.reference_rcs == 5.0
    assert args.reference_range == 30.0
    assert args.reference_snr_db == 20.0
    assert args.snr_threshold_db == 0.0
    assert args.format == "table"


def test_main_table_format_exits_zero(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(["scenarios/basic/two_targets.yaml"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Scenario:" in captured.out
    assert captured.err == ""


def test_main_json_format_is_valid_json(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(["scenarios/basic/two_targets.yaml", "--format", "json"])
    captured = capsys.readouterr()

    assert exit_code == 0
    payload = json.loads(captured.out)
    assert payload["seed"] == 42


def test_main_missing_scenario_file_exits_one_cleanly(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(["scenarios/does_not_exist.yaml"])
    captured = capsys.readouterr()

    assert exit_code == 1
    assert captured.out == ""
    assert "radarsim: error:" in captured.err
