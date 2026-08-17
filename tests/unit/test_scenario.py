from pathlib import Path

import pytest

from radarsim.io.scenario import load_scenario


def test_load_scenario_parses_two_targets_fixture() -> None:
    config = load_scenario(Path("scenarios/basic/two_targets.yaml"))

    assert config.duration == 5.0
    assert config.timestep == 1.0
    assert config.seed == 42
    assert len(config.targets) == 2

    target_a = config.targets[0]
    assert target_a.id == "target-A"
    assert target_a.model == "constant_velocity"
    assert target_a.position == (10.0, 30.0)
    assert target_a.velocity == (2.0, -1.0)
    assert target_a.acceleration == (0.0, 0.0)
    assert target_a.rcs == 5.0


def test_missing_simulation_block_raises(tmp_path: Path) -> None:
    scenario = tmp_path / "bad.yaml"
    scenario.write_text("targets: []\n")
    with pytest.raises(ValueError, match="simulation"):
        load_scenario(scenario)


def test_empty_targets_list_raises(tmp_path: Path) -> None:
    scenario = tmp_path / "bad.yaml"
    scenario.write_text("simulation:\n  duration: 1.0\n  timestep: 1.0\n  seed: 1\ntargets: []\n")
    with pytest.raises(ValueError, match="targets"):
        load_scenario(scenario)


def test_unknown_motion_model_raises(tmp_path: Path) -> None:
    scenario = tmp_path / "bad.yaml"
    scenario.write_text(
        "simulation:\n  duration: 1.0\n  timestep: 1.0\n  seed: 1\n"
        "targets:\n  - id: t1\n    model: warp_speed\n    position: [0.0]\n    velocity: [0.0]\n"
    )
    with pytest.raises(ValueError, match="model"):
        load_scenario(scenario)


def test_mismatched_dimensionality_raises(tmp_path: Path) -> None:
    scenario = tmp_path / "bad.yaml"
    scenario.write_text(
        "simulation:\n  duration: 1.0\n  timestep: 1.0\n  seed: 1\n"
        "targets:\n  - id: t1\n    model: constant_velocity\n"
        "    position: [0.0, 0.0]\n    velocity: [0.0]\n"
    )
    with pytest.raises(ValueError, match="dimensionality"):
        load_scenario(scenario)
