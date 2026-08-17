from pathlib import Path

import numpy as np

from radarsim.core.world import World
from radarsim.io.scenario import load_scenario

_EXPECTED_FINAL_POSITIONS = {
    "target-A": (20.0, 25.0),
    "target-B": (-15.0, 25.0),
}


def test_basic_two_target_scenario_golden_trajectory() -> None:
    """Fixed-seed golden trajectory for scenarios/basic/two_targets.yaml.

    Guards against accidental behavior changes in the target/clock/world
    plumbing during future refactors -- if this ever needs to change,
    it should be a deliberate, reviewed change to the fixture or model.
    """
    config = load_scenario(Path("scenarios/basic/two_targets.yaml"))
    world = World.from_scenario(config)

    history = world.run()
    final_positions = {gt.target_id: tuple(gt.position) for gt in history[-1]}

    assert set(final_positions) == set(_EXPECTED_FINAL_POSITIONS)
    for target_id, expected in _EXPECTED_FINAL_POSITIONS.items():
        np.testing.assert_allclose(final_positions[target_id], expected)
