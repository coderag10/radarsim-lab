from pathlib import Path

import numpy as np

from radarsim.core.world import World
from radarsim.io.scenario import load_scenario


def test_two_targets_scenario_trajectory_matches_hand_computation() -> None:
    config = load_scenario(Path("scenarios/basic/two_targets.yaml"))
    world = World.from_scenario(config)

    history = world.run()

    # t=0 initial snapshot, then 5 steps of dt=1.0 -> 6 entries total
    assert len(history) == 6
    assert [step[0].timestamp for step in history] == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]

    by_id = {gt.target_id: gt for gt in history[-1]}

    # Target A: pos (10,30) + vel (2,-1) * 5s = (20, 25)
    np.testing.assert_allclose(by_id["target-A"].position, [20.0, 25.0])
    # Target B: pos (-20,15) + vel (1,2) * 5s = (-15, 25)
    np.testing.assert_allclose(by_id["target-B"].position, [-15.0, 25.0])
