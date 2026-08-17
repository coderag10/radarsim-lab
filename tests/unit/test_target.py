import numpy as np

from radarsim.targets.motion_models import ConstantVelocityModel
from radarsim.targets.rcs import ConstantRCS
from radarsim.targets.target import Target


def _make_target() -> Target:
    return Target(
        target_id="t1",
        position=np.array([0.0, 0.0]),
        velocity=np.array([1.0, 2.0]),
        acceleration=np.array([0.0, 0.0]),
        motion_model=ConstantVelocityModel(),
        rcs_model=ConstantRCS(4.0),
    )


def test_snapshot_does_not_mutate_state() -> None:
    target = _make_target()
    snapshot = target.snapshot(timestamp=0.0)

    assert snapshot.timestamp == 0.0
    assert snapshot.target_id == "t1"
    np.testing.assert_allclose(snapshot.position, [0.0, 0.0])
    assert snapshot.rcs == 4.0

    snapshot.position[:] = 999.0
    np.testing.assert_allclose(target.position, [0.0, 0.0])


def test_step_mutates_state_and_returns_matching_ground_truth() -> None:
    target = _make_target()
    ground_truth = target.step(dt=1.0, timestamp=1.0)

    np.testing.assert_allclose(target.position, [1.0, 2.0])
    np.testing.assert_allclose(ground_truth.position, [1.0, 2.0])
    assert ground_truth.timestamp == 1.0
