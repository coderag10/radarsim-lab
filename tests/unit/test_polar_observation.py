import numpy as np

from radarsim.tracking.polar_observation import polar_observation, polar_observation_jacobian


def test_observation_hand_computed_at_origin() -> None:
    state = np.array([3.0, 4.0, 0.0, 5.0])
    sensor_position = np.zeros(2)

    observed = polar_observation(state, sensor_position)

    np.testing.assert_allclose(observed, [5.0, 4.0, np.arctan2(4.0, 3.0)])


def test_observation_hand_computed_offset_sensor() -> None:
    # same relative geometry as the origin case, shifted by (1, 1)
    state = np.array([4.0, 5.0, 0.0, 5.0])
    sensor_position = np.array([1.0, 1.0])

    observed = polar_observation(state, sensor_position)

    np.testing.assert_allclose(observed, [5.0, 4.0, np.arctan2(4.0, 3.0)])


def test_jacobian_hand_computed_at_origin() -> None:
    state = np.array([3.0, 4.0, 0.0, 5.0])
    sensor_position = np.zeros(2)

    jacobian = polar_observation_jacobian(state, sensor_position)

    expected = np.array(
        [
            [0.6, 0.8, 0.0, 0.0],
            [-0.48, 0.36, 0.6, 0.8],
            [-0.16, 0.12, 0.0, 0.0],
        ]
    )
    np.testing.assert_allclose(jacobian, expected, atol=1e-9)
