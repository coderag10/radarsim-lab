import numpy as np
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from radarsim.tracking.prediction import constant_velocity_transition, predict_state


def test_predict_state_hand_computed() -> None:
    state = np.array([1.0, 2.0, 3.0, 4.0])
    covariance = np.eye(4)
    transition = constant_velocity_transition(dt=1.0)
    process_noise = np.eye(4) * 0.1

    new_state, new_covariance = predict_state(state, covariance, transition, process_noise)

    np.testing.assert_allclose(new_state, [4.0, 6.0, 3.0, 4.0])
    expected_covariance = transition @ covariance @ transition.T + process_noise
    np.testing.assert_allclose(new_covariance, expected_covariance)


def test_constant_velocity_transition_structure() -> None:
    transition = constant_velocity_transition(dt=2.0)
    expected = np.array(
        [
            [1.0, 0.0, 2.0, 0.0],
            [0.0, 1.0, 0.0, 2.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    np.testing.assert_allclose(transition, expected)


_vec4 = arrays(
    dtype=np.float64,
    shape=4,
    elements=st.floats(min_value=-100.0, max_value=100.0, allow_nan=False, allow_infinity=False),
)
_mat4 = arrays(
    dtype=np.float64,
    shape=(4, 4),
    elements=st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False),
)


@given(
    state=_vec4,
    covariance_seed=_mat4,
    noise_seed=_mat4,
    dt=st.floats(min_value=0.01, max_value=10.0, allow_nan=False, allow_infinity=False),
)
def test_predicted_covariance_stays_symmetric(
    state: np.ndarray, covariance_seed: np.ndarray, noise_seed: np.ndarray, dt: float
) -> None:
    covariance = covariance_seed @ covariance_seed.T
    process_noise = noise_seed @ noise_seed.T
    transition = constant_velocity_transition(dt)

    _, new_covariance = predict_state(state, covariance, transition, process_noise)

    np.testing.assert_allclose(new_covariance, new_covariance.T, atol=1e-6)
