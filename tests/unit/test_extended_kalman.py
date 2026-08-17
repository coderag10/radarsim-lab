import numpy as np
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from radarsim.tracking.filters.extended_kalman import ExtendedKalmanFilter
from radarsim.tracking.filters.kalman import KalmanFilter
from radarsim.tracking.prediction import constant_velocity_transition


def test_predict_matches_manual_linearization() -> None:
    transition = constant_velocity_transition(dt=1.0)
    process_noise = np.eye(4) * 0.2
    ekf = ExtendedKalmanFilter(
        transition_fn=lambda state: transition @ state,
        transition_jacobian=lambda state: transition,
        observation_fn=lambda state: state[:2],
        observation_jacobian=lambda state: np.eye(2, 4),
        process_noise=process_noise,
        measurement_noise=np.eye(2),
    )
    state = np.array([1.0, 2.0, 3.0, 4.0])
    covariance = np.eye(4) * 2.0

    new_state, new_covariance = ekf.predict(state, covariance)

    np.testing.assert_allclose(new_state, transition @ state)
    np.testing.assert_allclose(
        new_covariance, transition @ covariance @ transition.T + process_noise
    )


def test_update_with_linear_observation_matches_kalman_filter() -> None:
    observation = np.eye(2, 4)
    measurement_noise = np.diag([0.5, 0.5])
    ekf = ExtendedKalmanFilter(
        transition_fn=lambda state: state,
        transition_jacobian=lambda state: np.eye(4),
        observation_fn=lambda state: observation @ state,
        observation_jacobian=lambda state: observation,
        process_noise=np.zeros((4, 4)),
        measurement_noise=measurement_noise,
    )
    kf = KalmanFilter(
        transition=np.eye(4),
        observation=observation,
        process_noise=np.zeros((4, 4)),
        measurement_noise=measurement_noise,
    )
    state = np.array([0.0, 0.0, 1.0, 1.0])
    covariance = np.eye(4)
    measurement = np.array([1.0, 1.0])

    ekf_state, ekf_covariance = ekf.update(state, covariance, measurement)
    kf_state, kf_covariance = kf.update(state, covariance, measurement)

    np.testing.assert_allclose(ekf_state, kf_state)
    np.testing.assert_allclose(ekf_covariance, kf_covariance)


def test_measurement_noise_override_pulls_estimate_closer_to_measurement() -> None:
    observation = np.eye(2, 4)
    ekf = ExtendedKalmanFilter(
        transition_fn=lambda state: state,
        transition_jacobian=lambda state: np.eye(4),
        observation_fn=lambda state: observation @ state,
        observation_jacobian=lambda state: observation,
        process_noise=np.zeros((4, 4)),
        measurement_noise=np.diag([1.0, 1.0]),
    )
    state = np.array([0.0, 0.0, 0.0, 0.0])
    covariance = np.eye(4)
    measurement = np.array([1.0, 1.0])

    default_state, _ = ekf.update(state, covariance, measurement)
    tight_state, _ = ekf.update(
        state, covariance, measurement, measurement_noise=np.diag([0.01, 0.01])
    )

    default_error = np.linalg.norm(default_state[:2] - measurement)
    tight_error = np.linalg.norm(tight_state[:2] - measurement)
    assert tight_error < default_error


_vec4 = arrays(
    dtype=np.float64,
    shape=4,
    elements=st.floats(min_value=-50.0, max_value=50.0, allow_nan=False, allow_infinity=False),
)
_vec2 = arrays(
    dtype=np.float64,
    shape=2,
    elements=st.floats(min_value=-50.0, max_value=50.0, allow_nan=False, allow_infinity=False),
)
_mat4 = arrays(
    dtype=np.float64,
    shape=(4, 4),
    elements=st.floats(min_value=-5.0, max_value=5.0, allow_nan=False, allow_infinity=False),
)
_mat2 = arrays(
    dtype=np.float64,
    shape=(2, 2),
    elements=st.floats(min_value=-5.0, max_value=5.0, allow_nan=False, allow_infinity=False),
)


@given(state=_vec4, covariance_seed=_mat4, noise_seed=_mat2, measurement=_vec2)
def test_update_does_not_increase_uncertainty(
    state: np.ndarray, covariance_seed: np.ndarray, noise_seed: np.ndarray, measurement: np.ndarray
) -> None:
    covariance = covariance_seed @ covariance_seed.T + np.eye(4) * 0.1
    measurement_noise = noise_seed @ noise_seed.T + np.eye(2) * 0.1
    observation = np.eye(2, 4)
    ekf = ExtendedKalmanFilter(
        transition_fn=lambda s: s,
        transition_jacobian=lambda s: np.eye(4),
        observation_fn=lambda s: observation @ s,
        observation_jacobian=lambda s: observation,
        process_noise=np.zeros((4, 4)),
        measurement_noise=measurement_noise,
    )

    _, new_covariance = ekf.update(state, covariance, measurement)

    assert np.trace(new_covariance) <= np.trace(covariance) + 1e-6
