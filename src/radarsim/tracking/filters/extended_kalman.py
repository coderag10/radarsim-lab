from __future__ import annotations

from collections.abc import Callable

import numpy as np


class ExtendedKalmanFilter:
    """EKF for nonlinear motion/measurement models, linearized via Jacobians at each step.

    Known limitation: `update`'s innovation is plain subtraction, so an
    angle-valued observation wrapping through +-pi would be computed
    incorrectly. Not handled here -- this class is generic (it doesn't
    know which of its caller-supplied observation dimensions, if any,
    is an angle); callers with wraparound-prone observations need to
    handle it themselves (e.g. a custom `observation_fn` that keeps the
    angle in a safe range for the scenario at hand).
    """

    def __init__(
        self,
        transition_fn: Callable[[np.ndarray], np.ndarray],
        transition_jacobian: Callable[[np.ndarray], np.ndarray],
        observation_fn: Callable[[np.ndarray], np.ndarray],
        observation_jacobian: Callable[[np.ndarray], np.ndarray],
        process_noise: np.ndarray,
        measurement_noise: np.ndarray,
    ) -> None:
        self.transition_fn = transition_fn
        self.transition_jacobian = transition_jacobian
        self.observation_fn = observation_fn
        self.observation_jacobian = observation_jacobian
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise

    def predict(self, state: np.ndarray, covariance: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        transition = self.transition_jacobian(state)
        new_state = self.transition_fn(state)
        new_covariance = transition @ covariance @ transition.T + self.process_noise
        return new_state, new_covariance

    def update(
        self,
        state: np.ndarray,
        covariance: np.ndarray,
        measurement: np.ndarray,
        measurement_noise: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Update (state, covariance) with `measurement`.

        `measurement_noise` overrides the constructor's
        `measurement_noise` for this call -- same reasoning as
        `radarsim.tracking.filters.kalman.KalmanFilter.update`.
        """
        observation = self.observation_jacobian(state)
        noise = self.measurement_noise if measurement_noise is None else measurement_noise

        innovation = measurement - self.observation_fn(state)
        innovation_covariance = observation @ covariance @ observation.T + noise
        kalman_gain = covariance @ observation.T @ np.linalg.inv(innovation_covariance)

        new_state = state + kalman_gain @ innovation
        new_covariance = (np.eye(len(state)) - kalman_gain @ observation) @ covariance
        return new_state, new_covariance
