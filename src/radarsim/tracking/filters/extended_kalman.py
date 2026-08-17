from __future__ import annotations

from collections.abc import Callable

import numpy as np


class ExtendedKalmanFilter:
    """EKF for nonlinear motion/measurement models, linearized via Jacobians at each step."""

    def __init__(
        self,
        transition_fn: Callable[[np.ndarray], np.ndarray],
        transition_jacobian: Callable[[np.ndarray], np.ndarray],
        observation_fn: Callable[[np.ndarray], np.ndarray],
        observation_jacobian: Callable[[np.ndarray], np.ndarray],
        process_noise: np.ndarray,
        measurement_noise: np.ndarray,
    ) -> None:
        raise NotImplementedError

    def predict(self, state: np.ndarray, covariance: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    def update(
        self, state: np.ndarray, covariance: np.ndarray, measurement: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError
