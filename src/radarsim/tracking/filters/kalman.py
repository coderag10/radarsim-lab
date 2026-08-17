from __future__ import annotations

import numpy as np


class KalmanFilter:
    """Linear Kalman filter for (approximately) linear-Gaussian motion models."""

    def __init__(
        self,
        transition: np.ndarray,
        observation: np.ndarray,
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
