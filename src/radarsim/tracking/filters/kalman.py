from __future__ import annotations

import numpy as np

from radarsim.tracking.prediction import predict_state


class KalmanFilter:
    """Linear Kalman filter for (approximately) linear-Gaussian motion models."""

    def __init__(
        self,
        transition: np.ndarray,
        observation: np.ndarray,
        process_noise: np.ndarray,
        measurement_noise: np.ndarray,
    ) -> None:
        self.transition = transition
        self.observation = observation
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise

    def predict(self, state: np.ndarray, covariance: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return predict_state(state, covariance, self.transition, self.process_noise)

    def update(
        self,
        state: np.ndarray,
        covariance: np.ndarray,
        measurement: np.ndarray,
        measurement_noise: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Update (state, covariance) with `measurement` via the observation model.

        `measurement_noise` overrides the constructor's `measurement_noise`
        for this call -- real measurement covariance often varies per
        observation (e.g. a polar->Cartesian conversion's covariance
        depends on that measurement's range/angle), so the constructor
        value is only a fallback default, not the last word.
        """
        observation = self.observation
        noise = self.measurement_noise if measurement_noise is None else measurement_noise

        innovation = measurement - observation @ state
        innovation_covariance = observation @ covariance @ observation.T + noise
        kalman_gain = covariance @ observation.T @ np.linalg.inv(innovation_covariance)

        new_state = state + kalman_gain @ innovation
        new_covariance = (np.eye(len(state)) - kalman_gain @ observation) @ covariance
        return new_state, new_covariance
