from __future__ import annotations

import numpy as np


def predict_state(
    state: np.ndarray,
    covariance: np.ndarray,
    transition: np.ndarray,
    process_noise: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Propagate a track's (state, covariance) forward one timestep.

    state' = transition @ state
    covariance' = transition @ covariance @ transition.T + process_noise
    """
    new_state = transition @ state
    new_covariance = transition @ covariance @ transition.T + process_noise
    return new_state, new_covariance


def constant_velocity_transition(dt: float) -> np.ndarray:
    """4x4 constant-velocity transition matrix for state `[x, y, vx, vy]`.

    `x' = x + vx*dt`, `y' = y + vy*dt`, velocity unchanged -- the same
    model as `radarsim.targets.motion_models.ConstantVelocityModel`,
    expressed as a matrix for use with `predict_state`/`KalmanFilter`.
    """
    return np.array(
        [
            [1.0, 0.0, dt, 0.0],
            [0.0, 1.0, 0.0, dt],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
