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
    raise NotImplementedError
