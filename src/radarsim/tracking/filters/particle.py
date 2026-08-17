from __future__ import annotations

from collections.abc import Callable

import numpy as np


class ParticleFilter:
    """Sequential Monte Carlo filter for non-Gaussian / highly nonlinear estimation problems."""

    def __init__(
        self,
        num_particles: int,
        transition_fn: Callable[[np.ndarray, np.random.Generator], np.ndarray],
        likelihood_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
        rng: np.random.Generator,
    ) -> None:
        raise NotImplementedError

    def predict(self) -> None:
        raise NotImplementedError

    def update(self, measurement: np.ndarray) -> None:
        raise NotImplementedError

    def estimate(self) -> tuple[np.ndarray, np.ndarray]:
        """Return the (weighted mean, covariance) point estimate of the particle set."""
        raise NotImplementedError
