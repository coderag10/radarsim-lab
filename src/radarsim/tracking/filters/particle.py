from __future__ import annotations

from collections.abc import Callable

import numpy as np


class ParticleFilter:
    """Sequential Monte Carlo filter for non-Gaussian / highly nonlinear estimation problems.

    Signature note: the constructor requires `initial_state`/
    `initial_covariance` (not in the original stub) to seed the
    particle set immediately, rather than a separate `initialize()`
    call -- this filter is inherently stateful (unlike
    `KalmanFilter`/`ExtendedKalmanFilter`, which take state/covariance
    per call), so a single-shot construction avoids `Optional`-typed
    particle/weight fields. `likelihood_fn`'s return type is corrected
    to `float` (a per-particle scalar weight), not the stub's
    `np.ndarray`.
    """

    def __init__(
        self,
        num_particles: int,
        transition_fn: Callable[[np.ndarray, np.random.Generator], np.ndarray],
        likelihood_fn: Callable[[np.ndarray, np.ndarray], float],
        rng: np.random.Generator,
        initial_state: np.ndarray,
        initial_covariance: np.ndarray,
    ) -> None:
        self.num_particles = num_particles
        self.transition_fn = transition_fn
        self.likelihood_fn = likelihood_fn
        self._rng = rng
        self.particles = rng.multivariate_normal(
            initial_state, initial_covariance, size=num_particles
        )
        self.weights = np.full(num_particles, 1.0 / num_particles)

    def predict(self) -> None:
        self.particles = np.array(
            [self.transition_fn(particle, self._rng) for particle in self.particles]
        )

    def update(self, measurement: np.ndarray) -> None:
        likelihoods = np.array(
            [self.likelihood_fn(particle, measurement) for particle in self.particles]
        )
        unnormalized = self.weights * likelihoods
        total = unnormalized.sum()
        if total == 0.0:
            raise ValueError(
                "all particle likelihoods are zero; measurement is inconsistent with every particle"
            )
        self.weights = unnormalized / total
        self._resample_if_needed()

    def _resample_if_needed(self) -> None:
        effective_sample_size = 1.0 / np.sum(self.weights**2)
        if effective_sample_size < self.num_particles / 2:
            indices = self._rng.choice(
                self.num_particles, size=self.num_particles, replace=True, p=self.weights
            )
            self.particles = self.particles[indices]
            self.weights = np.full(self.num_particles, 1.0 / self.num_particles)

    def estimate(self) -> tuple[np.ndarray, np.ndarray]:
        """Return the (weighted mean, covariance) point estimate of the particle set."""
        mean = np.average(self.particles, axis=0, weights=self.weights)
        deviations = self.particles - mean
        covariance = (deviations.T * self.weights) @ deviations
        return mean, covariance
