import numpy as np
import pytest

from radarsim.tracking.filters.particle import ParticleFilter


def _make_filter(
    num_particles: int = 5,
    transition_fn=lambda state, rng: state,
    likelihood_fn=lambda state, measurement: 1.0,
    initial_state: np.ndarray | None = None,
    initial_covariance: np.ndarray | None = None,
    seed: int = 0,
) -> ParticleFilter:
    return ParticleFilter(
        num_particles=num_particles,
        transition_fn=transition_fn,
        likelihood_fn=likelihood_fn,
        rng=np.random.default_rng(seed),
        initial_state=np.zeros(1) if initial_state is None else initial_state,
        initial_covariance=np.eye(1) if initial_covariance is None else initial_covariance,
    )


def test_initialize_particle_count_and_uniform_weights() -> None:
    pf = _make_filter(num_particles=100, initial_state=np.zeros(2), initial_covariance=np.eye(2))
    assert pf.particles.shape == (100, 2)
    np.testing.assert_allclose(pf.weights, np.full(100, 0.01))


def test_estimate_matches_hand_computed_weighted_mean_and_covariance() -> None:
    pf = _make_filter(num_particles=3)
    pf.particles = np.array([[0.0], [2.0], [4.0]])
    pf.weights = np.array([0.5, 0.25, 0.25])

    mean, covariance = pf.estimate()

    np.testing.assert_allclose(mean, [1.5])
    expected_var = 0.5 * (0 - 1.5) ** 2 + 0.25 * (2 - 1.5) ** 2 + 0.25 * (4 - 1.5) ** 2
    np.testing.assert_allclose(covariance, [[expected_var]])


def test_predict_applies_transition_fn_to_every_particle() -> None:
    pf = _make_filter(
        num_particles=5,
        transition_fn=lambda state, rng: state + 1.0,
        initial_state=np.zeros(1),
        initial_covariance=np.zeros((1, 1)),  # degenerate: all particles start exactly at 0
    )
    pf.predict()
    np.testing.assert_allclose(pf.particles, np.ones((5, 1)))


def test_update_reweights_by_likelihood_and_renormalizes() -> None:
    pf = _make_filter(
        num_particles=2,
        likelihood_fn=lambda state, measurement: float(state[0]),
    )
    pf.particles = np.array([[1.0], [3.0]])
    pf.weights = np.array([0.5, 0.5])

    pf.update(measurement=np.zeros(1))

    # unnormalized = 0.5*[1,3] = [0.5, 1.5] -> normalized [0.25, 0.75]
    # ESS = 1/(0.25^2+0.75^2) = 1.6 > num_particles/2=1.0 -> no resampling
    np.testing.assert_allclose(pf.weights, [0.25, 0.75])


def test_update_resamples_when_effective_sample_size_is_low() -> None:
    pf = _make_filter(
        num_particles=4,
        likelihood_fn=lambda state, measurement: 1.0 if state[0] > 0.5 else 1e-9,
    )
    pf.particles = np.array([[1.0], [0.0], [0.0], [0.0]])
    pf.weights = np.full(4, 0.25)

    pf.update(measurement=np.zeros(1))

    np.testing.assert_allclose(pf.weights, np.full(4, 0.25))
    np.testing.assert_allclose(pf.particles, np.full((4, 1), 1.0))


def test_update_all_zero_likelihood_raises_value_error() -> None:
    pf = _make_filter(likelihood_fn=lambda state, measurement: 0.0)
    with pytest.raises(ValueError, match="likelihood"):
        pf.update(measurement=np.zeros(1))
