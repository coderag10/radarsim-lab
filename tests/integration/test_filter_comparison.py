import numpy as np

from radarsim.tracking.filters.kalman import KalmanFilter
from radarsim.tracking.filters.particle import ParticleFilter


def test_particle_filter_converges_to_kalman_filter_on_linear_gaussian_problem() -> None:
    """A correct particle filter should approximate the Kalman filter's exact
    posterior on a problem where the KF is provably optimal (linear-Gaussian):
    constant-velocity position/velocity tracking from noisy position-only
    measurements. Agreement here is real evidence the reweight/resample
    machinery works, not just that the filter runs without crashing.
    """
    rng_measurements = np.random.default_rng(42)

    dt = 1.0
    true_velocity = 1.0
    position_noise_std = 0.5
    num_steps = 20

    transition = np.array([[1.0, dt], [0.0, 1.0]])
    process_noise = np.diag([0.001, 0.001])
    observation = np.array([[1.0, 0.0]])
    measurement_noise = np.array([[position_noise_std**2]])

    true_position = 0.0
    measurements = []
    for _ in range(num_steps):
        true_position += true_velocity * dt
        measurements.append(true_position + rng_measurements.normal(0.0, position_noise_std))

    kf = KalmanFilter(
        transition=transition,
        observation=observation,
        process_noise=process_noise,
        measurement_noise=measurement_noise,
    )
    kf_state = np.array([0.0, 0.0])
    kf_covariance = np.eye(2) * 10.0
    for measurement in measurements:
        kf_state, kf_covariance = kf.predict(kf_state, kf_covariance)
        kf_state, kf_covariance = kf.update(kf_state, kf_covariance, np.array([measurement]))

    def transition_fn(state: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        noise = rng.multivariate_normal(np.zeros(2), process_noise)
        return transition @ state + noise

    def likelihood_fn(state: np.ndarray, measurement: np.ndarray) -> float:
        residual = measurement[0] - state[0]
        return float(np.exp(-0.5 * (residual / position_noise_std) ** 2))

    pf = ParticleFilter(
        num_particles=5000,
        transition_fn=transition_fn,
        likelihood_fn=likelihood_fn,
        rng=np.random.default_rng(7),
        initial_state=np.array([0.0, 0.0]),
        initial_covariance=np.eye(2) * 10.0,
    )
    for measurement in measurements:
        pf.predict()
        pf.update(np.array([measurement]))
    pf_state, _ = pf.estimate()

    assert abs(pf_state[0] - kf_state[0]) < 1.0
    assert abs(pf_state[1] - kf_state[1]) < 1.0
