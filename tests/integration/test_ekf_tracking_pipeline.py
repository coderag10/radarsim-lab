from pathlib import Path

import numpy as np

from radarsim.core.rng import SeededRNG
from radarsim.core.world import World
from radarsim.io.scenario import load_scenario
from radarsim.radar.model import RadarModel, calibrate_radar_constant
from radarsim.tracking.filters.extended_kalman import ExtendedKalmanFilter
from radarsim.tracking.polar_observation import polar_observation, polar_observation_jacobian
from radarsim.tracking.prediction import constant_velocity_transition


def test_ekf_tracks_two_target_scenario_directly_on_polar_measurements() -> None:
    config = load_scenario(Path("scenarios/basic/two_targets.yaml"))
    world = World.from_scenario(config)
    history = world.run()

    radar_constant = calibrate_radar_constant(
        reference_rcs=5.0, reference_range=30.0, reference_snr_db=20.0
    )
    sensor_position = np.zeros(2)
    radar = RadarModel(
        sensor_id="radar-1",
        position=sensor_position,
        noise_std=np.array([0.3, 0.1, 0.01]),
        radar_constant=radar_constant,
        rng=SeededRNG(seed=config.seed).spawn("radar"),
    )

    transition = constant_velocity_transition(dt=config.timestep)
    ekf = ExtendedKalmanFilter(
        transition_fn=lambda state: transition @ state,
        transition_jacobian=lambda state: transition,
        observation_fn=lambda state: polar_observation(state, sensor_position),
        observation_jacobian=lambda state: polar_observation_jacobian(state, sensor_position),
        process_noise=np.eye(4) * 0.05,
        measurement_noise=np.eye(3),  # overridden per-call below
    )

    tracks: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for truths in history:
        for truth in truths:
            measurement = radar.observe(truth)
            raw_measurement = np.array(
                [measurement.range, measurement.radial_velocity, measurement.angle]
            )
            if truth.target_id not in tracks:
                x = measurement.range * np.cos(measurement.angle)
                y = measurement.range * np.sin(measurement.angle)
                tracks[truth.target_id] = (np.array([x, y, 0.0, 0.0]), np.eye(4) * 10.0)
                continue

            state, covariance = tracks[truth.target_id]
            state, covariance = ekf.predict(state, covariance)
            state, covariance = ekf.update(
                state, covariance, raw_measurement, measurement_noise=measurement.covariance
            )
            tracks[truth.target_id] = (state, covariance)

    true_final = {truth.target_id: truth.position for truth in history[-1]}
    for target_id, (state, _) in tracks.items():
        assert np.linalg.norm(state[:2] - true_final[target_id]) < 2.0
