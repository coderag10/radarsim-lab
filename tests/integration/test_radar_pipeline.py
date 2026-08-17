from pathlib import Path

import numpy as np
import pytest

from radarsim.core.rng import SeededRNG
from radarsim.core.world import World
from radarsim.io.scenario import load_scenario
from radarsim.radar.model import RadarModel


def test_radar_observes_world_ground_truth_zero_noise() -> None:
    config = load_scenario(Path("scenarios/basic/two_targets.yaml"))
    world = World.from_scenario(config)
    final_truths = world.run()[-1]

    radar = RadarModel(
        sensor_id="radar-1",
        position=np.zeros(2),
        noise_std=np.zeros(3),
        radar_constant=1.0,
        rng=SeededRNG(seed=config.seed).spawn("radar"),
    )

    for truth in final_truths:
        measurement = radar.observe(truth)
        expected_range = float(np.linalg.norm(truth.position))
        unit_los = truth.position / expected_range
        expected_radial_velocity = float(np.dot(truth.velocity, unit_los))

        assert measurement.range == pytest.approx(expected_range)
        assert measurement.radial_velocity == pytest.approx(expected_radial_velocity)
        assert measurement.sensor_id == "radar-1"
        assert measurement.timestamp == truth.timestamp


def test_radar_observes_world_ground_truth_with_noise_stays_close_to_truth() -> None:
    config = load_scenario(Path("scenarios/basic/two_targets.yaml"))
    world = World.from_scenario(config)
    final_truths = world.run()[-1]

    noise_std = np.array([0.5, 0.2, 0.01])
    radar = RadarModel(
        sensor_id="radar-1",
        position=np.zeros(2),
        noise_std=noise_std,
        radar_constant=1.0,
        rng=SeededRNG(seed=config.seed).spawn("radar"),
    )

    for truth in final_truths:
        measurement = radar.observe(truth)
        expected_range = float(np.linalg.norm(truth.position))

        assert np.isfinite(measurement.range)
        assert abs(measurement.range - expected_range) < 5 * noise_std[0]
