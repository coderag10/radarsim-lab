from pathlib import Path

import numpy as np

from radarsim.core.rng import SeededRNG
from radarsim.core.world import World
from radarsim.detection.detectors import ThresholdDetector
from radarsim.io.scenario import load_scenario
from radarsim.radar.model import RadarModel, calibrate_radar_constant
from radarsim.types import RadarMeasurement


def _observe_final_targets(radar_constant: float) -> list[RadarMeasurement]:
    config = load_scenario(Path("scenarios/basic/two_targets.yaml"))
    world = World.from_scenario(config)
    final_truths = world.run()[-1]

    radar = RadarModel(
        sensor_id="radar-1",
        position=np.zeros(2),
        noise_std=np.zeros(3),
        radar_constant=radar_constant,
        rng=SeededRNG(seed=config.seed).spawn("radar"),
    )
    return [radar.observe(truth) for truth in final_truths]


def test_low_threshold_detects_both_targets() -> None:
    radar_constant = calibrate_radar_constant(
        reference_rcs=5.0, reference_range=30.0, reference_snr_db=20.0
    )
    measurements = _observe_final_targets(radar_constant)

    detector = ThresholdDetector(snr_threshold_db=0.0)
    detections = detector.detect(measurements)

    assert len(detections) == 2


def test_high_threshold_detects_neither_target() -> None:
    radar_constant = calibrate_radar_constant(
        reference_rcs=5.0, reference_range=30.0, reference_snr_db=20.0
    )
    measurements = _observe_final_targets(radar_constant)

    detector = ThresholdDetector(snr_threshold_db=60.0)
    detections = detector.detect(measurements)

    assert len(detections) == 0
