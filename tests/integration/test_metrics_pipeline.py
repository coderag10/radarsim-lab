from pathlib import Path

import numpy as np

from radarsim.core.rng import SeededRNG
from radarsim.core.world import World
from radarsim.detection.detectors import ThresholdDetector
from radarsim.io.scenario import load_scenario
from radarsim.metrics.metrics import detection_probability, position_rmse
from radarsim.radar.model import RadarModel, calibrate_radar_constant
from radarsim.tracking.association import NearestNeighbor
from radarsim.tracking.filters.kalman import KalmanFilter
from radarsim.tracking.prediction import constant_velocity_transition
from radarsim.tracking.tracker import Tracker
from radarsim.types import TrackStatus


def test_metrics_over_full_pipeline() -> None:
    config = load_scenario(Path("scenarios/basic/two_targets.yaml"))
    world = World.from_scenario(config)
    history = world.run()

    radar_constant = calibrate_radar_constant(
        reference_rcs=5.0, reference_range=30.0, reference_snr_db=20.0
    )
    radar = RadarModel(
        sensor_id="radar-1",
        position=np.zeros(2),
        noise_std=np.array([0.3, 0.1, 0.01]),
        radar_constant=radar_constant,
        rng=SeededRNG(seed=config.seed).spawn("radar"),
    )
    detector = ThresholdDetector(snr_threshold_db=0.0)
    kalman_filter = KalmanFilter(
        transition=constant_velocity_transition(dt=config.timestep),
        observation=np.eye(2, 4),
        process_noise=np.eye(4) * 0.05,
        measurement_noise=np.eye(2),
    )
    tracker = Tracker(
        kalman_filter=kalman_filter,
        association=NearestNeighbor(gate_threshold=10.0),
        confirm_hits=3,
        max_misses=3,
    )

    all_target_ids = {truth.target_id for truth in history[0]}
    detected_target_ids: set[str] = set()
    final_tracks = []
    for truths in history:
        measurements = [radar.observe(truth) for truth in truths]
        detections = detector.detect(measurements)
        detected_measurements = {d.measurement for d in detections}
        for truth, measurement in zip(truths, measurements, strict=True):
            if measurement in detected_measurements:
                detected_target_ids.add(truth.target_id)
        final_tracks = tracker.step(truths[0].timestamp, detections)

    pd = detection_probability(len(detected_target_ids), len(all_target_ids))
    assert pd == 1.0

    active_tracks = [t for t in final_tracks if t.status == TrackStatus.ACTIVE]
    rmse = position_rmse(active_tracks, history[-1])
    assert rmse < 2.0
