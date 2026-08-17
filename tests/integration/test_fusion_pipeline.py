from pathlib import Path

import numpy as np

from radarsim.core.rng import SeededRNG
from radarsim.core.world import World
from radarsim.detection.detectors import ThresholdDetector
from radarsim.fusion.fuse import fuse_estimates, match_tracks
from radarsim.io.scenario import load_scenario
from radarsim.metrics.metrics import position_rmse
from radarsim.radar.model import RadarModel, calibrate_radar_constant
from radarsim.tracking.association import NearestNeighbor
from radarsim.tracking.filters.kalman import KalmanFilter
from radarsim.tracking.prediction import constant_velocity_transition
from radarsim.tracking.tracker import Tracker
from radarsim.types import GroundTruth, TrackStatus


def _build_tracker(dt: float, sensor_position: np.ndarray) -> Tracker:
    kalman_filter = KalmanFilter(
        transition=constant_velocity_transition(dt=dt),
        observation=np.eye(2, 4),
        process_noise=np.eye(4) * 0.05,
        measurement_noise=np.eye(2),
    )
    return Tracker(
        kalman_filter=kalman_filter,
        association=NearestNeighbor(gate_threshold=10.0, sensor_position=sensor_position),
        sensor_position=sensor_position,
        confirm_hits=3,
        max_misses=3,
    )


def _run_trial(
    history: list[list[GroundTruth]],
    dt: float,
    position_a: np.ndarray,
    position_b: np.ndarray,
    radar_constant: float,
    seed: int,
) -> tuple[float, float, float]:
    """Run both sensors' independent pipelines for one noise realization.

    Returns (rmse_a, rmse_b, rmse_fused) for that single trial.
    """
    radar_a = RadarModel(
        sensor_id="radar-a",
        position=position_a,
        noise_std=np.array([0.3, 0.1, 0.01]),
        radar_constant=radar_constant,
        rng=SeededRNG(seed=seed).spawn("radar-a"),
    )
    radar_b = RadarModel(
        sensor_id="radar-b",
        position=position_b,
        noise_std=np.array([0.3, 0.1, 0.01]),
        radar_constant=radar_constant,
        rng=SeededRNG(seed=seed).spawn("radar-b"),
    )
    detector = ThresholdDetector(snr_threshold_db=0.0)
    tracker_a = _build_tracker(dt, position_a)
    tracker_b = _build_tracker(dt, position_b)

    tracks_a: list = []
    tracks_b: list = []
    for truths in history:
        detections_a = detector.detect([radar_a.observe(truth) for truth in truths])
        detections_b = detector.detect([radar_b.observe(truth) for truth in truths])
        tracks_a = tracker_a.step(truths[0].timestamp, detections_a)
        tracks_b = tracker_b.step(truths[0].timestamp, detections_b)

    active_a = [t for t in tracks_a if t.status == TrackStatus.ACTIVE]
    active_b = [t for t in tracks_b if t.status == TrackStatus.ACTIVE]
    truth = history[-1]

    rmse_a = position_rmse(active_a, truth)
    rmse_b = position_rmse(active_b, truth)

    pairs = match_tracks(active_a, active_b, gate_threshold=10.0)
    fused_tracks = [fuse_estimates([a, b]) for a, b in pairs]
    rmse_fused = position_rmse(fused_tracks, truth)

    return rmse_a, rmse_b, rmse_fused


def test_two_radar_fusion_reduces_expected_position_error() -> None:
    """Inverse-covariance-weighted fusion minimizes *expected* squared
    error, not necessarily the error of any single noisy realization --
    the `test_fuse.py` property test already covers the guaranteed
    property (fused covariance trace never exceeds either input's).
    Here, pooling squared error across many independent noise draws is
    the statistically honest way to check fusion actually helps on this
    scenario, rather than asserting a single-sample comparison that the
    theorem doesn't actually promise.
    """
    config = load_scenario(Path("scenarios/basic/two_targets.yaml"))
    world = World.from_scenario(config)
    history = world.run()

    radar_constant = calibrate_radar_constant(
        reference_rcs=5.0, reference_range=30.0, reference_snr_db=20.0
    )
    position_a = np.array([0.0, 0.0])
    position_b = np.array([0.0, 50.0])

    num_trials = 40
    results = [
        _run_trial(
            history, config.timestep, position_a, position_b, radar_constant, config.seed + i
        )
        for i in range(num_trials)
    ]

    mean_sq_a = float(np.mean([rmse_a**2 for rmse_a, _, _ in results]))
    mean_sq_b = float(np.mean([rmse_b**2 for _, rmse_b, _ in results]))
    mean_sq_fused = float(np.mean([rmse_fused**2 for _, _, rmse_fused in results]))

    pooled_rmse_a = np.sqrt(mean_sq_a)
    pooled_rmse_b = np.sqrt(mean_sq_b)
    pooled_rmse_fused = np.sqrt(mean_sq_fused)

    assert pooled_rmse_fused <= min(pooled_rmse_a, pooled_rmse_b)
