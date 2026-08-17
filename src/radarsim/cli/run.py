from __future__ import annotations

import json
from dataclasses import dataclass
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
from radarsim.types import TrackEstimate, TrackStatus

# Tracker/filter internals, not exposed as CLI flags -- the same values
# validated across every integration test since Phase 5.
_GATE_THRESHOLD = 10.0
_CONFIRM_HITS = 3
_MAX_MISSES = 3
_PROCESS_NOISE_SCALE = 0.05
_INITIAL_VELOCITY_VARIANCE = 100.0

_SENSOR_ID = "radar-1"


@dataclass(frozen=True, slots=True)
class RunResult:
    """Summary of one full scenario run: World -> RadarModel -> ThresholdDetector -> Tracker."""

    scenario_path: str
    duration: float
    timestep: float
    seed: int
    num_targets: int
    sensor_id: str
    sensor_position: tuple[float, float]
    tracks: list[TrackEstimate]
    detection_probability: float
    position_rmse: float | None


def run_scenario(
    scenario_path: Path,
    *,
    radar_position: np.ndarray,
    noise_std: np.ndarray,
    reference_rcs: float,
    reference_range: float,
    reference_snr_db: float,
    snr_threshold_db: float,
) -> RunResult:
    """Run a scenario end-to-end and summarize the result.

    `position_rmse` is `None` (not a crash) when no track reaches
    `ACTIVE` -- a real, reachable outcome if `snr_threshold_db` is set
    too high for anything to be detected.
    """
    config = load_scenario(scenario_path)
    world = World.from_scenario(config)
    history = world.run()

    radar_constant = calibrate_radar_constant(reference_rcs, reference_range, reference_snr_db)
    radar = RadarModel(
        sensor_id=_SENSOR_ID,
        position=radar_position,
        noise_std=noise_std,
        radar_constant=radar_constant,
        rng=SeededRNG(seed=config.seed).spawn("radar"),
    )
    detector = ThresholdDetector(snr_threshold_db=snr_threshold_db)
    kalman_filter = KalmanFilter(
        transition=constant_velocity_transition(dt=config.timestep),
        observation=np.eye(2, 4),
        process_noise=np.eye(4) * _PROCESS_NOISE_SCALE,
        measurement_noise=np.eye(2),
    )
    tracker = Tracker(
        kalman_filter=kalman_filter,
        association=NearestNeighbor(gate_threshold=_GATE_THRESHOLD, sensor_position=radar_position),
        sensor_position=radar_position,
        confirm_hits=_CONFIRM_HITS,
        max_misses=_MAX_MISSES,
        initial_velocity_variance=_INITIAL_VELOCITY_VARIANCE,
    )

    all_target_ids = {truth.target_id for truth in history[0]}
    detected_target_ids: set[str] = set()
    tracks: list[TrackEstimate] = []
    for truths in history:
        measurements = [radar.observe(truth) for truth in truths]
        detections = detector.detect(measurements)
        detected_measurements = {detection.measurement for detection in detections}
        for truth, measurement in zip(truths, measurements, strict=True):
            if measurement in detected_measurements:
                detected_target_ids.add(truth.target_id)
        tracks = tracker.step(truths[0].timestamp, detections)

    pd = detection_probability(len(detected_target_ids), len(all_target_ids))

    active_tracks = [track for track in tracks if track.status == TrackStatus.ACTIVE]
    rmse = position_rmse(active_tracks, history[-1]) if active_tracks else None

    return RunResult(
        scenario_path=scenario_path.as_posix(),
        duration=config.duration,
        timestep=config.timestep,
        seed=config.seed,
        num_targets=len(all_target_ids),
        sensor_id=_SENSOR_ID,
        sensor_position=(float(radar_position[0]), float(radar_position[1])),
        tracks=tracks,
        detection_probability=pd,
        position_rmse=rmse,
    )


def format_table(result: RunResult) -> str:
    lines = [
        f"Scenario: {result.scenario_path}",
        f"  Duration: {result.duration}s  Timestep: {result.timestep}s  "
        f"Seed: {result.seed}  Targets: {result.num_targets}",
        "",
        f"Sensor: {result.sensor_id} @ ({result.sensor_position[0]}, {result.sensor_position[1]})",
        "",
    ]

    timestamp = result.tracks[0].timestamp if result.tracks else result.duration
    lines.append(f"Tracking results (t={timestamp}s):")
    if not result.tracks:
        lines.append("  (no tracks)")
    else:
        lines.append(f"  {'ID':<10}{'Status':<10}{'Position':<20}{'Velocity':<20}")
        for track in result.tracks:
            position = f"({track.state[0]:.2f}, {track.state[1]:.2f})"
            velocity = f"({track.state[2]:.2f}, {track.state[3]:.2f})"
            row = f"  {track.track_id:<10}{track.status.name:<10}{position:<20}{velocity:<20}"
            lines.append(row)

    rmse_str = f"{result.position_rmse:.3f}" if result.position_rmse is not None else "N/A"
    lines += [
        "",
        "Metrics:",
        f"  Detection probability: {result.detection_probability * 100:.1f}%",
        f"  Position RMSE:          {rmse_str}",
    ]
    return "\n".join(lines)


def format_json(result: RunResult) -> str:
    payload = {
        "scenario": result.scenario_path,
        "duration": result.duration,
        "timestep": result.timestep,
        "seed": result.seed,
        "num_targets": result.num_targets,
        "sensor": {"id": result.sensor_id, "position": list(result.sensor_position)},
        "tracks": [
            {
                "track_id": track.track_id,
                "status": track.status.name,
                "position": [float(track.state[0]), float(track.state[1])],
                "velocity": [float(track.state[2]), float(track.state[3])],
            }
            for track in result.tracks
        ],
        "metrics": {
            "detection_probability": result.detection_probability,
            "position_rmse": result.position_rmse,
        },
    }
    return json.dumps(payload, indent=2)
