from __future__ import annotations

from pydantic import BaseModel

from radarsim.cli.run import RunResult


class RunRequest(BaseModel):
    """Request body for `POST /runs`. Mirrors the CLI's flags and defaults exactly."""

    scenario: str
    radar_x: float = 0.0
    radar_y: float = 0.0
    noise_std: tuple[float, float, float] = (0.3, 0.1, 0.01)
    reference_rcs: float = 5.0
    reference_range: float = 30.0
    reference_snr_db: float = 20.0
    snr_threshold_db: float = 0.0


class SensorInfo(BaseModel):
    id: str
    position: tuple[float, float]


class GroundTruthOut(BaseModel):
    target_id: str
    position: tuple[float, float]


class TrackOut(BaseModel):
    track_id: str
    status: str
    position: tuple[float, float]
    velocity: tuple[float, float]


class MetricsOut(BaseModel):
    detection_probability: float
    position_rmse: float | None


class RunResponse(BaseModel):
    """Response body for `POST /runs`. Same shape `cli.run.format_json` produces."""

    scenario: str
    duration: float
    timestep: float
    seed: int
    num_targets: int
    sensor: SensorInfo
    ground_truth: list[GroundTruthOut]
    tracks: list[TrackOut]
    metrics: MetricsOut

    @classmethod
    def from_run_result(cls, result: RunResult) -> RunResponse:
        return cls(
            scenario=result.scenario_path,
            duration=result.duration,
            timestep=result.timestep,
            seed=result.seed,
            num_targets=result.num_targets,
            sensor=SensorInfo(id=result.sensor_id, position=result.sensor_position),
            ground_truth=[
                GroundTruthOut(
                    target_id=truth.target_id,
                    position=(float(truth.position[0]), float(truth.position[1])),
                )
                for truth in result.ground_truth
            ],
            tracks=[
                TrackOut(
                    track_id=track.track_id,
                    status=track.status.name,
                    position=(float(track.state[0]), float(track.state[1])),
                    velocity=(float(track.state[2]), float(track.state[3])),
                )
                for track in result.tracks
            ],
            metrics=MetricsOut(
                detection_probability=result.detection_probability,
                position_rmse=result.position_rmse,
            ),
        )
