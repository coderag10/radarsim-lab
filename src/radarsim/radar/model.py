from __future__ import annotations

import numpy as np

from radarsim.types import GroundTruth, RadarMeasurement

_MEASUREMENT_DIM = 3  # [range, radial_velocity, angle]


class RadarModel:
    """Converts true target state into a noisy `RadarMeasurement`.

    true state -> radar model -> + noise -> measurement

    Must never hand back or retain the `GroundTruth` it observes;
    downstream stages only ever see the `RadarMeasurement` it produces.

    Operates in the 2D (x, y) plane only -- bearing angle isn't
    well-defined for arbitrary dimensionality, and sensor
    orientation/boresight isn't modeled yet, so `angle` is the
    world-frame bearing (`atan2`), not relative to any radar heading.
    """

    def __init__(
        self, sensor_id: str, position: np.ndarray, noise_std: np.ndarray, rng: np.random.Generator
    ) -> None:
        if position.shape != (2,):
            raise ValueError(f"radar {sensor_id!r} position must be 2D, got shape {position.shape}")
        if noise_std.shape != (_MEASUREMENT_DIM,):
            raise ValueError(
                f"radar {sensor_id!r} noise_std must have shape ({_MEASUREMENT_DIM},) "
                f"(range, radial_velocity, angle), got shape {noise_std.shape}"
            )

        self.sensor_id = sensor_id
        self.position = position
        self.noise_std = noise_std
        self._rng = rng

    def observe(self, truth: GroundTruth) -> RadarMeasurement:
        """Produce a synthetic noisy measurement of `truth`."""
        if truth.position.shape != (2,):
            raise ValueError(
                f"target {truth.target_id!r} position must be 2D, got shape {truth.position.shape}"
            )

        line_of_sight = truth.position - self.position
        range_true = float(np.linalg.norm(line_of_sight))
        if range_true == 0.0:
            raise ValueError(
                f"target {truth.target_id!r} is co-located with radar {self.sensor_id!r}; "
                "range/angle are undefined"
            )

        unit_los = line_of_sight / range_true
        radial_velocity_true = float(np.dot(truth.velocity, unit_los))
        angle_true = float(np.arctan2(line_of_sight[1], line_of_sight[0]))

        true_values = np.array([range_true, radial_velocity_true, angle_true])
        noisy = true_values + self._rng.normal(0.0, self.noise_std)

        return RadarMeasurement(
            timestamp=truth.timestamp,
            sensor_id=self.sensor_id,
            range=float(noisy[0]),
            radial_velocity=float(noisy[1]),
            angle=float(noisy[2]),
            covariance=np.diag(self.noise_std**2),
        )
