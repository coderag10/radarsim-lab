from __future__ import annotations

import numpy as np

from radarsim.types import GroundTruth, RadarMeasurement

_MEASUREMENT_DIM = 3  # [range, radial_velocity, angle]


def calibrate_radar_constant(
    reference_rcs: float, reference_range: float, reference_snr_db: float
) -> float:
    """Solve `SNR_linear = radar_constant * rcs / range**4` for `radar_constant`.

    Lets a caller calibrate by a physically meaningful statement --
    "at `reference_range` with a `reference_rcs` target I want about
    `reference_snr_db` dB" -- instead of reasoning about the raw
    constant directly.
    """
    reference_snr_linear = 10.0 ** (reference_snr_db / 10.0)
    return float(reference_snr_linear * reference_range**4 / reference_rcs)


class RadarModel:
    """Converts true target state into a noisy `RadarMeasurement`.

    true state -> radar model -> + noise -> measurement

    Must never hand back or retain the `GroundTruth` it observes;
    downstream stages only ever see the `RadarMeasurement` it produces.

    Operates in the 2D (x, y) plane only -- bearing angle isn't
    well-defined for arbitrary dimensionality, and sensor
    orientation/boresight isn't modeled yet, so `angle` is the
    world-frame bearing (`atan2`), not relative to any radar heading.

    `RadarMeasurement.snr` follows the classic simplified radar range
    equation `SNR_linear = radar_constant * rcs / range**4`, computed
    from the target's *true* range/RCS (not the noisy measured range)
    -- it represents expected signal strength for that geometry, with
    no per-pulse fluctuation (Swerling) model, matching the same
    analytic-not-waveform-level simplification the rest of this class
    already makes. `radar_constant` lumps transmit power, antenna
    gain, wavelength, receiver noise figure, and system losses into
    one calibration knob -- see `calibrate_radar_constant`.
    """

    def __init__(
        self,
        sensor_id: str,
        position: np.ndarray,
        noise_std: np.ndarray,
        radar_constant: float,
        rng: np.random.Generator,
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
        self.radar_constant = radar_constant
        self._rng = rng

    def observe(self, truth: GroundTruth) -> RadarMeasurement:
        """Produce a synthetic noisy measurement of `truth`."""
        if truth.position.shape != (2,):
            raise ValueError(
                f"target {truth.target_id!r} position must be 2D, got shape {truth.position.shape}"
            )
        if truth.rcs <= 0.0:
            raise ValueError(f"target {truth.target_id!r} has non-positive rcs {truth.rcs!r}")

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

        snr_linear = self.radar_constant * truth.rcs / range_true**4
        snr_db = 10.0 * np.log10(snr_linear)

        return RadarMeasurement(
            timestamp=truth.timestamp,
            sensor_id=self.sensor_id,
            range=float(noisy[0]),
            radial_velocity=float(noisy[1]),
            angle=float(noisy[2]),
            covariance=np.diag(self.noise_std**2),
            snr=float(snr_db),
        )
