from __future__ import annotations

import numpy as np

from radarsim.types import RadarMeasurement


def polar_to_cartesian_measurement(
    measurement: RadarMeasurement, sensor_position: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Convert a polar `RadarMeasurement` to an approximate Cartesian position measurement.

    `x = sensor_x + range*cos(angle)`, `y = sensor_y + range*sin(angle)`
    -- range/angle are sensor-relative, so `sensor_position` is required
    to land in world-frame Cartesian coordinates (the frame
    `radarsim.tracking.Tracker`'s state lives in). The returned
    covariance is the polar (range, angle) covariance linearized
    through this transform's Jacobian -- the standard "converted
    measurement" technique for feeding polar radar returns into a
    Cartesian-state linear Kalman filter. This is an approximation:
    the true converted-measurement distribution is only approximately
    Gaussian, and de-biasing corrections exist but aren't implemented
    here. Radial velocity isn't converted -- this filter tracks
    position/velocity from position observations alone.
    """
    r = measurement.range
    theta = measurement.angle

    position = sensor_position + np.array([r * np.cos(theta), r * np.sin(theta)])

    sigma_range_sq = measurement.covariance[0, 0]
    sigma_angle_sq = measurement.covariance[2, 2]
    jacobian = np.array(
        [
            [np.cos(theta), -r * np.sin(theta)],
            [np.sin(theta), r * np.cos(theta)],
        ]
    )
    polar_covariance = np.diag([sigma_range_sq, sigma_angle_sq])
    position_covariance = jacobian @ polar_covariance @ jacobian.T

    return position, position_covariance
