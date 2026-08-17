from __future__ import annotations

import numpy as np


def polar_observation(state: np.ndarray, sensor_position: np.ndarray) -> np.ndarray:
    """Nonlinear observation model h(state) = [range, radial_velocity, angle].

    `state` is `[x, y, vx, vy]`. Field order matches
    `RadarMeasurement.covariance`'s diagonal exactly, so an EKF using
    this model can consume `measurement.covariance` directly -- no
    polar->Cartesian conversion needed (contrast with
    `radarsim.tracking.measurement_conversion`, which the linear-KF
    path needs precisely because it *doesn't* use this nonlinear model).
    """
    dx = state[0] - sensor_position[0]
    dy = state[1] - sensor_position[1]
    vx, vy = state[2], state[3]

    r = float(np.hypot(dx, dy))
    radial_velocity = (dx * vx + dy * vy) / r
    angle = float(np.arctan2(dy, dx))

    return np.array([r, radial_velocity, angle])


def polar_observation_jacobian(state: np.ndarray, sensor_position: np.ndarray) -> np.ndarray:
    """Jacobian of `polar_observation` with respect to `state`, evaluated at `state`.

    Closed-form partial derivatives of [range, radial_velocity, angle]
    w.r.t. [x, y, vx, vy]. `radial_velocity`'s derivatives w.r.t.
    position use the quotient rule: d(rv)/dx = vx/r - rv*dx/r^2 (and
    symmetrically for y).
    """
    dx = state[0] - sensor_position[0]
    dy = state[1] - sensor_position[1]
    vx, vy = state[2], state[3]

    r = float(np.hypot(dx, dy))
    radial_velocity = (dx * vx + dy * vy) / r

    d_range = np.array([dx / r, dy / r, 0.0, 0.0])
    d_radial_velocity = np.array(
        [
            vx / r - radial_velocity * dx / r**2,
            vy / r - radial_velocity * dy / r**2,
            dx / r,
            dy / r,
        ]
    )
    d_angle = np.array([-dy / r**2, dx / r**2, 0.0, 0.0])

    return np.array([d_range, d_radial_velocity, d_angle])
