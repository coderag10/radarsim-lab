import numpy as np

from radarsim.tracking.measurement_conversion import polar_to_cartesian_measurement
from radarsim.types import RadarMeasurement


def _measurement(
    range_: float, angle: float, covariance: np.ndarray | None = None
) -> RadarMeasurement:
    if covariance is None:
        covariance = np.diag([1.0, 1.0, 0.01])
    return RadarMeasurement(
        timestamp=0.0,
        sensor_id="radar-1",
        range=range_,
        radial_velocity=0.0,
        angle=angle,
        covariance=covariance,
        snr=20.0,
    )


def test_angle_zero_gives_position_on_x_axis() -> None:
    position, _ = polar_to_cartesian_measurement(
        _measurement(range_=5.0, angle=0.0), sensor_position=np.zeros(2)
    )
    np.testing.assert_allclose(position, [5.0, 0.0], atol=1e-9)


def test_angle_quarter_turn_gives_position_on_y_axis() -> None:
    position, _ = polar_to_cartesian_measurement(
        _measurement(range_=5.0, angle=np.pi / 2), sensor_position=np.zeros(2)
    )
    np.testing.assert_allclose(position, [0.0, 5.0], atol=1e-9)


def test_sensor_position_offsets_result_into_world_frame() -> None:
    position, _ = polar_to_cartesian_measurement(
        _measurement(range_=5.0, angle=0.0), sensor_position=np.array([10.0, -20.0])
    )
    np.testing.assert_allclose(position, [15.0, -20.0], atol=1e-9)


def test_covariance_jacobian_at_zero_angle() -> None:
    # [range_var, radial_velocity_var(unused), angle_var]
    covariance_polar = np.diag([2.0, 99.0, 0.5])
    _, position_covariance = polar_to_cartesian_measurement(
        _measurement(range_=10.0, angle=0.0, covariance=covariance_polar),
        sensor_position=np.zeros(2),
    )
    # at angle=0: jacobian = [[1, 0], [0, r]] -> cov = diag(range_var, r^2 * angle_var)
    expected = np.diag([2.0, 10.0**2 * 0.5])
    np.testing.assert_allclose(position_covariance, expected, atol=1e-9)
