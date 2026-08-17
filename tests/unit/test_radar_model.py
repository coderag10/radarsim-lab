import numpy as np
import pytest
from hypothesis import assume, given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from radarsim.core.rng import SeededRNG
from radarsim.radar.model import RadarModel, calibrate_radar_constant
from radarsim.types import GroundTruth

_RADAR_CONSTANT = 1.0  # placeholder for tests that don't assert on SNR


def _truth(
    position: list[float],
    velocity: list[float],
    timestamp: float = 1.0,
    target_id: str = "t1",
    rcs: float = 1.0,
) -> GroundTruth:
    position_arr = np.array(position, dtype=float)
    return GroundTruth(
        timestamp=timestamp,
        target_id=target_id,
        position=position_arr,
        velocity=np.array(velocity, dtype=float),
        acceleration=np.zeros_like(position_arr),
        rcs=rcs,
    )


def _zero_noise_rng() -> np.random.Generator:
    return np.random.default_rng(0)


def test_observe_hand_computed_no_noise() -> None:
    radar = RadarModel(
        sensor_id="radar-1",
        position=np.zeros(2),
        noise_std=np.zeros(3),
        radar_constant=_RADAR_CONSTANT,
        rng=_zero_noise_rng(),
    )
    truth = _truth(position=[3.0, 4.0], velocity=[0.0, 5.0], timestamp=2.5)

    measurement = radar.observe(truth)

    assert measurement.range == pytest.approx(5.0)
    assert measurement.radial_velocity == pytest.approx(4.0)
    assert measurement.angle == pytest.approx(np.arctan2(4.0, 3.0))
    assert measurement.timestamp == 2.5
    assert measurement.sensor_id == "radar-1"


def test_covariance_is_diagonal_of_noise_std_squared() -> None:
    noise_std = np.array([1.0, 2.0, 0.1])
    radar = RadarModel(
        sensor_id="radar-1",
        position=np.zeros(2),
        noise_std=noise_std,
        radar_constant=_RADAR_CONSTANT,
        rng=_zero_noise_rng(),
    )
    measurement = radar.observe(_truth(position=[10.0, 0.0], velocity=[0.0, 0.0]))
    np.testing.assert_allclose(measurement.covariance, np.diag(noise_std**2))


def test_noise_is_applied_and_reproducible_given_same_seed() -> None:
    noise_std = np.array([1.0, 1.0, 0.1])
    truth = _truth(position=[10.0, 0.0], velocity=[1.0, 0.0])

    radar_a = RadarModel(
        sensor_id="radar-1",
        position=np.zeros(2),
        noise_std=noise_std,
        radar_constant=_RADAR_CONSTANT,
        rng=SeededRNG(seed=7).spawn("radar"),
    )
    radar_b = RadarModel(
        sensor_id="radar-1",
        position=np.zeros(2),
        noise_std=noise_std,
        radar_constant=_RADAR_CONSTANT,
        rng=SeededRNG(seed=7).spawn("radar"),
    )

    measurement_a = radar_a.observe(truth)
    measurement_b = radar_b.observe(truth)

    assert measurement_a.range == pytest.approx(measurement_b.range)
    assert measurement_a.range != pytest.approx(10.0)


def test_co_located_target_raises_value_error() -> None:
    radar = RadarModel(
        sensor_id="radar-1",
        position=np.array([5.0, 5.0]),
        noise_std=np.zeros(3),
        radar_constant=_RADAR_CONSTANT,
        rng=_zero_noise_rng(),
    )
    with pytest.raises(ValueError, match="co-located"):
        radar.observe(_truth(position=[5.0, 5.0], velocity=[0.0, 0.0]))


def test_non_2d_sensor_position_raises_value_error() -> None:
    with pytest.raises(ValueError, match="2D"):
        RadarModel(
            sensor_id="radar-1",
            position=np.zeros(3),
            noise_std=np.zeros(3),
            radar_constant=_RADAR_CONSTANT,
            rng=_zero_noise_rng(),
        )


def test_non_2d_target_position_raises_value_error() -> None:
    radar = RadarModel(
        sensor_id="radar-1",
        position=np.zeros(2),
        noise_std=np.zeros(3),
        radar_constant=_RADAR_CONSTANT,
        rng=_zero_noise_rng(),
    )
    truth = _truth(position=[1.0, 2.0, 3.0], velocity=[0.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="2D"):
        radar.observe(truth)


def test_mismatched_noise_std_shape_raises_value_error() -> None:
    with pytest.raises(ValueError, match="noise_std"):
        RadarModel(
            sensor_id="radar-1",
            position=np.zeros(2),
            noise_std=np.zeros(2),
            radar_constant=_RADAR_CONSTANT,
            rng=_zero_noise_rng(),
        )


def test_non_positive_rcs_raises_value_error() -> None:
    radar = RadarModel(
        sensor_id="radar-1",
        position=np.zeros(2),
        noise_std=np.zeros(3),
        radar_constant=_RADAR_CONSTANT,
        rng=_zero_noise_rng(),
    )
    truth = _truth(position=[10.0, 0.0], velocity=[0.0, 0.0], rcs=0.0)
    with pytest.raises(ValueError, match="rcs"):
        radar.observe(truth)


def test_snr_follows_radar_equation() -> None:
    radar_constant = 1_000_000.0
    rcs = 4.0
    radar = RadarModel(
        sensor_id="radar-1",
        position=np.zeros(2),
        noise_std=np.zeros(3),
        radar_constant=radar_constant,
        rng=_zero_noise_rng(),
    )
    truth = _truth(position=[10.0, 0.0], velocity=[0.0, 0.0], rcs=rcs)

    measurement = radar.observe(truth)

    expected_snr_db = 10.0 * np.log10(radar_constant * rcs / 10.0**4)
    assert measurement.snr == pytest.approx(expected_snr_db)


def test_calibrate_radar_constant_reproduces_reference_snr() -> None:
    reference_rcs = 5.0
    reference_range = 30.0
    reference_snr_db = 20.0
    radar_constant = calibrate_radar_constant(reference_rcs, reference_range, reference_snr_db)

    radar = RadarModel(
        sensor_id="radar-1",
        position=np.zeros(2),
        noise_std=np.zeros(3),
        radar_constant=radar_constant,
        rng=_zero_noise_rng(),
    )
    truth = _truth(position=[reference_range, 0.0], velocity=[0.0, 0.0], rcs=reference_rcs)

    measurement = radar.observe(truth)

    assert measurement.snr == pytest.approx(reference_snr_db)


_finite_vec2 = arrays(
    dtype=np.float64,
    shape=2,
    elements=st.floats(min_value=-1e3, max_value=1e3, allow_nan=False, allow_infinity=False),
)


@given(sensor_position=_finite_vec2, target_position=_finite_vec2)
def test_zero_noise_range_matches_analytic_norm(
    sensor_position: np.ndarray, target_position: np.ndarray
) -> None:
    assume(not np.allclose(sensor_position, target_position))

    radar = RadarModel(
        sensor_id="radar-1",
        position=sensor_position,
        noise_std=np.zeros(3),
        radar_constant=_RADAR_CONSTANT,
        rng=_zero_noise_rng(),
    )
    truth = _truth(position=list(target_position), velocity=[0.0, 0.0])

    measurement = radar.observe(truth)

    expected_range = float(np.linalg.norm(target_position - sensor_position))
    assert measurement.range == pytest.approx(expected_range)
