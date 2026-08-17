import numpy as np
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from radarsim.targets.motion_models import ConstantAccelerationModel, ConstantVelocityModel


def test_constant_velocity_step_hand_computed() -> None:
    model = ConstantVelocityModel()
    position = np.array([10.0, 30.0])
    velocity = np.array([2.0, -1.0])
    acceleration = np.array([0.0, 0.0])

    new_position, new_velocity, new_acceleration = model.step(
        position, velocity, acceleration, dt=1.0
    )

    np.testing.assert_allclose(new_position, [12.0, 29.0])
    np.testing.assert_allclose(new_velocity, [2.0, -1.0])
    np.testing.assert_allclose(new_acceleration, [0.0, 0.0])


def test_constant_acceleration_step_hand_computed() -> None:
    model = ConstantAccelerationModel()
    position = np.array([0.0, 0.0])
    velocity = np.array([1.0, 0.0])
    acceleration = np.array([2.0, 0.0])

    new_position, new_velocity, new_acceleration = model.step(
        position, velocity, acceleration, dt=2.0
    )

    # x = 0 + 1*2 + 0.5*2*2^2 = 6; v = 1 + 2*2 = 5
    np.testing.assert_allclose(new_position, [6.0, 0.0])
    np.testing.assert_allclose(new_velocity, [5.0, 0.0])
    np.testing.assert_allclose(new_acceleration, [2.0, 0.0])


_finite_vec2 = arrays(
    dtype=np.float64,
    shape=2,
    elements=st.floats(min_value=-1e3, max_value=1e3, allow_nan=False, allow_infinity=False),
)


@given(position=_finite_vec2, velocity=_finite_vec2, dt=st.floats(min_value=0.0, max_value=10.0))
def test_constant_velocity_conserves_velocity(
    position: np.ndarray, velocity: np.ndarray, dt: float
) -> None:
    model = ConstantVelocityModel()
    acceleration = np.zeros(2)
    _, new_velocity, new_acceleration = model.step(position, velocity, acceleration, dt)
    np.testing.assert_allclose(new_velocity, velocity)
    np.testing.assert_allclose(new_acceleration, 0.0)


@given(
    velocity=_finite_vec2,
    acceleration=_finite_vec2,
    dt=st.floats(min_value=0.0, max_value=10.0),
)
def test_constant_acceleration_velocity_update(
    velocity: np.ndarray, acceleration: np.ndarray, dt: float
) -> None:
    model = ConstantAccelerationModel()
    position = np.zeros(2)
    _, new_velocity, new_acceleration = model.step(position, velocity, acceleration, dt)
    np.testing.assert_allclose(new_velocity, velocity + acceleration * dt)
    np.testing.assert_allclose(new_acceleration, acceleration)
