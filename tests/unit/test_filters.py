import numpy as np
import pytest
from hypothesis import assume, given
from hypothesis import strategies as st

from radarsim.signals.filters import moving_average


def test_moving_average_hand_computed() -> None:
    signal = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = moving_average(signal, window=3)
    np.testing.assert_allclose(result, [2.0, 3.0, 4.0])


def test_moving_average_window_one_is_noop() -> None:
    signal = np.array([1.0, 2.0, 3.0])
    result = moving_average(signal, window=1)
    np.testing.assert_allclose(result, signal)


def test_moving_average_output_length() -> None:
    signal = np.arange(10.0)
    result = moving_average(signal, window=4)
    assert len(result) == len(signal) - 4 + 1


def test_moving_average_window_too_small_raises() -> None:
    with pytest.raises(ValueError, match="window"):
        moving_average(np.array([1.0, 2.0]), window=0)


def test_moving_average_window_too_large_raises() -> None:
    with pytest.raises(ValueError, match="window"):
        moving_average(np.array([1.0, 2.0]), window=3)


@given(
    value=st.floats(
        min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False, allow_subnormal=False
    ),
    length=st.integers(min_value=1, max_value=50),
    window=st.integers(min_value=1, max_value=50),
)
def test_moving_average_constant_signal_is_invariant(
    value: float, length: int, window: int
) -> None:
    assume(window <= length)
    signal = np.full(length, value)
    result = moving_average(signal, window)
    np.testing.assert_allclose(result, value)
