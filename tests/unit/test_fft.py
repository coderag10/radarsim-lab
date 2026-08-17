import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from radarsim.signals.fft import compute_spectrum, range_doppler_axes, range_doppler_map


def _bin_aligned_time_axis(n: int, sample_rate: float) -> np.ndarray:
    return np.arange(n) / sample_rate


def test_compute_spectrum_real_sine_peaks_at_frequency() -> None:
    n = 1024
    sample_rate = 1024.0  # bin spacing = 1 Hz, keeps f0 exactly bin-aligned
    f0 = 50.0
    t = _bin_aligned_time_axis(n, sample_rate)
    signal = np.sin(2 * np.pi * f0 * t)

    frequencies, magnitude = compute_spectrum(signal, sample_rate)

    peak_freq = frequencies[np.argmax(magnitude)]
    assert abs(abs(peak_freq) - f0) < 1e-6


def test_compute_spectrum_complex_exponential_is_asymmetric() -> None:
    n = 1024
    sample_rate = 1024.0
    f0 = 50.0
    t = _bin_aligned_time_axis(n, sample_rate)
    signal = np.exp(2j * np.pi * f0 * t)

    frequencies, magnitude = compute_spectrum(signal, sample_rate)

    peak_index = np.argmax(magnitude)
    assert abs(frequencies[peak_index] - f0) < 1e-6

    negative_freq_index = np.argmin(np.abs(frequencies + f0))
    assert magnitude[negative_freq_index] < 1e-6 * magnitude[peak_index]


_finite_signal = arrays(
    dtype=np.float64,
    shape=st.integers(min_value=2, max_value=64),
    elements=st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False),
)


@given(signal=_finite_signal)
def test_parseval_theorem_holds(signal: np.ndarray) -> None:
    sample_rate = 100.0
    _, magnitude = compute_spectrum(signal, sample_rate)
    n = signal.shape[-1]

    time_energy = float(np.sum(np.abs(signal) ** 2))
    freq_energy = float(np.sum(magnitude**2) / n)

    assert time_energy == pytest.approx(freq_energy, rel=1e-6, abs=1e-9)


def test_range_doppler_map_locates_known_range_and_doppler_bin() -> None:
    num_pulses = 8
    num_fast_time = 16
    sample_rate = 16.0  # beat-frequency bin spacing = 1 Hz
    prf = 8.0  # doppler bin spacing = 1 Hz
    range_bin_freq = 3.0  # Hz, bin-aligned
    doppler_freq = 2.0  # Hz, bin-aligned

    n = np.arange(num_fast_time)
    pulse_index = np.arange(num_pulses)
    fast_time_tone = np.exp(2j * np.pi * range_bin_freq * n / sample_rate)
    doppler_phase = np.exp(2j * np.pi * doppler_freq * pulse_index / prf)
    samples = doppler_phase[:, None] * fast_time_tone[None, :]

    rd_map = range_doppler_map(samples, sample_rate, prf)
    peak_doppler_bin, peak_range_bin = np.unravel_index(np.argmax(rd_map), rd_map.shape)

    assert peak_range_bin == 3
    assert peak_doppler_bin == 6
    assert rd_map[peak_doppler_bin, peak_range_bin] > 100 * np.median(rd_map)


def test_range_doppler_map_requires_2d_input() -> None:
    with pytest.raises(ValueError, match="2D"):
        range_doppler_map(np.zeros(10), sample_rate=1.0, prf=1.0)


def test_range_doppler_axes_matches_hand_computed_bins() -> None:
    doppler_hz, beat_freq_hz = range_doppler_axes(
        num_pulses=8, num_fast_time_samples=16, sample_rate=16.0, prf=8.0
    )

    assert doppler_hz.shape == (8,)
    assert beat_freq_hz.shape == (16,)
    np.testing.assert_allclose(doppler_hz, [-4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0])
    np.testing.assert_allclose(beat_freq_hz[:4], [0.0, 1.0, 2.0, 3.0])
