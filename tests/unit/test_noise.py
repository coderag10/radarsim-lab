import numpy as np
import pytest

from radarsim.signals.noise import additive_white_gaussian_noise


def test_awgn_variance_matches_requested_snr_real_signal() -> None:
    rng = np.random.default_rng(42)
    n = 200_000
    signal = np.full(n, 2.0)  # power = 4.0
    snr_db = 10.0

    noisy = additive_white_gaussian_noise(signal, snr_db, rng)
    added_noise = noisy - signal

    expected_noise_power = 4.0 / (10 ** (snr_db / 10))
    measured_noise_power = float(np.var(added_noise))

    assert measured_noise_power == pytest.approx(expected_noise_power, rel=0.05)


def test_awgn_variance_matches_requested_snr_complex_signal() -> None:
    rng = np.random.default_rng(42)
    n = 200_000
    signal = np.full(n, 1.0 + 1.0j)  # power = |1+1j|^2 = 2.0
    snr_db = 6.0

    noisy = additive_white_gaussian_noise(signal, snr_db, rng)
    added_noise = noisy - signal

    expected_noise_power = 2.0 / (10 ** (snr_db / 10))
    measured_noise_power = float(np.mean(np.abs(added_noise) ** 2))

    assert measured_noise_power == pytest.approx(expected_noise_power, rel=0.05)


def test_awgn_reproducible_given_same_seed() -> None:
    signal = np.linspace(-1, 1, 100)
    noisy_a = additive_white_gaussian_noise(signal, 5.0, np.random.default_rng(7))
    noisy_b = additive_white_gaussian_noise(signal, 5.0, np.random.default_rng(7))
    np.testing.assert_array_equal(noisy_a, noisy_b)


def test_awgn_all_zero_signal_raises_value_error() -> None:
    with pytest.raises(ValueError, match="all-zero"):
        additive_white_gaussian_noise(np.zeros(10), 5.0, np.random.default_rng(0))
