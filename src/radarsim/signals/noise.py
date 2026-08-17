from __future__ import annotations

import numpy as np


def additive_white_gaussian_noise(
    signal: np.ndarray, snr_db: float, rng: np.random.Generator
) -> np.ndarray:
    """Add AWGN to `signal` at the given SNR (dB), drawn from `rng`.

    `snr_db = 10*log10(signal_power / noise_power)`, where
    `signal_power = mean(|signal|**2)`. For a complex `signal`, total
    noise power is split evenly between the real and imaginary parts
    (independent Gaussian I/Q noise), matching how radar baseband
    noise is normally modeled.
    """
    signal_power = float(np.mean(np.abs(signal) ** 2))
    if signal_power == 0.0:
        raise ValueError("cannot add noise at a finite SNR to an all-zero signal")

    noise_power = signal_power / (10.0 ** (snr_db / 10.0))

    if np.iscomplexobj(signal):
        component_std = np.sqrt(noise_power / 2.0)
        noise = rng.normal(0.0, component_std, size=signal.shape) + 1j * rng.normal(
            0.0, component_std, size=signal.shape
        )
    else:
        noise = rng.normal(0.0, np.sqrt(noise_power), size=signal.shape)

    return signal + noise
