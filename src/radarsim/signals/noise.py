from __future__ import annotations

import numpy as np


def additive_white_gaussian_noise(
    signal: np.ndarray, snr_db: float, rng: np.random.Generator
) -> np.ndarray:
    """Add AWGN to `signal` at the given SNR (dB), drawn from `rng`."""
    raise NotImplementedError
