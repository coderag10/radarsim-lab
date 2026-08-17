from __future__ import annotations

import numpy as np


def moving_average(signal: np.ndarray, window: int) -> np.ndarray:
    """Smooth `signal` with a simple moving-average filter of size `window`.

    `mode="valid"` convolution: no fabricated edge padding, so the
    result is shorter than `signal` by `window - 1` samples.
    """
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")
    if window > signal.shape[-1]:
        raise ValueError(f"window ({window}) cannot exceed signal length ({signal.shape[-1]})")

    kernel = np.ones(window) / window
    return np.convolve(signal, kernel, mode="valid")
