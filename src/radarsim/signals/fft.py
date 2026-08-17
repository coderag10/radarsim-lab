from __future__ import annotations

import numpy as np


def compute_spectrum(samples: np.ndarray, sample_rate: float) -> tuple[np.ndarray, np.ndarray]:
    """Return (frequencies, magnitude spectrum) of a sampled signal via FFT."""
    raise NotImplementedError


def range_doppler_map(samples: np.ndarray, sample_rate: float, prf: float) -> np.ndarray:
    """Compute a 2D range-Doppler map from a fast-time/slow-time sample matrix.

    `samples` is expected as (slow_time_pulses, fast_time_samples);
    `prf` is the pulse repetition frequency driving the Doppler axis.
    """
    raise NotImplementedError
