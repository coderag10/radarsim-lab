from __future__ import annotations

import numpy as np


def compute_spectrum(samples: np.ndarray, sample_rate: float) -> tuple[np.ndarray, np.ndarray]:
    """Return (frequencies, magnitude spectrum) of a sampled signal via FFT.

    Two-sided, frequency-centered (`fftshift`) spectrum -- correct for
    complex baseband/I-Q signals (asymmetric spectrum, the physically
    interesting case for radar) and equally valid for real-valued
    signals (symmetric spectrum). Operates along the last axis, so a
    batch of signals (shape `(..., n_samples)`) works for free.
    """
    n = samples.shape[-1]
    spectrum = np.fft.fftshift(np.fft.fft(samples, axis=-1), axes=-1)
    frequencies = np.fft.fftshift(np.fft.fftfreq(n, d=1.0 / sample_rate))
    magnitude = np.abs(spectrum)
    return frequencies, magnitude


def range_doppler_map(samples: np.ndarray, sample_rate: float, prf: float) -> np.ndarray:
    """Compute a 2D range-Doppler map from a fast-time/slow-time sample matrix.

    `samples` is `(num_pulses, num_fast_time_samples)` -- one pulse's
    fast-time samples per row. FFT along fast time (axis=1) resolves
    range; FFT along slow time (axis=0, across pulses, per range bin)
    resolves Doppler, `fftshift`ed so zero velocity is centered.
    Returns the `(num_pulses, num_fast_time_samples)` magnitude map.

    `sample_rate`/`prf` don't scale this array -- see
    `range_doppler_axes` to convert bin index to physical frequency.
    """
    if samples.ndim != 2:
        raise ValueError(
            f"samples must be a 2D (pulses, fast_time) matrix, got shape {samples.shape}"
        )

    range_compressed = np.fft.fft(samples, axis=1)
    doppler = np.fft.fftshift(np.fft.fft(range_compressed, axis=0), axes=0)
    return np.abs(doppler)


def range_doppler_axes(
    num_pulses: int, num_fast_time_samples: int, sample_rate: float, prf: float
) -> tuple[np.ndarray, np.ndarray]:
    """Return (doppler_hz, beat_freq_hz) bin-center frequencies for `range_doppler_map`.

    `doppler_hz` is a complete Doppler-frequency axis (velocity is one
    multiplication away: `velocity = doppler_hz * wavelength / 2`,
    wavelength not modeled here). `beat_freq_hz` stops at frequency,
    not range in meters -- that conversion needs an FMCW chirp
    bandwidth/slope this codebase doesn't model yet, so it isn't faked
    here.
    """
    doppler_hz = np.fft.fftshift(np.fft.fftfreq(num_pulses, d=1.0 / prf))
    beat_freq_hz = np.fft.fftfreq(num_fast_time_samples, d=1.0 / sample_rate)
    return doppler_hz, beat_freq_hz
