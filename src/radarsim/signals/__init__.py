"""Sampling, noise, FFT, filtering, and range-Doppler representations."""

from radarsim.signals.fft import compute_spectrum, range_doppler_axes, range_doppler_map
from radarsim.signals.filters import moving_average
from radarsim.signals.noise import additive_white_gaussian_noise

__all__ = [
    "additive_white_gaussian_noise",
    "compute_spectrum",
    "moving_average",
    "range_doppler_axes",
    "range_doppler_map",
]
