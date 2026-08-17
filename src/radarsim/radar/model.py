from __future__ import annotations

import numpy as np

from radarsim.types import GroundTruth, RadarMeasurement


class RadarModel:
    """Converts true target state into a noisy `RadarMeasurement`.

    true state -> radar model -> + noise -> measurement

    Must never hand back or retain the `GroundTruth` it observes;
    downstream stages only ever see the `RadarMeasurement` it produces.
    """

    def __init__(self, sensor_id: str, noise_std: np.ndarray, rng: np.random.Generator) -> None:
        raise NotImplementedError

    def observe(self, truth: GroundTruth) -> RadarMeasurement:
        """Produce a synthetic noisy measurement of `truth`."""
        raise NotImplementedError
