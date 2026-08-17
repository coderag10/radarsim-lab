from __future__ import annotations

from dataclasses import dataclass

from radarsim.types.measurement import RadarMeasurement


@dataclass(frozen=True, slots=True, eq=False)
class Detection:
    """A measurement that has cleared the detector's threshold.

    Produced by `radarsim.detection` from a `RadarMeasurement` (or
    processed signal). Consumed by `radarsim.tracking`.

    `eq=False`: see `radarsim.types.state.GroundTruth` docstring.
    """

    timestamp: float
    measurement: RadarMeasurement
    confidence: float
    snr: float
