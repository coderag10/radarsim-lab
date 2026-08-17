from __future__ import annotations

import math
from typing import Protocol

from radarsim.types import Detection, RadarMeasurement


class Detector(Protocol):
    """Common interface for detectors: measurements/signal -> candidate detections."""

    def detect(self, measurements: list[RadarMeasurement]) -> list[Detection]: ...


class ThresholdDetector:
    """Accepts measurements whose SNR clears a fixed threshold (simple CFAR-style detector).

    Confidence is a logistic curve centered at the threshold: exactly
    0.5 right at `snr_threshold_db`, approaching 1.0 as SNR climbs
    above it. Anything below the threshold is dropped outright rather
    than scored low -- this detector only ever filters real
    `RadarMeasurement`s down, it doesn't synthesize detections.
    """

    def __init__(self, snr_threshold_db: float, confidence_scale_db: float = 3.0) -> None:
        if confidence_scale_db <= 0.0:
            raise ValueError(f"confidence_scale_db must be > 0, got {confidence_scale_db}")
        self.snr_threshold_db = snr_threshold_db
        self.confidence_scale_db = confidence_scale_db

    def detect(self, measurements: list[RadarMeasurement]) -> list[Detection]:
        return [
            Detection(
                timestamp=measurement.timestamp,
                measurement=measurement,
                confidence=self._confidence(measurement.snr),
                snr=measurement.snr,
            )
            for measurement in measurements
            if measurement.snr >= self.snr_threshold_db
        ]

    def _confidence(self, snr_db: float) -> float:
        return 1.0 / (1.0 + math.exp(-(snr_db - self.snr_threshold_db) / self.confidence_scale_db))
