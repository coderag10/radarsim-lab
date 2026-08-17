from __future__ import annotations

from typing import Protocol

from radarsim.types import Detection, RadarMeasurement


class Detector(Protocol):
    """Common interface for detectors: measurements/signal -> candidate detections."""

    def detect(self, measurements: list[RadarMeasurement]) -> list[Detection]: ...


class ThresholdDetector:
    """Accepts measurements whose SNR clears a fixed threshold (simple CFAR-style detector)."""

    def __init__(self, snr_threshold_db: float) -> None:
        raise NotImplementedError

    def detect(self, measurements: list[RadarMeasurement]) -> list[Detection]:
        raise NotImplementedError
