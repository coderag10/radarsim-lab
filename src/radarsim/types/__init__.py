"""Shared data contracts passed between pipeline stages.

See docs/ARCHITECTURE.md for the rule these types exist to enforce:
ground truth, measurements, detections, and track estimates are
distinct, non-interchangeable types.
"""

from radarsim.types.detection import Detection
from radarsim.types.measurement import RadarMeasurement
from radarsim.types.state import GroundTruth
from radarsim.types.track import TrackEstimate, TrackStatus

__all__ = [
    "Detection",
    "GroundTruth",
    "RadarMeasurement",
    "TrackEstimate",
    "TrackStatus",
]
