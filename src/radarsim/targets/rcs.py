from __future__ import annotations

from typing import Protocol


class RCSModel(Protocol):
    """Common interface for radar-cross-section models."""

    def rcs(self, aspect_angle: float) -> float:
        """Return radar cross-section (m^2) as seen from `aspect_angle` (radians)."""
        ...


class ConstantRCS:
    """A fixed RCS regardless of aspect angle -- the simplest useful model."""

    def __init__(self, value: float) -> None:
        raise NotImplementedError

    def rcs(self, aspect_angle: float) -> float:
        raise NotImplementedError
