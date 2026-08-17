from __future__ import annotations

from typing import Protocol

import numpy as np


class MotionModel(Protocol):
    """Common interface for target kinematic models."""

    def step(
        self,
        position: np.ndarray,
        velocity: np.ndarray,
        acceleration: np.ndarray,
        dt: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Integrate one timestep forward, returning (position, velocity, acceleration)."""
        ...


class ConstantVelocityModel:
    """x_{t+1} = x_t + v_t * dt; velocity held constant, acceleration always zero."""

    def step(
        self,
        position: np.ndarray,
        velocity: np.ndarray,
        acceleration: np.ndarray,
        dt: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        raise NotImplementedError


class ConstantAccelerationModel:
    """x_{t+1} = x_t + v_t*dt + 0.5*a_t*dt^2;  v_{t+1} = v_t + a_t*dt."""

    def step(
        self,
        position: np.ndarray,
        velocity: np.ndarray,
        acceleration: np.ndarray,
        dt: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        raise NotImplementedError
