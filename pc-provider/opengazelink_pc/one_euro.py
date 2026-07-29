from __future__ import annotations

import math
import time

import numpy as np


class OneEuroFilter2D:
    """Low-latency adaptive low-pass filter for a normalized 2D gaze point."""

    def __init__(
        self,
        min_cutoff: float = 1.0,
        beta: float = 2.0,
        derivative_cutoff: float = 1.0,
    ) -> None:
        self.configure(min_cutoff, beta, derivative_cutoff)
        self._raw_value: np.ndarray | None = None
        self._filtered_value: np.ndarray | None = None
        self._derivative: np.ndarray | None = None
        self._timestamp: float | None = None

    @staticmethod
    def _alpha(dt: float, cutoff: float) -> float:
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def configure(self, min_cutoff: float, beta: float, derivative_cutoff: float) -> None:
        self.min_cutoff = max(0.01, float(min_cutoff))
        self.beta = max(0.0, float(beta))
        self.derivative_cutoff = max(0.01, float(derivative_cutoff))

    def reset(self) -> None:
        self._raw_value = None
        self._filtered_value = None
        self._derivative = None
        self._timestamp = None

    def update(self, point: tuple[float, float], timestamp: float | None = None) -> tuple[float, float]:
        value = np.asarray(point, dtype=np.float64)
        if value.shape != (2,) or not np.isfinite(value).all():
            raise ValueError("One Euro filter requires a finite 2D point")
        now = time.monotonic() if timestamp is None else float(timestamp)
        if self._raw_value is None or self._filtered_value is None or self._timestamp is None:
            self._raw_value = value.copy()
            self._filtered_value = value.copy()
            self._derivative = np.zeros(2, dtype=np.float64)
            self._timestamp = now
            return float(value[0]), float(value[1])

        dt = now - self._timestamp
        if not math.isfinite(dt) or dt <= 1e-6:
            dt = 1.0 / 30.0
        derivative = (value - self._raw_value) / dt
        derivative_alpha = self._alpha(dt, self.derivative_cutoff)
        assert self._derivative is not None
        derivative_hat = derivative_alpha * derivative + (1.0 - derivative_alpha) * self._derivative
        cutoff = self.min_cutoff + self.beta * np.abs(derivative_hat)
        value_alpha = 1.0 / (1.0 + 1.0 / (2.0 * math.pi * cutoff * dt))
        filtered = value_alpha * value + (1.0 - value_alpha) * self._filtered_value
        self._raw_value = value
        self._filtered_value = filtered
        self._derivative = derivative_hat
        self._timestamp = now
        return float(filtered[0]), float(filtered[1])
