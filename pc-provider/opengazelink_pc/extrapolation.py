from __future__ import annotations

from collections import deque
import math

import numpy as np


class FixedHorizonExtrapolator2D:
    """State-aware fixed-delay predictor tuned for roughly 30 FPS input."""

    def __init__(
        self, horizon_ms: float = 85.0, max_lead_fraction: float = 0.12,
        history_ms: float = 150.0,
    ) -> None:
        self.history_ms = float(history_ms)
        self._samples: deque[tuple[float, np.ndarray]] = deque(maxlen=6)
        self._velocity = np.zeros(2, dtype=np.float64)
        self._instant_velocity = np.zeros(2, dtype=np.float64)
        self._fixation_center: np.ndarray | None = None
        self._state = "fixation"
        self._motion_evidence = 0
        self.configure(horizon_ms, max_lead_fraction)

    def configure(self, horizon_ms: float, max_lead_fraction: float) -> None:
        self.horizon_ms = max(0.0, min(300.0, float(horizon_ms)))
        self.max_lead_fraction = max(0.0, min(0.5, float(max_lead_fraction)))

    def reset(self) -> None:
        self._samples.clear()
        self._velocity[:] = 0.0
        self._instant_velocity[:] = 0.0
        self._fixation_center = None
        self._state = "fixation"
        self._motion_evidence = 0

    def update(
        self,
        raw_point: tuple[float, float],
        stable_point: tuple[float, float],
        source_timestamp_ms: float,
        screen_size: tuple[int, int],
    ) -> tuple[tuple[float, float], dict]:
        raw = np.asarray(raw_point, dtype=np.float64)
        stable = np.asarray(stable_point, dtype=np.float64)
        timestamp = float(source_timestamp_ms)
        if not np.isfinite(raw).all() or not np.isfinite(stable).all() or not math.isfinite(timestamp):
            raise ValueError("gaze extrapolation requires finite points and timestamp")

        if self._samples:
            elapsed = timestamp - self._samples[-1][0]
            if elapsed <= 0.0 or elapsed > 250.0:
                self.reset()

        previous_state = self._state
        previous_instant = self._instant_velocity.copy()
        previous_speed = float(np.linalg.norm(previous_instant))
        self._samples.append((timestamp, raw))
        cutoff = timestamp - self.history_ms
        while len(self._samples) > 2 and self._samples[0][0] < cutoff:
            self._samples.popleft()

        diagonal = math.hypot(max(1, int(screen_size[0])), max(1, int(screen_size[1])))
        fixation_radius = max(4.0, diagonal * 0.004)
        jump_distance = max(4.0 * fixation_radius, diagonal * 0.03)
        direction_consistency = 1.0
        step_distance = 0.0
        dt_ms = 0.0
        if len(self._samples) >= 2:
            previous_timestamp, previous_raw = self._samples[-2]
            dt_ms = max(1.0, timestamp - previous_timestamp)
            delta = raw - previous_raw
            step_distance = float(np.linalg.norm(delta))
            self._instant_velocity = delta / dt_ms
            current_speed = float(np.linalg.norm(self._instant_velocity))
            if previous_speed > 1e-9 and current_speed > 1e-9:
                direction_consistency = float(
                    np.dot(previous_instant, self._instant_velocity)
                    / (previous_speed * current_speed)
                )
        else:
            self._instant_velocity[:] = 0.0
            current_speed = 0.0

        points = np.asarray([point for _, point in self._samples], dtype=np.float64)
        centre = np.median(points, axis=0)
        spatial_span = float(np.max(np.linalg.norm(points - centre, axis=1)))
        if self._fixation_center is None:
            self._fixation_center = raw.copy()
        distance_from_fixation = float(np.linalg.norm(raw - self._fixation_center))

        is_quiet = step_distance <= fixation_radius
        is_jump = step_distance >= jump_distance
        is_reversal = direction_consistency < -0.15 and previous_speed > 1e-6
        is_braking = (
            previous_state in {"continuous_motion", "jump"}
            and previous_speed > 1e-6
            and (is_quiet or is_reversal or current_speed < 0.45 * previous_speed)
        )

        reset_filter = False
        phase = "steady"
        if len(self._samples) < 2:
            self._state = "fixation"
        elif is_jump:
            # A large reversal is a new saccade, not braking from the previous motion.
            self._state = "jump"
            self._motion_evidence = 0
            phase = "rising"
        elif is_braking:
            self._state = "landing"
            self._motion_evidence = 0
            reset_filter = True
            phase = "landing"
        elif previous_state == "jump":
            if direction_consistency >= 0.35 and not is_quiet:
                self._state = "jump"
                phase = "rising"
            else:
                self._state = "landing"
                reset_filter = True
                phase = "landing"
        elif previous_state == "landing":
            if is_quiet:
                self._state = "fixation"
                self._fixation_center = raw.copy()
                phase = "settled"
            else:
                self._state = "continuous_motion"
                phase = "tracking"
        else:
            moving_from_fixation = distance_from_fixation > 1.5 * fixation_radius
            moving_consistently = direction_consistency >= 0.25 and not is_quiet
            self._motion_evidence = (
                min(2, self._motion_evidence + 1)
                if moving_from_fixation or moving_consistently
                else max(0, self._motion_evidence - 1)
            )
            if previous_state == "continuous_motion" or self._motion_evidence >= 2:
                self._state = "continuous_motion"
                phase = "tracking"
            else:
                self._state = "fixation"

        if self._state == "fixation":
            self._velocity *= 0.35
            if spatial_span <= fixation_radius:
                self._fixation_center = 0.9 * self._fixation_center + 0.1 * raw
        elif self._state == "continuous_motion":
            measured = self._instant_velocity
            if previous_speed > 1e-9 and direction_consistency >= 0.25:
                measured = 0.72 * self._instant_velocity + 0.28 * previous_instant
            self._velocity = 0.72 * measured + 0.28 * self._velocity
        elif self._state == "jump":
            self._velocity = self._instant_velocity.copy()
        else:
            self._velocity[:] = 0.0

        filter_bypassed = self._state in {"jump", "landing"}
        base = raw.copy() if filter_bypassed else stable.copy()
        lead = np.zeros(2, dtype=np.float64)
        if self._state == "continuous_motion":
            lead = self._velocity * self.horizon_ms
        elif self._state == "jump":
            # Experimental short jump lead. Earlier offline proxy scores did
            # not establish useful compensation in the user's live testing.
            jump_gain_ms = min(20.0, 0.25 * self.horizon_ms)
            lead = self._instant_velocity * jump_gain_ms

        lead_length = float(np.linalg.norm(lead))
        maximum = self.max_lead_fraction * diagonal
        if lead_length < 2.0:
            lead[:] = 0.0
            lead_length = 0.0
        elif maximum >= 0.0 and lead_length > maximum:
            lead *= maximum / lead_length if lead_length else 0.0
            lead_length = maximum

        unclamped = base + lead
        predicted = np.clip(
            unclamped,
            np.zeros(2, dtype=np.float64),
            np.asarray([
                max(0, int(screen_size[0]) - 1),
                max(0, int(screen_size[1]) - 1),
            ], dtype=np.float64),
        )
        was_clamped = not np.allclose(predicted, unclamped, atol=1e-9)
        mode = {
            "fixation": "fixation",
            "continuous_motion": "continuous_motion",
            "jump": "jump_or_landing",
            "landing": "jump_or_landing",
        }[self._state]
        return (float(predicted[0]), float(predicted[1])), {
            "horizon_ms": self.horizon_ms,
            "velocity_px_per_ms": self._velocity.tolist(),
            "instant_velocity_px_per_ms": self._instant_velocity.tolist(),
            "lead_px": lead.tolist(),
            "lead_distance_px": lead_length,
            "sample_count": len(self._samples),
            "mode": mode,
            "phase": phase,
            "spatial_span_px": spatial_span,
            "step_distance_px": step_distance,
            "distance_from_fixation_px": distance_from_fixation,
            "fixation_radius_px": fixation_radius,
            "jump_distance_px": jump_distance,
            "direction_consistency": direction_consistency,
            "filter_bypassed": filter_bypassed,
            "reset_filter": reset_filter,
            "screen_clamped": was_clamped,
        }
