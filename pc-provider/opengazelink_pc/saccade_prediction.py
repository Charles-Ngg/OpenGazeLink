"""Causal continuation and stopping-distance prediction for observed saccades.

Coordinates are in local screen-plane degrees, timestamps in source
milliseconds. No stimulus, future samples, or personal fitted weights enter
these functions. Secant velocities are corrected to the current sample time
during braking. Inadequate temporal support returns None, not a guessed jump.
"""
import numpy as np


MAX_SAMPLE_INTERVAL_S = .020
MAX_CONTINUATION_S = .012


def short_continuation(times_ms, points, horizon_ms, noise):
    """Use measured direction for at most 12 ms, shorten it when braking."""
    times, points = np.asarray(times_ms, float), np.asarray(points, float)
    if len(times) < 3:
        return None
    steps = np.diff(points[-3:], axis=0)
    intervals = np.diff(times[-3:]) / 1000
    # At 30 Hz, two apparently coherent secants can span an entire short
    # saccade. They do not establish that motion is still in progress.
    if np.any(intervals <= 0) or np.max(intervals) > MAX_SAMPLE_INTERVAL_S:
        return None
    velocity = steps / intervals[:, None]
    speeds = np.linalg.norm(velocity, axis=1)
    if min(speeds) < 25 or np.dot(velocity[0], velocity[1]) < .7 * np.prod(speeds):
        return None
    duration = min(max(0., float(horizon_ms)) / 1000, MAX_CONTINUATION_S, float(intervals.mean()))
    current_speed = speeds[1]
    if speeds[1] < speeds[0]:
        # Constant-deceleration stopping distance; never extrapolate through
        # the velocity zero crossing or continue a pre-landing peak velocity.
        deceleration = (speeds[0] - speeds[1]) / max(.001, intervals.mean())
        # Secant velocity belongs to the middle of the previous exposure
        # interval. Project it to the current time before computing a stop.
        current_speed = max(0., speeds[1] - deceleration * intervals[-1] / 2)
        stop = current_speed / max(1e-9, deceleration)
        duration = min(duration, stop)
        duration *= max(0., 1 - .5 * duration / max(.001, stop))
    if np.linalg.norm(steps[-1]) < noise * 1.5:
        return None
    delta = velocity[-1] * (current_speed / max(1e-9, speeds[1])) * duration
    return delta if np.linalg.norm(delta) > .001 else None


def braking_prediction(times_ms, points, horizon_ms, noise):
    """Estimate a stop only after two consistent observed decelerations."""
    times, points = np.asarray(times_ms, float), np.asarray(points, float)
    if len(times) < 4:
        return None
    dt = np.diff(times[-4:]) / 1000
    if np.any(dt <= 0) or np.max(dt) > MAX_SAMPLE_INTERVAL_S:
        return None
    velocity = np.diff(points[-4:], axis=0) / dt[:, None]
    speeds = np.linalg.norm(velocity, axis=1)
    if min(speeds) < 25:
        return None
    direction = velocity[-1] / speeds[-1]
    along = velocity @ direction
    if np.any(along < .9 * speeds):
        return None
    accelerations = np.diff(along) / ((dt[:-1] + dt[1:]) / 2)
    if np.any(accelerations >= 0):
        return None
    decelerations = -accelerations
    if decelerations.min() < .35 * decelerations.max():
        return None
    if along[0] - along[-1] < noise * 2 / max(.001, dt.mean()):
        return None
    deceleration = float(np.mean(decelerations))
    current_speed = max(0., along[-1] - deceleration * dt[-1] / 2)
    remaining = current_speed / max(1e-9, deceleration)
    if remaining > .05:
        return None
    t = min(max(0., float(horizon_ms)) / 1000, remaining)
    distance = current_speed * t - .5 * deceleration * t ** 2
    if distance <= .001:
        return None
    endpoint = points[-1] + direction * .5 * current_speed * remaining
    return direction * distance, endpoint, dict(remaining_ms=remaining * 1000, trusted_horizon_ms=t * 1000,
        deceleration_deg_s2=deceleration, current_speed_deg_s=current_speed)
