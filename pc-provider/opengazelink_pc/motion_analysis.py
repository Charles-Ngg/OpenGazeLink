from __future__ import annotations

import json
import math
from pathlib import Path
import numpy as np

from .extrapolation import FixedHorizonExtrapolator2D


def _point(record: dict, key: str) -> np.ndarray | None:
    value = np.asarray(record.get(key), dtype=np.float64)
    return value if value.shape == (2,) and np.isfinite(value).all() else None


def load_motion_recording(path: Path) -> tuple[dict, list[dict]]:
    metadata: dict = {}
    frames: list[dict] = []
    with Path(path).open("r", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("type") == "session":
                metadata = dict(record.get("metadata") or {})
            elif record.get("type") == "frame" and record.get("valid"):
                raw = _point(record, "raw_combined_px")
                if raw is not None and int(record.get("phone_sensor_time_ns") or 0) > 0:
                    frames.append(record)
    frames.sort(key=lambda item: int(item["phone_sensor_time_ns"]))
    deduplicated: list[dict] = []
    for frame in frames:
        if deduplicated and frame["phone_sensor_time_ns"] <= deduplicated[-1]["phone_sensor_time_ns"]:
            continue
        deduplicated.append(frame)
    return metadata, deduplicated


def _cap_lead(lead: np.ndarray, maximum: float) -> np.ndarray:
    length = float(np.linalg.norm(lead))
    if length > maximum > 0.0:
        return lead * (maximum / length)
    return lead


def _future_targets(times: np.ndarray, points: np.ndarray, horizon_ms: float) -> np.ndarray:
    result = np.full_like(points, np.nan)
    for index, timestamp in enumerate(times):
        target_time = timestamp + horizon_ms
        right = int(np.searchsorted(times, target_time, side="left"))
        if right <= index or right >= len(times):
            continue
        left = right - 1
        gap = times[right] - times[left]
        if gap <= 0.0 or gap > 100.0:
            continue
        fraction = (target_time - times[left]) / gap
        result[index] = points[left] + fraction * (points[right] - points[left])
    return result


def _summary(errors: np.ndarray, mask: np.ndarray) -> dict:
    values = errors[mask & np.isfinite(errors)]
    if not len(values):
        return {"count": 0}
    return {
        "count": int(len(values)),
        "mean_px": float(np.mean(values)),
        "median_px": float(np.median(values)),
        "p90_px": float(np.percentile(values, 90.0)),
        "p95_px": float(np.percentile(values, 95.0)),
        "max_px": float(np.max(values)),
    }


def _vector_cosine(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    result = np.full(len(left), np.nan, dtype=np.float64)
    lengths = np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)
    valid = (
        np.isfinite(left).all(axis=1)
        & np.isfinite(right).all(axis=1)
        & (lengths > 1e-9)
    )
    result[valid] = np.sum(left[valid] * right[valid], axis=1) / lengths[valid]
    return np.clip(result, -1.0, 1.0)


def _distribution(values: np.ndarray, mask: np.ndarray) -> dict:
    selected = values[mask & np.isfinite(values)]
    if not len(selected):
        return {"count": 0}
    return {
        "count": int(len(selected)),
        "mean": float(np.mean(selected)),
        "median": float(np.median(selected)),
        "p10": float(np.percentile(selected, 10.0)),
        "p90": float(np.percentile(selected, 90.0)),
    }


def _frame_vectors(values: np.ndarray, times: np.ndarray) -> np.ndarray:
    result = np.full_like(values, np.nan, dtype=np.float64)
    for index in range(1, len(values)):
        dt = times[index] - times[index - 1]
        if 1.0 <= dt <= 100.0 and np.isfinite(values[index - 1:index + 1]).all():
            result[index] = (values[index] - values[index - 1]) / dt
    return result


def _state_gated_prediction(
    raw: np.ndarray,
    velocity: np.ndarray,
    rising_mask: np.ndarray,
    maximum_lead: float,
    *,
    gain_ms: float | None = None,
    horizon_ms: float = 80.0,
    damping_tau_ms: float | None = None,
    confidence: np.ndarray | None = None,
) -> np.ndarray:
    prediction = raw.copy()
    if damping_tau_ms is not None:
        gain = damping_tau_ms * (1.0 - math.exp(-horizon_ms / damping_tau_ms))
    elif gain_ms is not None:
        gain = gain_ms
    else:
        raise ValueError("state-gated prediction requires a gain or damping tau")
    weights = np.ones(len(raw), dtype=np.float64) if confidence is None else confidence
    for index in np.flatnonzero(rising_mask & np.isfinite(velocity).all(axis=1)):
        lead = _cap_lead(velocity[index] * gain * weights[index], maximum_lead)
        prediction[index] = raw[index] + lead
    return prediction


def _saccade_episodes(
    times: np.ndarray,
    raw: np.ndarray,
    phases: np.ndarray,
    velocity: np.ndarray,
    maximum_lead: float,
) -> tuple[list[dict], dict[str, np.ndarray], np.ndarray]:
    episode_ids = np.full(len(raw), -1, dtype=np.int64)
    endpoint_targets = np.full_like(raw, np.nan)
    episodes: list[dict] = []
    index = 0
    while index < len(raw):
        if phases[index] != "rising":
            index += 1
            continue
        start = index
        while index + 1 < len(raw) and phases[index + 1] == "rising":
            index += 1
        end = index
        search_end = min(len(raw), end + 9)
        settled = next(
            (candidate for candidate in range(end + 1, search_end) if phases[candidate] == "settled"),
            None,
        )
        if settled is None:
            index += 1
            continue
        endpoint_end = settled + 1
        while (
            endpoint_end < min(len(raw), settled + 4)
            and phases[endpoint_end] not in {"rising", "tracking"}
        ):
            endpoint_end += 1
        endpoint = np.median(raw[settled:endpoint_end], axis=0)
        episode_id = len(episodes)
        episode_ids[start:end + 1] = episode_id
        endpoint_targets[start:end + 1] = endpoint
        first_step = raw[start] - raw[start - 1] if start > 0 else np.zeros(2, dtype=np.float64)
        remaining = endpoint - raw[start]
        speed_squared = float(np.dot(velocity[start], velocity[start]))
        effective_gain = (
            float(np.dot(remaining, velocity[start]) / speed_squared)
            if speed_squared > 1e-9 else float("nan")
        )
        episodes.append({
            "id": episode_id,
            "start_index": int(start),
            "end_index": int(end),
            "settled_index": int(settled),
            "rising_frame_count": int(end - start + 1),
            "duration_to_settle_ms": float(times[settled] - times[start]),
            "first_step_px": first_step.tolist(),
            "first_step_distance_px": float(np.linalg.norm(first_step)),
            "remaining_to_endpoint_px": remaining.tolist(),
            "remaining_distance_px": float(np.linalg.norm(remaining)),
            "first_velocity_effective_gain_ms": (
                effective_gain if math.isfinite(effective_gain) else None
            ),
            "endpoint_px": endpoint.tolist(),
        })
        index += 1

    endpoint_predictions: dict[str, np.ndarray] = {"raw_hold": raw.copy()}
    for gain_ms in (16.0, 20.0, 25.0, 30.0):
        endpoint_predictions[f"gain{int(gain_ms)}ms"] = _state_gated_prediction(
            raw,
            velocity,
            episode_ids >= 0,
            maximum_lead,
            gain_ms=gain_ms,
        )
    return episodes, endpoint_predictions, endpoint_targets


def _ridge_predictions(
    times: np.ndarray,
    raw: np.ndarray,
    targets: np.ndarray,
    train_mask: np.ndarray,
    maximum_lead: float,
) -> tuple[np.ndarray, dict]:
    count = len(raw)
    features = np.full((count, 6), np.nan, dtype=np.float64)
    for index in range(2, count):
        dt1 = max(1.0, times[index] - times[index - 1])
        dt0 = max(1.0, times[index - 1] - times[index - 2])
        velocity1 = (raw[index] - raw[index - 1]) / dt1
        velocity0 = (raw[index - 1] - raw[index - 2]) / dt0
        acceleration = (velocity1 - velocity0) / max(1.0, 0.5 * (dt1 + dt0))
        features[index] = [
            velocity1[0], velocity1[1], acceleration[0], acceleration[1],
            float(np.linalg.norm(velocity1)), 1.0,
        ]
    usable = train_mask & np.isfinite(features).all(axis=1) & np.isfinite(targets).all(axis=1)
    if int(np.sum(usable)) < 12:
        return np.full_like(raw, np.nan), {"trained": False, "training_samples": int(np.sum(usable))}
    x = features[usable]
    y = targets[usable] - raw[usable]
    scale = np.std(x[:, :-1], axis=0)
    scale[scale < 1e-9] = 1.0
    x_scaled = x.copy()
    x_scaled[:, :-1] /= scale
    regularizer = np.eye(x_scaled.shape[1], dtype=np.float64) * 1e-2
    regularizer[-1, -1] = 0.0
    weights = np.linalg.solve(x_scaled.T @ x_scaled + regularizer, x_scaled.T @ y)
    predictions = np.full_like(raw, np.nan)
    valid = np.isfinite(features).all(axis=1)
    projected = features[valid].copy()
    projected[:, :-1] /= scale
    leads = projected @ weights
    predictions[valid] = raw[valid] + np.asarray([
        _cap_lead(lead, maximum_lead) for lead in leads
    ])
    return predictions, {
        "trained": True,
        "training_samples": int(np.sum(usable)),
        "feature_scale": scale.tolist(),
        "weights": weights.tolist(),
    }


def analyze_motion_recording(path: Path, horizon_ms: float = 80.0) -> dict:
    metadata, frames = load_motion_recording(path)
    if len(frames) < 8:
        raise ValueError("motion recording needs at least 8 valid frames")
    times = np.asarray([int(item["phone_sensor_time_ns"]) / 1_000_000.0 for item in frames])
    raw = np.asarray([_point(item, "raw_combined_px") for item in frames], dtype=np.float64)
    filtered = np.asarray([
        _point(item, "filtered_combined_px") if _point(item, "filtered_combined_px") is not None else raw[index]
        for index, item in enumerate(frames)
    ], dtype=np.float64)
    recorded = np.asarray([
        _point(item, "output_combined_px") if _point(item, "output_combined_px") is not None else raw[index]
        for index, item in enumerate(frames)
    ], dtype=np.float64)
    screen = metadata.get("screen") or {}
    diagonal = math.hypot(float(screen.get("width") or 1920), float(screen.get("height") or 1080))
    extrapolation_config = metadata.get("extrapolation") or {}
    maximum_lead_fraction = float(extrapolation_config.get("max_lead_fraction") or 0.12)
    maximum_lead = maximum_lead_fraction * diagonal
    targets = _future_targets(times, raw, horizon_ms)
    target_valid = np.isfinite(targets).all(axis=1)
    steps = np.zeros(len(raw), dtype=np.float64)
    steps[1:] = np.linalg.norm(raw[1:] - raw[:-1], axis=1)
    states = [((item.get("postprocess") or {}).get("extrapolation_state") or {}) for item in frames]
    modes = np.asarray([
        str(state.get("mode") or "") for state in states
    ])
    phases = np.asarray([
        str(state.get("phase") or "") for state in states
    ])
    motion_mask = target_valid & ((steps >= 0.01 * diagonal) | (modes == "continuous_motion") | (modes == "jump_or_landing"))
    saccade_mask = target_valid & ((steps >= 0.03 * diagonal) | (modes == "jump_or_landing"))
    rising_mask = target_valid & (phases == "rising")
    landing_mask = target_valid & (phases == "landing")
    first_rising_mask = rising_mask & (np.roll(phases, 1) != "rising")
    first_rising_mask[0] = rising_mask[0]
    continuing_rising_mask = rising_mask & ~first_rising_mask
    split_time = times[0] + 0.6 * (times[-1] - times[0])
    train_mask = saccade_mask & (times <= split_time)
    holdout_mask = target_valid & (times > split_time)

    predictions: dict[str, np.ndarray] = {
        "raw_hold": raw.copy(),
        "filtered_hold": filtered.copy(),
        "recorded_output": recorded.copy(),
    }
    replayed_runtime = np.full_like(raw, np.nan)
    replayed_phases: list[str] = []
    replayed_predictor = FixedHorizonExtrapolator2D(
        float(extrapolation_config.get("horizon_ms") or horizon_ms),
        maximum_lead_fraction,
    )
    screen_size = (
        int(screen.get("width") or 1920),
        int(screen.get("height") or 1080),
    )
    for index in range(len(raw)):
        point, state = replayed_predictor.update(
            tuple(raw[index]), tuple(filtered[index]), float(times[index]), screen_size,
        )
        replayed_runtime[index] = point
        replayed_phases.append(str(state.get("phase") or ""))
    predictions["replayed_runtime_candidate"] = replayed_runtime
    replayed_phases_array = np.asarray(replayed_phases)
    velocity2 = np.full_like(raw, np.nan)
    regression3 = np.full_like(raw, np.nan)
    acceleration3 = np.full_like(raw, np.nan)
    damped20 = np.full_like(raw, np.nan)
    damped35 = np.full_like(raw, np.nan)
    damped50 = np.full_like(raw, np.nan)
    for index in range(1, len(raw)):
        dt = times[index] - times[index - 1]
        if not 1.0 <= dt <= 100.0:
            continue
        velocity = (raw[index] - raw[index - 1]) / dt
        velocity2[index] = raw[index] + _cap_lead(velocity * horizon_ms, maximum_lead)
        for tau, destination in ((20.0, damped20), (35.0, damped35), (50.0, damped50)):
            lead = velocity * tau * (1.0 - math.exp(-horizon_ms / tau))
            destination[index] = raw[index] + _cap_lead(lead, maximum_lead)
        if index < 2:
            continue
        local_times = times[index - 2:index + 1] - times[index]
        if np.max(np.diff(times[index - 2:index + 1])) > 100.0:
            continue
        design = np.column_stack((local_times, np.ones(3, dtype=np.float64)))
        slope = np.linalg.lstsq(design, raw[index - 2:index + 1], rcond=None)[0][0]
        regression3[index] = raw[index] + _cap_lead(slope * horizon_ms, maximum_lead)
        previous_dt = times[index - 1] - times[index - 2]
        previous_velocity = (raw[index - 1] - raw[index - 2]) / max(1.0, previous_dt)
        acceleration = (velocity - previous_velocity) / max(1.0, 0.5 * (dt + previous_dt))
        lead = velocity * horizon_ms + 0.5 * acceleration * horizon_ms * horizon_ms
        acceleration3[index] = raw[index] + _cap_lead(lead, maximum_lead)
    predictions.update({
        "velocity_2point": velocity2,
        "linear_regression_3point": regression3,
        "constant_acceleration_3point": acceleration3,
        "damped_velocity_tau20": damped20,
        "damped_velocity_tau35": damped35,
        "damped_velocity_tau50": damped50,
    })
    instant_velocity = _frame_vectors(raw, times)
    for gain_ms in (16.0, 20.0, 25.0, 30.0):
        predictions[f"rising_only_gain{int(gain_ms)}ms"] = _state_gated_prediction(
            raw,
            instant_velocity,
            rising_mask,
            maximum_lead,
            gain_ms=gain_ms,
            horizon_ms=horizon_ms,
        )
    for tau_ms in (20.0, 35.0):
        predictions[f"rising_only_damped_tau{int(tau_ms)}"] = _state_gated_prediction(
            raw,
            instant_velocity,
            rising_mask,
            maximum_lead,
            damping_tau_ms=tau_ms,
            horizon_ms=horizon_ms,
        )
    rising_30_then_16 = _state_gated_prediction(
        raw,
        instant_velocity,
        first_rising_mask,
        maximum_lead,
        gain_ms=30.0,
        horizon_ms=horizon_ms,
    )
    continuation = _state_gated_prediction(
        raw,
        instant_velocity,
        continuing_rising_mask,
        maximum_lead,
        gain_ms=16.0,
        horizon_ms=horizon_ms,
    )
    rising_30_then_16[continuing_rising_mask] = continuation[continuing_rising_mask]
    predictions["rising_gain30_then16ms"] = rising_30_then_16

    right_points = np.asarray([
        _point(item, "right_px") if _point(item, "right_px") is not None else [np.nan, np.nan]
        for item in frames
    ], dtype=np.float64)
    left_points = np.asarray([
        _point(item, "left_px") if _point(item, "left_px") is not None else [np.nan, np.nan]
        for item in frames
    ], dtype=np.float64)
    right_angles = np.asarray([
        item.get("right_angles_rad", [np.nan, np.nan]) for item in frames
    ], dtype=np.float64)
    left_angles = np.asarray([
        item.get("left_angles_rad", [np.nan, np.nan]) for item in frames
    ], dtype=np.float64)
    head_rotation = np.asarray([
        item.get("head_rotation_rad", [np.nan, np.nan, np.nan]) for item in frames
    ], dtype=np.float64)
    head_translation = np.asarray([
        item.get("head_translation_cm", [np.nan, np.nan, np.nan]) for item in frames
    ], dtype=np.float64)
    right_point_velocity = _frame_vectors(right_points, times)
    left_point_velocity = _frame_vectors(left_points, times)
    right_angle_velocity = _frame_vectors(right_angles, times)
    left_angle_velocity = _frame_vectors(left_angles, times)
    head_rotation_velocity = _frame_vectors(head_rotation, times)
    head_translation_velocity = _frame_vectors(head_translation, times)
    binocular_point_cosine = _vector_cosine(right_point_velocity, left_point_velocity)
    binocular_angle_cosine = _vector_cosine(right_angle_velocity, left_angle_velocity)
    future_displacement = targets - raw
    velocity_future_cosine = _vector_cosine(instant_velocity, future_displacement)
    speed = np.linalg.norm(instant_velocity, axis=1)
    previous_speed = np.roll(speed, 1)
    previous_speed[0] = np.nan
    speed_ratio = speed / np.maximum(previous_speed, 1e-9)
    head_rotation_speed = np.linalg.norm(head_rotation_velocity, axis=1)
    head_translation_speed = np.linalg.norm(head_translation_velocity, axis=1)

    binocular_confidence = np.clip((binocular_point_cosine + 0.2) / 1.2, 0.0, 1.0)
    binocular_confidence[~np.isfinite(binocular_confidence)] = 0.0
    predictions["rising_gain20_binocular_gate"] = _state_gated_prediction(
        raw,
        instant_velocity,
        rising_mask,
        maximum_lead,
        gain_ms=20.0,
        horizon_ms=horizon_ms,
        confidence=binocular_confidence,
    )
    episodes, endpoint_predictions, endpoint_targets = _saccade_episodes(
        times,
        raw,
        phases,
        instant_velocity,
        maximum_lead,
    )
    endpoint_valid = np.isfinite(endpoint_targets).all(axis=1)
    endpoint_methods = {
        name: _summary(
            np.linalg.norm(prediction - endpoint_targets, axis=1),
            endpoint_valid,
        )
        for name, prediction in endpoint_predictions.items()
    }
    ridge, ridge_model = _ridge_predictions(
        times, raw, targets, train_mask, maximum_lead,
    )
    predictions["learned_endpoint_ridge"] = ridge

    methods: dict[str, dict] = {}
    errors_by_method: dict[str, np.ndarray] = {}
    for name, prediction in predictions.items():
        errors = np.linalg.norm(prediction - targets, axis=1)
        errors[~np.isfinite(prediction).all(axis=1)] = np.nan
        errors_by_method[name] = errors
        methods[name] = {
            "all": _summary(errors, target_valid),
            "motion": _summary(errors, motion_mask),
            "saccade_candidates": _summary(errors, saccade_mask),
            "temporal_holdout": _summary(errors, holdout_mask),
            "saccade_holdout": _summary(errors, saccade_mask & holdout_mask),
            "rising": _summary(errors, rising_mask),
            "rising_holdout": _summary(errors, rising_mask & holdout_mask),
            "first_rising": _summary(errors, first_rising_mask),
            "first_rising_holdout": _summary(errors, first_rising_mask & holdout_mask),
            "continuing_rising": _summary(errors, continuing_rising_mask),
            "continuing_rising_holdout": _summary(errors, continuing_rising_mask & holdout_mask),
            "landing": _summary(errors, landing_mask),
            "landing_holdout": _summary(errors, landing_mask & holdout_mask),
        }
    ranked = [
        (stats["saccade_holdout"].get("p90_px", float("inf")), name)
        for name, stats in methods.items()
        if stats["saccade_holdout"].get("count", 0) >= 3
    ]
    best_p90_method = min(ranked)[1] if ranked else "raw_hold"
    state_gated_ranked = [
        (stats["saccade_holdout"].get("mean_px", float("inf")), name)
        for name, stats in methods.items()
        if name.startswith("rising_only_") and stats["saccade_holdout"].get("count", 0) >= 3
    ]
    recommended_method = min(state_gated_ranked)[1] if state_gated_ranked else "raw_hold"

    def worst_windows_for(method: str, limit: int) -> list[dict]:
        errors = errors_by_method[method].copy()
        focus = saccade_mask & holdout_mask
        if int(np.sum(focus & np.isfinite(errors))) < 3:
            focus = motion_mask & holdout_mask
        errors[~focus] = np.nan
        worst_indices = np.argsort(np.nan_to_num(errors, nan=-1.0))[-limit:][::-1]
        windows = []
        for index in worst_indices:
            if not np.isfinite(errors[index]):
                continue
            context = []
            for nearby in range(max(0, index - 4), min(len(frames), index + 5)):
                context.append({
                    "offset": int(nearby) - int(index),
                    "phone_time_ms": float(times[nearby]),
                    "raw_px": raw[nearby].tolist(),
                    "filtered_px": filtered[nearby].tolist(),
                    "future_target_px": (
                        targets[nearby].tolist()
                        if np.isfinite(targets[nearby]).all() else None
                    ),
                    "mode": str(modes[nearby]),
                    "phase": str(phases[nearby]),
                    "step_px": float(steps[nearby]),
                    "binocular_point_direction_cosine": (
                        float(binocular_point_cosine[nearby])
                        if np.isfinite(binocular_point_cosine[nearby]) else None
                    ),
                    "speed_ratio": (
                        float(speed_ratio[nearby]) if np.isfinite(speed_ratio[nearby]) else None
                    ),
                })
            windows.append({
                "index": int(index),
                "phone_time_ms": float(times[index]),
                "error_px": float(errors[index]),
                "prediction_px": predictions[method][index].tolist(),
                "target_80ms_px": targets[index].tolist(),
                "context": context,
            })
        return windows

    worst_by_method = {
        name: worst_windows_for(name, 4) for name in predictions
    }
    return {
        "schema": "opengazelink-motion-analysis-v1",
        "source": str(Path(path).resolve()),
        "horizon_ms": float(horizon_ms),
        "screen_diagonal_px": diagonal,
        "maximum_lead_fraction": maximum_lead_fraction,
        "maximum_lead_px": maximum_lead,
        "valid_frames": len(frames),
        "target_frames": int(np.sum(target_valid)),
        "motion_frames": int(np.sum(motion_mask)),
        "saccade_candidate_frames": int(np.sum(saccade_mask)),
        "rising_frames": int(np.sum(rising_mask)),
        "first_rising_frames": int(np.sum(first_rising_mask)),
        "continuing_rising_frames": int(np.sum(continuing_rising_mask)),
        "landing_frames": int(np.sum(landing_mask)),
        "temporal_split_phone_time_ms": float(split_time),
        "methods": methods,
        "confidence_signals": {
            "binocular_point_direction_cosine": {
                "rising": _distribution(binocular_point_cosine, rising_mask),
                "rising_holdout": _distribution(binocular_point_cosine, rising_mask & holdout_mask),
                "landing": _distribution(binocular_point_cosine, landing_mask),
            },
            "binocular_eye_angle_direction_cosine": {
                "rising": _distribution(binocular_angle_cosine, rising_mask),
                "rising_holdout": _distribution(binocular_angle_cosine, rising_mask & holdout_mask),
                "landing": _distribution(binocular_angle_cosine, landing_mask),
            },
            "velocity_to_future_direction_cosine": {
                "rising": _distribution(velocity_future_cosine, rising_mask),
                "rising_holdout": _distribution(velocity_future_cosine, rising_mask & holdout_mask),
                "landing": _distribution(velocity_future_cosine, landing_mask),
            },
            "speed_ratio_to_previous_frame": {
                "rising": _distribution(speed_ratio, rising_mask),
                "rising_holdout": _distribution(speed_ratio, rising_mask & holdout_mask),
                "landing": _distribution(speed_ratio, landing_mask),
            },
            "head_rotation_speed_rad_per_ms": {
                "rising": _distribution(head_rotation_speed, rising_mask),
                "landing": _distribution(head_rotation_speed, landing_mask),
            },
            "head_translation_speed_cm_per_ms": {
                "rising": _distribution(head_translation_speed, rising_mask),
                "landing": _distribution(head_translation_speed, landing_mask),
            },
        },
        "settled_endpoint_analysis": {
            "complete_episode_count": len(episodes),
            "rising_frames_with_endpoint": int(np.sum(endpoint_valid)),
            "methods": endpoint_methods,
            "episodes": episodes,
        },
        "phase_replay": {
            "changed_frame_count": int(np.sum(replayed_phases_array != phases)),
            "recorded_large_jump_as_landing_count": int(np.sum(
                (steps >= 0.03 * diagonal) & (phases == "landing")
            )),
            "replayed_large_jump_as_landing_count": int(np.sum(
                (steps >= 0.03 * diagonal) & (replayed_phases_array == "landing")
            )),
        },
        "learned_endpoint_ridge": ridge_model,
        "best_saccade_holdout_p90_method": best_p90_method,
        "recommended_state_gated_method": recommended_method,
        "worst_windows": worst_windows_for(recommended_method, 8),
        "worst_windows_by_method": worst_by_method,
    }


def write_motion_analysis(path: Path, horizon_ms: float = 80.0, output_path: Path | None = None) -> Path:
    source = Path(path)
    result = analyze_motion_recording(source, horizon_ms)
    destination = output_path or source.with_name(f"{source.stem}-analysis-{int(horizon_ms)}ms.json")
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return destination
