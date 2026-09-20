"""Small, model-bound stage-two calibration for automatic fixation gating.

Only training fixations estimate parameters. Validation may reject them; test
sequences are reported but never choose parameters. No spatial weights change.
"""
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

from .paths import DATA_DIR

PROFILE_PATH = DATA_DIR / "event-stability-profile.json"
MODEL_PATH = DATA_DIR / "conditioned-video-model.json"
SCHEMA = "opengazelink-auto-stability-v1"
PARAMETER_RANGES = {"noise_prior_deg": (.04, 1.5), "noise_floor_deg": (.04, .75), "settle_ms": (16., 50.)}


def model_digest(metadata):
    return metadata.get("variants", {}).get("conditioned_video", {}).get("module_sha256", "")


def context_signature(config, camera):
    config = asdict(config) if is_dataclass(config) else config
    keys = ("input_source", "screen_width", "screen_height", "screen_diagonal_inches",
            "camera_offset_x_cm", "camera_offset_y_cm", "camera_offset_z_cm",
            "windows_camera_index", "windows_camera_width", "windows_camera_height",
            "windows_camera_fps", "windows_camera_fov_x_degrees", "rotate", "mirror",
            "one_euro_min_cutoff", "one_euro_beta", "one_euro_derivative_cutoff")
    camera_keys = ("rawWidth", "rawHeight", "width", "height", "fx", "fy", "cx", "cy", "rotate", "mirror", "source")
    value = {"config": {key: config.get(key) for key in keys},
             "camera": {key: camera.get(key) for key in camera_keys},
             "camera_id": camera.get("sourceMetadata", {}).get("cameraId")}
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def fixation_windows(rows, split):
    windows = {}
    for i, row in enumerate(rows):
        if row.get("split") != split or row.get("weight", 0) <= .5:
            continue
        if row.get("target_age_ms") is not None:
            if row["target_age_ms"] >= 700 and row.get("target_onset_trial_ms") is not None:
                key = (row["trial_id"], row.get("capture_segment", 0), round(row["target_onset_trial_ms"], 3))
                windows.setdefault(key, []).append(i)
            continue
        age = float(row.get("trial_age_ms", -1))
        for phase, (start, end) in enumerate(((0, 1200), (1200, 2700), (2700, 4201))):
            if start + 700 <= age < end:
                windows.setdefault((row["trial_id"], phase), []).append(i)
                break
    return [ids for ids in windows.values() if len(ids) >= 8]


def estimate_profile(raw, times, scales, rows):
    """Estimate a noise prior and landing quiet period from training only."""
    raw, times, scales = map(np.asarray, (raw, times, scales))
    windows = fixation_windows(rows, "train")
    trials = {rows[ids[0]]["trial_id"] for ids in windows}
    if len(windows) < 6 or len(trials) < 3:
        return None
    noise, intervals = [], []
    for ids in windows:
        t, points = times[ids], raw[ids] / scales[ids, None]
        dt = np.diff(t)
        if (not np.isfinite(points).all() or not np.isfinite(t).all()
                or np.any(dt <= 0) or np.any(dt > 100) or t[-1] - t[0] < 120):
            continue
        design = np.c_[np.ones(len(t)), (t - t.mean()) / 1000]
        residual = points - design @ np.linalg.lstsq(design, points, rcond=None)[0]
        noise.append(float(np.median(np.linalg.norm(residual, axis=1)) / 1.177))
        intervals.extend(dt.tolist())
    if len(noise) < 6:
        return None
    prior = float(np.clip(np.median(noise), .04, 1.5))
    return {"calibrated": True, "noise_prior_deg": prior,
            "noise_floor_deg": float(np.clip(prior * .5, .04, .75)),
            "settle_ms": float(np.clip(2 * np.median(intervals), 16., 50.)),
            "training_trials": len(trials), "training_fixations": len(noise)}


def validate_profile(profile, baseline, candidate, targets, rows, baseline_states, candidate_states):
    """Validation gates noise reduction, location error and spurious predictions."""
    if profile is None:
        return {"accepted": False, "reason": "insufficient_training_fixations"}
    windows = fixation_windows(rows, "validation")
    if len(windows) < 6 or len({rows[ids[0]]["trial_id"] for ids in windows}) < 2:
        return {"accepted": False, "reason": "insufficient_validation_fixations"}
    ids = np.concatenate(windows)
    if not all(np.isfinite(points[ids]).all() for points in (baseline, candidate, targets)):
        return {"accepted": False, "reason": "invalid_validation_output"}

    def metrics(points, states):
        jitter = [np.sqrt(np.mean(np.sum((points[window] - np.median(points[window], axis=0)) ** 2, axis=1)))
                  for window in windows]
        return {"jitter_px": float(np.median(jitter)),
                "p75_error_px": float(np.percentile(np.linalg.norm(points[ids] - targets[ids], axis=1), 75)),
                "settled_prediction_frames": sum(bool(states[i].get("prediction_active")) for i in ids),
                "settled_saccade_frames": sum(states[i].get("mode", "").startswith("event_saccade") for i in ids),
                "stability_switches": sum(bool(states[a].get("stability_active")) != bool(states[b].get("stability_active"))
                                          for window in windows for a, b in zip(window, window[1:]))}

    before, after = metrics(baseline, baseline_states), metrics(candidate, candidate_states)
    accepted = (after["jitter_px"] <= before["jitter_px"] * 1.10 + .5
                and after["p75_error_px"] <= before["p75_error_px"] * 1.05 + 1.
                and after["settled_prediction_frames"] <= before["settled_prediction_frames"]
                and after["settled_saccade_frames"] <= before["settled_saccade_frames"]
                and after["stability_switches"] <= before["stability_switches"])
    return {"accepted": bool(accepted), "reason": "accepted" if accepted else "validation_regression",
            "baseline": before, "candidate": after, "validation_fixations": len(windows)}


def read_profile(config, camera, metadata=None, *, path=PROFILE_PATH, model_path=MODEL_PATH):
    """Return None on missing, stale, foreign or corrupt evidence."""
    try:
        profile = json.loads(Path(path).read_text(encoding="utf-8"))
        metadata = metadata if metadata is not None else json.loads(Path(model_path).read_text(encoding="utf-8"))
        if (profile.get("schema") != SCHEMA or not profile.get("accepted")
                or not model_digest(metadata) or profile.get("model_sha256") != model_digest(metadata)
                or profile.get("context_signature") != context_signature(config, camera)):
            return None
        parameters = profile["parameters"]
        if not parameters.get("calibrated"):
            return None
        for name, (low, high) in PARAMETER_RANGES.items():
            value = float(parameters[name])
            if not np.isfinite(value) or not low <= value <= high:
                return None
        return profile
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return None


def publish_profile(report, config, camera, *, path=PROFILE_PATH, model_path=MODEL_PATH):
    """Publish only against the same active model, using an atomic replacement."""
    decision = report.get("stability_calibration", {})
    if not decision.get("accepted") or not decision.get("parameters"):
        return False
    metadata = json.loads(Path(model_path).read_text(encoding="utf-8"))
    if not model_digest(metadata) or report.get("model_sha256") != model_digest(metadata):
        return False
    profile = {"schema": SCHEMA, "accepted": True, "model_sha256": model_digest(metadata),
               "context_signature": context_signature(config, camera),
               "parameters": decision["parameters"], "validation": decision,
               "created_at": datetime.now(timezone.utc).isoformat(),
               "report": str(Path(report["training_directory"]) / "report.json")}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(profile, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)
    return True
