"""One capture, staged personal adaptation, sequence selection, TorchScript export.

No target locations, phases, or target-change flags enter the network. Stable
rails supervise perpendicular position and a bounded tangential interval.
"""
from __future__ import annotations

import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
from collections import Counter

import numpy as np
import torch
from torch.nn import functional as F

from .normalized_eye import screen_camera_origin
from .conditioned_eye_model import runtime_geometry_reference, stabilize_runtime_geometry
from .paths import DATA_DIR
from .video_dataset import ALIGNMENT_POLICY, align_frames, read_jsonl
from .video_network import VideoAdapter, VideoAppearance, VideoInference
from .video_session import SCHEMA, write_json


def project(directions, weights, rotations, centers, origin, size):
    sign = directions.new_tensor([[1., 1., 1.], [-1., 1., 1.]])
    ray = torch.einsum("beij,bej->bei", rotations, directions * sign)
    denominator = ray[..., 2].clamp(max=-1e-5)
    distance = (origin[2] - centers[..., 2]) / denominator
    point = centers + distance[..., None] * ray
    xy = (point[..., :2] - origin[:2]) / size * directions.new_tensor([-1., 1.]) + .5
    return (xy * weights[..., None]).sum(1)


from .spatial_metrics import spatial_metrics as _metrics, eccentricity_degrees, acceptance, primary, supported


def candidate_is_safe(metrics, baseline, best_median):
    return acceptance(metrics, baseline) and primary(metrics)["median_px"] < best_median - 1e-4


def coordinate_loss(prediction, target, constraint=None):
    if constraint is None:
        return F.smooth_l1_loss(prediction, target, beta=.03)
    tangent = prediction.new_tensor(constraint["tangent"])
    normal = prediction.new_tensor(constraint["normal"])
    perpendicular = F.smooth_l1_loss(prediction @ normal, prediction.new_full((len(prediction),), constraint["normal_target"]), beta=.03)
    along = prediction @ tangent
    outside = F.relu(constraint["lower"] - along) + F.relu(along - constraint["upper"])
    return perpendicular + .15 * F.smooth_l1_loss(outside, torch.zeros_like(outside), beta=.03)


JOINT_HORIZONS_MS = (33., 67., 85., 100.)
JOINT_FUTURE_LOSS_WEIGHT = .35
JOINT_MOVING_THRESHOLD_PX = 12.


def joint_future_pairs(rows, weights=None, horizons=JOINT_HORIZONS_MS):
    """Build future observations without crossing trial, split, gap or reset.

    Coordinate-label weights do not gate these pairs.  Motion supervision uses
    the frozen personal spatial replay; stimulus labels remain spatial anchors.
    """
    groups = np.zeros(len(rows), dtype=np.int64)
    for index in range(1, len(rows)):
        previous, current = rows[index - 1], rows[index]
        discontinuity = (
            current["reset"]
            or current["index"] != previous["index"] + 1
            or current["split"] != previous["split"]
            or current["trial_id"] != previous["trial_id"]
            or current.get("capture_segment", 0) != previous.get("capture_segment", 0)
            or not 0 < current["source_ms"] - previous["source_ms"] <= 100.
        )
        groups[index] = groups[index - 1] + int(discontinuity)
    result = {index: [] for index in range(len(rows))}
    for group in np.unique(groups):
        ids = np.flatnonzero(groups == group)
        times = np.asarray([rows[index]["source_ms"] for index in ids], dtype=np.float64)
        if len(ids) < 2 or np.any(np.diff(times) <= 0):
            continue
        for local, source in enumerate(ids):
            for requested_ms in horizons:
                target_time = times[local] + requested_ms
                insertion = int(np.searchsorted(times, target_time, side="left"))
                candidates = [value for value in (insertion - 1, insertion)
                              if local < value < len(ids)]
                if not candidates:
                    continue
                future_local = min(candidates, key=lambda value: abs(times[value] - target_time))
                destination = int(ids[future_local])
                actual_ms = float(times[future_local] - times[local])
                if abs(actual_ms - requested_ms) > 20.:
                    continue
                result[int(source)].append((destination, actual_ms / 1000., 1.))
    return result


def joint_motion_teacher(points, rows, weights):
    """Reconstruct a frozen personal-spatial trajectory for future labels."""
    points = np.asarray(points, dtype=np.float32)
    segments = np.zeros(len(rows), dtype=np.int64)
    for index in range(1, len(rows)):
        previous, current = rows[index - 1], rows[index]
        discontinuity = (
            current["reset"]
            or current["index"] != previous["index"] + 1
            or current["split"] != previous["split"]
            or current["trial_id"] != previous["trial_id"]
            or current.get("capture_segment", 0) != previous.get("capture_segment", 0)
            or not 0 < current["source_ms"] - previous["source_ms"] <= 100.
        )
        segments[index] = segments[index - 1] + int(discontinuity)
    weights = np.asarray(weights)
    train = np.asarray([row['split'] == 'train' for row in rows])
    stable = (weights[1:] > .5) & (weights[:-1] > .5) & (segments[1:] == segments[:-1]) & train[1:] & train[:-1]
    differences = np.linalg.norm(np.diff(points, axis=0)[stable], axis=1)
    noise = float(np.clip(np.median(differences) / 1.177 if len(differences) else .008, .001, .03))
    # Lazy import avoids prediction_dataset -> video_training.project cycle.
    from .prediction_dataset import reconstruct
    times = np.asarray([row["source_ms"] for row in rows], dtype=np.float64)
    teacher = reconstruct(points, times, segments, noise)
    return np.asarray(teacher, dtype=np.float32), noise


def joint_event_teacher(teacher, rows, noise):
    """Compatibility API for the shared stimulus-conditioned offline labels."""
    from .motion_labels import label_motion
    labels = label_motion(teacher, rows, noise)
    return (labels["phase"], labels["landing"]-teacher,
            labels["remaining"], labels["landing_valid"])


def joint_motion_delta(motion, horizon):
    """Numpy event-aware displacement used by selection and testing."""
    pursuit = motion[:2]*horizon+.5*motion[2:4]*horizon*horizon
    if len(motion)==4:
        return pursuit
    logits=motion[4:7]-np.max(motion[4:7]); probability=np.exp(logits);probability/=probability.sum()
    landing=motion[7:9]*(1-np.exp(-3*horizon/max(.008,motion[9])))
    reliability=float(np.clip(.06/max(.003,motion[10]),.25,1.))
    return reliability*(probability[1]*pursuit+probability[2]*landing)


def joint_event_metrics(motion, ids, phase, landing, remaining, valid, wh):
    """Measure whether an 11-output head learned events, not just mean motion."""
    if motion.shape[1] != 11:
        return None
    ids=np.asarray(ids,dtype=np.int64); truth=phase[ids]; predicted=motion[:,4:7].argmax(1)
    confusion=np.zeros((3,3),np.int64)
    for expected,actual in zip(truth,predicted):
        confusion[expected,actual]+=1
    recall=[]
    for state in range(3):
        recall.append(float(confusion[state,state]/max(1,confusion[state].sum())))
    local=valid[ids]
    landing_error=np.linalg.norm((motion[local,7:9]-landing[ids[local]])*(wh-1),axis=1)
    time_error=np.abs(motion[local,9]-remaining[ids[local]])*1000
    return {"confusion_rows_true_columns_predicted":confusion.tolist(),
            "recall":{"fixation":recall[0],"pursuit":recall[1],"saccade_proxy":recall[2]},
            "balanced_accuracy":float(np.mean(recall)),"landing_samples":int(local.sum()),
            "landing_mean_px":float(landing_error.mean()) if len(landing_error) else None,
            "landing_p95_px":float(np.percentile(landing_error,95)) if len(landing_error) else None,
            "remaining_mean_abs_ms":float(time_error.mean()) if len(time_error) else None,
            "uncertainty_mean":float(motion[:,10].mean()),
            "uncertainty_by_state":{name:float(motion[truth==state,10].mean()) if np.any(truth==state) else None
                                    for state,name in enumerate(("fixation","pursuit","saccade_proxy"))}}


def balance_joint_future_pairs(pairs, teacher, wh, train_ids):
    """Give moving and stationary replay pairs equal mass per horizon."""
    train_ids = set(map(int, train_ids))
    classes = {}
    for source, values in pairs.items():
        if source not in train_ids:
            continue
        for destination, horizon, _ in values:
            requested = int(min(JOINT_HORIZONS_MS, key=lambda value: abs(value - horizon * 1000.)))
            distance = np.linalg.norm((teacher[destination] - teacher[source]) * (wh - 1))
            classes[(source, destination, horizon)] = (requested, distance > JOINT_MOVING_THRESHOLD_PX)
    counts = Counter(classes.values())
    totals = Counter(horizon for horizon, _ in classes.values())
    result = {}
    for source, values in pairs.items():
        result[source] = []
        for destination, horizon, _ in values:
            key = classes.get((source, destination, horizon))
            if key is None:
                pair_weight = 1.
            else:
                class_count = counts[key]
                present = sum(counts[(key[0], moving)] > 0 for moving in (False, True))
                pair_weight = totals[key[0]] / max(1, present * class_count)
            result[source].append((destination, horizon, float(np.clip(pair_weight, .25, 4.))))
    audit = {str(horizon): {
        "stationary": int(counts[(horizon, False)]),
        "moving": int(counts[(horizon, True)]),
    } for horizon in map(int, JOINT_HORIZONS_MS)}
    return result, audit


def joint_future_metrics(current, motion, ids, pairs, target, wh):
    """Evaluate y_hat(t+h) against the future replay teacher at t+h.

    ``current`` is only the source position y_hat(t).  The hold baseline also
    starts at that same source position, so the comparison isolates whether
    the learned velocity/acceleration improves prediction at each horizon.
    """
    position = {int(frame): offset for offset, frame in enumerate(ids)}
    errors, hold_errors, displacement_errors, zero_displacement_errors = [], [], [], []
    moving_flags = []
    by_horizon = {int(value): [[], [], []] for value in JOINT_HORIZONS_MS}
    for source in ids:
        source = int(source)
        for destination, horizon, _ in pairs.get(source, ()):
            local_motion = motion[position[source]]
            delta = joint_motion_delta(local_motion, horizon)
            predicted = current[position[source]] + delta
            # destination is the future frame selected by joint_future_pairs.
            # The teacher at source is never used as the future coordinate.
            future_error = np.linalg.norm((predicted - target[destination]) * (wh - 1))
            hold_error = np.linalg.norm((current[position[source]] - target[destination]) * (wh - 1))
            target_delta = target[destination] - target[source]
            displacement_errors.append(np.linalg.norm((delta - target_delta) * (wh - 1)))
            zero_displacement_errors.append(np.linalg.norm(target_delta * (wh - 1)))
            moving = np.linalg.norm((target[destination] - target[source]) * (wh - 1)) > JOINT_MOVING_THRESHOLD_PX
            errors.append(future_error)
            hold_errors.append(hold_error)
            moving_flags.append(moving)
            requested_ms = min(JOINT_HORIZONS_MS, key=lambda value: abs(value - horizon * 1000.))
            by_horizon[int(requested_ms)][0].append(future_error)
            by_horizon[int(requested_ms)][1].append(hold_error)
            by_horizon[int(requested_ms)][2].append(moving)
    horizon_metrics = {}
    for requested_ms, (future_values, hold_values, local_moving) in by_horizon.items():
        if not future_values:
            continue
        future_values = np.asarray(future_values)
        hold_values = np.asarray(hold_values)
        local_moving = np.asarray(local_moving, dtype=bool)
        moving_future = future_values[local_moving]
        moving_hold = hold_values[local_moving]
        horizon_metrics[str(requested_ms)] = {
            "samples": int(len(future_values)),
            "future_mean_px": float(future_values.mean()),
            "hold_latest_mean_px": float(hold_values.mean()),
            "future_minus_hold_px": float(future_values.mean() - hold_values.mean()),
            "improves_hold": bool(future_values.mean() < hold_values.mean()),
            "moving_samples": int(local_moving.sum()),
            "moving_future_mean_px": float(moving_future.mean()) if len(moving_future) else None,
            "moving_hold_latest_mean_px": float(moving_hold.mean()) if len(moving_hold) else None,
            "moving_improves_hold": bool(len(moving_future) and moving_future.mean() < moving_hold.mean()),
        }
    if not errors:
        return {"samples": 0, "mean_px": None, "median_px": None, "p95_px": None,
                "hold_mean_px": None, "future_minus_hold_px": None,
                "displacement_mean_px": None, "zero_displacement_mean_px": None,
                "improves_hold": False, "by_horizon_ms": horizon_metrics,
                "moving_samples": 0, "moving_improves_hold": False,
                "evaluation": "y_hat(t+h) versus frozen_replay_teacher(t+h)"}
    errors, hold_errors = np.asarray(errors), np.asarray(hold_errors)
    displacement_errors = np.asarray(displacement_errors)
    zero_displacement_errors = np.asarray(zero_displacement_errors)
    moving_flags = np.asarray(moving_flags, dtype=bool)
    moving_errors, moving_hold = errors[moving_flags], hold_errors[moving_flags]
    return {"samples": int(len(errors)), "mean_px": float(errors.mean()),
            "median_px": float(np.median(errors)), "p95_px": float(np.percentile(errors, 95)),
            "hold_mean_px": float(hold_errors.mean()),
            "future_minus_hold_px": float(errors.mean() - hold_errors.mean()),
            "improves_hold": bool(errors.mean() < hold_errors.mean()),
            "displacement_mean_px": float(displacement_errors.mean()),
            "zero_displacement_mean_px": float(zero_displacement_errors.mean()),
            "displacement_improves_zero": bool(displacement_errors.mean() < zero_displacement_errors.mean()),
            "moving_displacement_mean_px": float(displacement_errors[moving_flags].mean()) if moving_flags.any() else None,
            "moving_zero_displacement_mean_px": float(zero_displacement_errors[moving_flags].mean()) if moving_flags.any() else None,
            "moving_displacement_improves_zero": bool(moving_flags.any() and displacement_errors[moving_flags].mean() < zero_displacement_errors[moving_flags].mean()),
            "moving_samples": int(moving_flags.sum()),
            "moving_mean_px": float(moving_errors.mean()) if len(moving_errors) else None,
            "moving_hold_mean_px": float(moving_hold.mean()) if len(moving_hold) else None,
            "moving_improves_hold": bool(len(moving_errors) and moving_errors.mean() < moving_hold.mean()),
            "by_horizon_ms": horizon_metrics,
            "evaluation": "y_hat(t+h) versus frozen_replay_teacher(t+h)"}


def joint_selection_score(current, current_baseline, future):
    """Optimize current and future gaze together while bounding current risk."""
    if not future.get("samples") or not future.get("improves_hold"):
        return float("inf")
    if future.get("moving_samples") and not future.get("moving_improves_hold"):
        return float("inf")
    horizon_100 = future.get("by_horizon_ms", {}).get("100", {})
    if horizon_100.get("moving_samples") and not horizon_100.get("moving_improves_hold"):
        return float("inf")
    if not supported(current) or not supported(current_baseline):
        return float("inf")
    current, current_baseline = primary(current), primary(current_baseline)
    if (current["median_px"] > current_baseline["median_px"] * 1.03
            or current["mean_px"] > current_baseline["mean_px"] * 1.03
            or current["p95_px"] > current_baseline["p95_px"] * 1.10):
        return float("inf")
    current_ratio = .5 * current["median_px"] / max(1., current_baseline["median_px"])
    current_ratio += .15 * current["mean_px"] / max(1., current_baseline["mean_px"])
    future_ratio = .35 * future["mean_px"] / max(1., future["hold_mean_px"])
    return float(current_ratio + future_ratio)


def camera_profile(camera_model):
    """Return the capture geometry fields that affect normalized eye inputs."""
    camera_model = camera_model or {}
    metadata = camera_model.get("sourceMetadata") or {}
    effective = metadata.get("effectiveStreamCrop") or {}
    software = metadata.get("softwareCrop") or {}
    result = {
        key: camera_model.get(key) for key in (
            "rawWidth", "rawHeight", "width", "height", "fx", "fy", "cx", "cy",
            "rotate", "mirror", "source",
        )
    }
    result.update(
        cameraId=metadata.get("cameraId"), lensFacing=metadata.get("lensFacing"),
        effectiveStreamCrop={
            key: effective.get(key) for key in ("left", "top", "width", "height")
        },
        softwareCrop={
            key: software.get(key) for key in ("left", "top", "width", "height", "right", "bottom")
        },
    )
    return result


def load_joint_frame_tensors(session_path, usable, base_item, cancelled=lambda: False):
    """Load full-rate replay tensors, with a reusable uncompressed array cache."""
    cache = Path(session_path) / "joint-frame-cache"
    manifest = cache / "metadata.json"
    replay_manifest = Path(session_path) / "replay-frames.jsonl"
    if not replay_manifest.is_file():
        replay_manifest = Path(session_path) / "frames.jsonl"
    replay_sha256 = hashlib.sha256(replay_manifest.read_bytes()).hexdigest()
    expected_indices = np.asarray([row["index"] for row in usable], dtype=np.int64)
    if manifest.is_file():
        cache_metadata = json.loads(manifest.read_text(encoding="utf-8"))
        index_path = cache / "frame-index.npy"
        if (cache_metadata.get("frames") == len(usable)
                and cache_metadata.get("replay_sha256") == replay_sha256 and index_path.is_file()
                and np.array_equal(np.load(index_path, mmap_mode="r"), expected_indices)):
            arrays = tuple(np.load(cache / name, mmap_mode="r") for name in (
                "images.npy", "raw-geometry.npy", "rotations.npy", "centers.npy",
            ))
            if (arrays[0].shape == (len(usable), 2, 2, 36, 64)
                    and arrays[1].shape == (len(usable), 2, 70)):
                geometry = stabilize_runtime_geometry(np.asarray(arrays[1]), base_item)
                return tuple(torch.from_numpy(np.asarray(value).copy() if not value.flags.writeable else value)
                             for value in (arrays[0], geometry, arrays[2], arrays[3])), True
    images, geometry, rotations, centers = [], [], [], []
    for offset, row in enumerate(usable):
        if offset % 240 == 0 and cancelled():
            raise RuntimeError("VIDEO training cancelled; capture retained")
        input_path = (Path(session_path) / row["input"]).resolve()
        input_path.relative_to(Path(session_path).resolve())
        with np.load(input_path, allow_pickle=False) as stored:
            images.append(stored["images"][:, 1].copy())
            geometry.append(np.concatenate((stored["head"], stored["points"][:, 1], stored["crop"]), axis=1).astype(np.float32))
            rotations.append(stored["rotation"].copy())
            centers.append(stored["center"].copy())
    raw_geometry = np.asarray(geometry)
    values = (np.asarray(images), raw_geometry, np.asarray(rotations), np.asarray(centers))
    cache.mkdir(exist_ok=True)
    temporary = cache / "metadata.json.tmp"
    for name, value in zip(("images.npy", "raw-geometry.npy", "rotations.npy", "centers.npy"), values):
        np.save(cache / name, value)
    np.save(cache / "frame-index.npy", expected_indices)
    temporary.write_text(json.dumps({"frames": len(usable), "replay_sha256": replay_sha256,
                                     "source": "complete_mediapipe_video_replay"},
                                    ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(manifest)
    values = (values[0], stabilize_runtime_geometry(raw_geometry, base_item), values[2], values[3])
    return tuple(torch.from_numpy(value) for value in values), False


def train_session(path, **kwargs):
    """Each attempt gets immutable provenance, checkpoints and alignment output."""
    path = Path(path)
    run_path = path / "training-runs" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S-%fZ")
    run_path.mkdir(parents=True)
    options = {k: str(v) if isinstance(v, Path) else v for k,v in kwargs.items() if k not in ("progress", "cancelled")}
    record = {"session":str(path.resolve()), "options":options, "alignment_policy":ALIGNMENT_POLICY}
    # Preserve outputs from older versions before updating convenience aliases.
    legacy = [p for p in path.iterdir() if p.is_file() and (p.name.startswith("conditioned-video") or p.name in ("training-report.json", "selection-predictions.npz"))]
    if legacy:
        archive = run_path / "previous-session-output"
        archive.mkdir()
        for previous in legacy:
            shutil.copy2(previous, archive / previous.name)
    from .paths import RESOURCE_ROOT
    source_dir = run_path / "source"
    source_dir.mkdir()
    source_hashes = {}
    for name in ("spatial_metrics.py", "head_coverage.py", "video_training.py", "training_runtime.py", "motion_labels.py", "prediction_dataset.py", "personal_binocular_training.py", "video_network.py", "video_dataset.py", "video_session.py", "video_archive.py", "video_replay.py", "normalized_eye.py", "conditioned_eye.py", "camera.py"):
        source = Path(__file__).with_name(name)
        if not source.is_file():
            source = RESOURCE_ROOT / "provenance" / "opengazelink_pc" / name
        if source.is_file():
            shutil.copy2(source, source_dir / name)
            source_hashes[name] = hashlib.sha256(source.read_bytes()).hexdigest()
    record["source_sha256"] = source_hashes
    write_json(run_path / "run.json", dict(record, state="running"))
    try:
        result = _train_session(path, run_path=run_path, **kwargs)
        write_json(run_path / "run.json", dict(record, state="complete"))
        return result
    except Exception as error:
        write_json(run_path / "run.json", dict(record, state="failed", error=str(error)))
        raise


def _train_session(path, *, run_path, base_path=None, epochs=30, progress=lambda _: None,
                   cancelled=lambda: False, pursuit_lag_ms=None, publish=True, train_temporal=True,
                   protected_joint=False, joint_frame_budget=0, detach_motion_state=False,
                   event_motion=False):
    path = Path(path)
    if not 1 <= epochs <= 300:
        raise ValueError("epochs must be between 1 and 300")
    if joint_frame_budget and (not protected_joint or joint_frame_budget<64):
        raise ValueError("joint frame budget requires protected joint training and at least 64 frames")
    session = json.loads((path / "session.json").read_text(encoding="utf-8"))
    if session.get("schema") != SCHEMA:
        raise ValueError("unsupported VIDEO session")
    if pursuit_lag_ms is not None and not 0 <= pursuit_lag_ms <= 300:
        raise ValueError("pursuit lag must be an explicit 0–300 ms experimental assumption")
    progress("video_alignment")
    rows = align_frames(read_jsonl(path / "frames.jsonl"), read_jsonl(path / "stimulus.jsonl"),
                        pursuit_lag_ms=pursuit_lag_ms, discarded_segments=session.get("discarded_segments", []))
    with (run_path / "alignment.jsonl").open("w", encoding="utf-8") as stream:
        for row in rows:
            # Full MediaPipe matrices live once in the compressed input NPZ.
            # Never duplicate verbose capture diagnostics into training runs.
            stream.write(json.dumps({key:value for key,value in row.items() if key != "diagnostics"},
                                    ensure_ascii=False, allow_nan=False) + "\n")
    shutil.copy2(path / "session.json", run_path / "capture-metadata.json")
    # Whole final-pass trajectories are held out, never random adjacent frames.
    usable = [row for row in rows if row["valid"] and row.get("input")]
    from .calibration_split import split_masks
    train_mask, selection_mask, test_mask = split_masks(usable, session.get("plan"))
    weights_np = np.array([row["weight"] for row in usable], dtype=np.float32)
    spatial_only = bool(session.get("plan")) and all(
        step.get("calibration_stage") == "spatial_v1" for step in session["plan"]
    )
    if spatial_only:
        # Stage one deliberately uses rail constraints (weight=.25), so the
        # old exact-anchor threshold would reject every line-only capture.
        train_min = (weights_np[train_mask] > 0).sum()
        selection_min = (weights_np[selection_mask] > 0).sum()
        if train_min < 60 or selection_min < 30:
            raise ValueError("有效直线数据不足：训练至少需要60帧约束，模型选择段至少30帧；原始数据已保留")
    elif (weights_np[train_mask] > .5).sum() < 60 or (weights_np[selection_mask] > .5).sum() < 30:
        raise ValueError("有效连续数据不足：训练至少需要60帧稳定锚点，模型选择段至少30帧；原始数据已保留")
    cfg = session["config"]
    capture_camera_profile = camera_profile(session.get("camera_model"))
    if base_path is None:
        from .personal_binocular_training import train_personal_binocular
        base_path = train_personal_binocular(
            path, rows, usable, run_path, cfg, capture_camera_profile,
            epochs=epochs, progress=progress, cancelled=cancelled,
        )
    base_path = Path(base_path)
    if not base_path.is_file():
        raise ValueError("双眼空间模型不存在；连续数据已保存，可随后重训")
    metadata = json.loads(base_path.read_text(encoding="utf-8"))
    base_item = metadata["variants"]["conditioned_binocular"]
    reference_indices, reference_values, reference_tolerance = runtime_geometry_reference(base_item)
    module_path = base_path.with_name(base_item["module_file"])
    if not base_item.get("raw_inputs"):
        raise ValueError("VIDEO training requires a raw-input binocular base")
    base_digest = hashlib.sha256(module_path.read_bytes()).hexdigest()
    if base_item.get("module_sha256") and base_digest != base_item["module_sha256"]:
        raise ValueError("base module checksum mismatch")
    shutil.copy2(base_path, run_path / "base-metadata.json")
    shutil.copy2(module_path, run_path / "base.pt")
    camera_position = [cfg["camera_offset_x_cm"], cfg["camera_offset_y_cm"], cfg["camera_offset_z_cm"]]
    wh = np.array([cfg["screen_width"], cfg["screen_height"]], dtype=np.float32)
    origin_np = screen_camera_origin(int(wh[0]), int(wh[1]), cfg["screen_diagonal_inches"], camera_position)
    size_np = cfg["screen_diagonal_inches"] * 2.54 * wh / np.linalg.norm(wh)
    from .training_runtime import configure_training_threads
    training_threads = configure_training_threads()
    torch.manual_seed(9301)
    base = torch.jit.load(str(module_path), map_location="cpu").eval()
    for parameter in base.parameters():
        parameter.requires_grad_(False)
    appearance = VideoAppearance(base.model.image).eval() if hasattr(base, "model") and hasattr(base.model, "image") else None
    feature_dim = 404 if appearance is not None else 148
    mean = np.array(base_item["normalization"]["mean"], np.float32)
    scale = np.array(base_item["normalization"]["scale"], np.float32)
    # Evaluate the installed personal VIDEO model as an additional publication
    # comparator; improvement over the public base alone must not replace a
    # better existing personal model. Geometry changes invalidate this comparison.
    incumbent, incumbent_predictions, incumbent_weights = None, [], []
    incumbent_metadata_path = DATA_DIR / "conditioned-video-model.json"
    if not spatial_only and incumbent_metadata_path.is_file():
        incumbent_metadata = json.loads(incumbent_metadata_path.read_text(encoding="utf-8"))
        compatible = (incumbent_metadata.get("input_source") == cfg["input_source"]
                      and incumbent_metadata.get("screen") == {"width":int(wh[0]), "height":int(wh[1])}
                      and incumbent_metadata.get("camera_position_screen_cm") == camera_position
                      and incumbent_metadata.get("screen_diagonal_inches") == cfg["screen_diagonal_inches"])
        if cfg["input_source"] == "phone_udp":
            compatible = compatible and incumbent_metadata.get("camera_profile") == capture_camera_profile
        if cfg["input_source"] == "windows_camera":
            compatible = compatible and incumbent_metadata.get("windows_camera") == {
                "device_index":cfg["windows_camera_index"], "width":cfg["windows_camera_width"],
                "height":cfg["windows_camera_height"], "fov_x_degrees":cfg["windows_camera_fov_x_degrees"],
                "rotate":cfg["rotate"], "mirror":cfg["mirror"]}
        if compatible:
            item = incumbent_metadata["variants"]["conditioned_video"]
            previous_module = incumbent_metadata_path.with_name(item["module_file"])
            if not previous_module.is_file() or hashlib.sha256(previous_module.read_bytes()).hexdigest() != item.get("module_sha256"):
                raise ValueError("现有 VIDEO 模型文件无法验证，原始采集保留；请先修复现有模型")
            incumbent = torch.jit.load(str(previous_module), map_location="cpu").eval()
            incumbent_dim = int(item.get("feature_dim",148))
            incumbent_h, incumbent_p = torch.zeros(1,64), torch.zeros(1,incumbent_dim)
            shutil.copy2(previous_module, run_path / "incumbent.pt")
            shutil.copy2(incumbent_metadata_path, run_path / "incumbent-metadata.json")
    frame_images, frame_geometry, rotation, center = load_joint_frame_tensors(
        path, usable, base_item, cancelled,
    )[0]
    features, base_predictions = [], []
    example = None
    for index, row in enumerate(usable):
        if cancelled():
            raise RuntimeError("VIDEO training cancelled; capture retained")
        images, geometry = frame_images[index], frame_geometry[index]
        geometry_np = geometry.numpy()
        if images.shape != (2, 2, 36, 64) or geometry.shape != (2, 70) or not torch.isfinite(geometry).all():
            raise ValueError("invalid VIDEO eye inputs")
        with torch.no_grad():
            direction, confidence = base(images, geometry)
            if incumbent is not None:
                discontinuity = (index == 0 or row["reset"] or row["index"] != usable[index-1]["index"]+1
                                 or train_mask[index] != train_mask[index-1] or selection_mask[index] != selection_mask[index-1])
                prior = incumbent(images, geometry, torch.tensor([[row["dt_ms"]]]), torch.tensor([[float(discontinuity)]]), incumbent_h, incumbent_p)
                incumbent_h, incumbent_p = prior[2], prior[3]
                incumbent_predictions.append(prior[0])
                incumbent_weights.append(prior[1])
        feature = torch.cat((direction.flatten(), confidence.flatten(),
                             torch.from_numpy(np.clip((geometry_np - mean) / scale, -10., 10.)).flatten()))
        if appearance is not None:
            with torch.no_grad():
                feature = torch.cat((feature, appearance(images).flatten()))
        features.append(feature)
        base_predictions.append(direction)
        example = (images, geometry)
    x = torch.stack(features)
    if not torch.isfinite(x).all():
        raise ValueError("non-finite pretrained VIDEO features")
    rotation = rotation.to(torch.float32)
    center = center.to(torch.float32)
    origin, size = torch.as_tensor(origin_np, dtype=torch.float32), torch.as_tensor(size_np, dtype=torch.float32)
    target = torch.tensor([row["target"] for row in usable], dtype=torch.float32)
    dt = torch.tensor([[row["dt_ms"]] for row in usable], dtype=torch.float32)
    # A missing/invalid row must reset the next valid input, even after filtering.
    reset = torch.tensor([[float(row["reset"] or index == 0 or row["index"] != usable[index - 1]["index"] + 1)]
                          for index, row in enumerate(usable)])
    weight = torch.from_numpy(weights_np)
    confidence = x[:, 6:8]
    train_ids, validation_ids, test_ids = np.flatnonzero(train_mask), np.flatnonzero(selection_mask), np.flatnonzero(test_mask)
    adapter = VideoAdapter(feature_dim, detach_motion_state=detach_motion_state, event_motion=event_motion)
    target_np = target.numpy()
    future_pairs = joint_future_pairs(usable)
    # Bound redundant dwell/slow-drag dominance while keeping sequences intact.
    keys = [(row["phase"],row["condition"],min(5,int(row["target"][0]*6)),min(5,int(row["target"][1]*6)),
             tuple(row["constraint"]["tangent"]) if row["constraint"] else ()) for row in usable]
    counts = Counter(keys[i] for i in train_ids if weights_np[i]>0)
    reference = float(np.median(list(counts.values()))) if counts else 1.
    balance = np.array([min(2.,max(.25,(reference/max(1,counts.get(key,1)))**.5)) for key in keys],np.float32)
    np.savez_compressed(run_path / "training-inputs.npz", features=x.numpy(), target=target_np,
                        frame_index=[row["index"] for row in usable], train=train_mask,
                        selection=selection_mask, test=test_mask, weight=weights_np, balance=balance)

    def pixels(direction, ids, current_weights=None):
        weights = confidence[ids] if current_weights is None else current_weights
        return project(direction, weights, rotation[ids], center[ids], origin, size)

    mean_tensor = torch.as_tensor(mean)
    scale_tensor = torch.as_tensor(scale)

    def live_features(index):
        directions, current_weights = base(frame_images[index], frame_geometry[index])
        geometry_features = ((frame_geometry[index] - mean_tensor) / scale_tensor).clamp(-10., 10.)
        feature = torch.cat((directions.reshape(1, 6), current_weights.reshape(1, 2),
                             geometry_features.reshape(1, 140)), dim=1)
        if appearance is not None:
            feature = torch.cat((feature, appearance(frame_images[index]).reshape(1, 256)), dim=1)
        return feature, current_weights.reshape(1, 2)

    def evaluate(ids):
        hidden, previous = torch.zeros(1, 64), torch.zeros(1, feature_dim)
        predictions, gates, motions = [], [], []
        adapter.eval(); base.eval()
        with torch.no_grad():
            for position, i in enumerate(ids):
                forced = reset[i:i+1] if position and i == ids[position-1] + 1 else torch.ones(1, 1)
                feature, current_weights = live_features(i)
                direction, hidden, previous, gate, motion = adapter(feature, dt[i:i+1], forced, hidden, previous)
                predictions.append(pixels(direction, [i], current_weights)[0].numpy())
                gates.append(float(gate.item()))
                motions.append(motion[0].numpy())
        return np.array(predictions), gates, np.array(motions)

    base_xy = pixels(torch.stack(base_predictions), np.arange(len(x))).numpy()
    # Freeze the personal spatial replay before joint optimization.  Its future
    # observations supervise displacement; display targets supervise current
    # spatial calibration only.
    motion_teacher, motion_teacher_noise = joint_motion_teacher(base_xy, usable, weights_np)
    from .motion_labels import label_motion
    event_labels = label_motion(motion_teacher, usable, motion_teacher_noise)
    event_phase, event_landing = event_labels["phase"], event_labels["landing"]-motion_teacher
    event_remaining, event_valid = event_labels["remaining"], event_labels["landing_valid"]
    event_confidence = event_labels["phase_conf"]
    event_counts = np.bincount(event_phase[train_ids][event_confidence[train_ids]>=.5], minlength=3)
    event_class_weights = torch.as_tensor(np.clip(event_counts.sum()/np.maximum(1,3*event_counts),.25,5),dtype=torch.float32)
    future_pairs, future_pair_audit = balance_joint_future_pairs(
        future_pairs, motion_teacher, wh, train_ids,
    )
    eccentricity = eccentricity_degrees(target_np, center.numpy(), origin_np, size_np)
    def metrics_for(prediction, ids):
        return _metrics(prediction, target_np[ids], weights_np[ids], wh,
                        eccentricity[ids], [usable[i] for i in ids])

    baseline = metrics_for(base_xy[validation_ids], validation_ids)
    incumbent_xy, incumbent_metrics = None, None
    if incumbent is not None:
        incumbent_xy = project(torch.stack(incumbent_predictions),torch.stack(incumbent_weights),rotation,center,origin,size).numpy()
        incumbent_metrics = metrics_for(incumbent_xy[validation_ids], validation_ids)
    history, stage_best = [], []
    best_state = copy.deepcopy(adapter.state_dict())
    best_base_state = copy.deepcopy(base.state_dict())
    best_score = 1.0
    selected_stage, selected_epoch = "personal_spatial", 0
    progress("video_joint_current_temporal_prediction")
    for parameter in base.parameters():
        parameter.requires_grad_(True)
    for parameter in adapter.parameters():
        parameter.requires_grad_(True)
    optimizer = torch.optim.AdamW([
        {"params": list(base.parameters()), "lr": 1e-6 if protected_joint else 1e-5},
        {"params": list(adapter.parameters()), "lr": 3e-4},
    ], weight_decay=1e-3)
    last_improvement = 0
    for epoch in range(epochs if train_temporal else 0):
        if cancelled():
            raise RuntimeError("VIDEO training cancelled; capture retained")
        adapter.train(); base.eval() if protected_joint else base.train()
        # Learn a useful temporal head before its gradients change the visual
        # mapping. Then jointly fine-tune the CNN under current-gaze anchoring.
        for parameter in base.parameters():
            parameter.requires_grad_(not protected_joint or epoch>=2)
        epoch_ids = train_ids
        if protected_joint:
            blocks = [train_ids[start:start+64] for start in range(0,len(train_ids),64)]
            order = np.random.default_rng(9301+epoch).permutation(len(blocks))
            if joint_frame_budget:
                order = order[:max(1,int(np.ceil(joint_frame_budget/64)))]
            epoch_ids = np.concatenate([blocks[index] for index in order])
        hidden, previous = torch.zeros(1, 64), torch.zeros(1, feature_dim)
        optimizer.zero_grad()
        losses = []
        for position, i in enumerate(epoch_ids):
            if position % 32 == 0 and cancelled():
                raise RuntimeError("VIDEO training cancelled; capture retained")
            forced = reset[i:i+1] if position and i == epoch_ids[position-1] + 1 and (not protected_joint or position%64) else torch.ones(1, 1)
            feature, current_weights = live_features(i)
            direction, hidden, previous, gate, motion = adapter(
                feature, dt[i:i+1], forced, hidden, previous,
            )
            current_point = pixels(direction, [i], current_weights)
            if protected_joint:
                # Every frame anchors spatial mapping, including moving frames
                # with no stimulus-coordinate supervision. Teacher is immutable.
                losses.append(F.smooth_l1_loss(current_point,torch.as_tensor(base_xy[i:i+1]),beta=.008))
            if weight[i] > 0:
                current_loss = coordinate_loss(current_point, target[i:i+1], usable[i]["constraint"])
                losses.append(weight[i] * float(balance[i]) * current_loss)
            for future_i, horizon, pair_weight in future_pairs.get(int(i), ()):
                pursuit_delta = motion[:, :2] * horizon + .5 * motion[:, 2:4] * horizon * horizon
                if event_motion:
                    probability = motion[:,4:7].softmax(1)
                    landing_delta = motion[:,7:9]*(1-torch.exp(-3*horizon/motion[:,9:10].clamp_min(.008)))
                    reliability = (.06/motion[:,10:11].clamp_min(.003)).clamp(.25,1.)
                    predicted_delta = reliability*(probability[:,1:2]*pursuit_delta+probability[:,2:3]*landing_delta)
                else:
                    predicted_delta = pursuit_delta
                teacher_delta = torch.as_tensor(
                    motion_teacher[future_i] - motion_teacher[i], dtype=predicted_delta.dtype,
                ).reshape(1, 2)
                future_loss = F.smooth_l1_loss(predicted_delta, teacher_delta, beta=.008)
                # Per-horizon moving/stationary balance prevents long dwells
                # from making a zero-motion head the optimum.
                losses.append(JOINT_FUTURE_LOSS_WEIGHT * pair_weight * future_loss)
                if event_motion:
                    observed_error=torch.linalg.vector_norm((predicted_delta-teacher_delta).detach(),dim=1).clamp(.003,.1)
                    losses.append(.03*F.smooth_l1_loss(motion[:,10],observed_error,beta=.01))
            if event_motion:
                losses.append(.02*float(event_confidence[i])*event_class_weights[event_phase[i]]*F.cross_entropy(motion[:,4:7],torch.as_tensor([event_phase[i]])))
                if event_valid[i]:
                    losses.append(.08*F.smooth_l1_loss(motion[:,7:9],torch.as_tensor(event_landing[i:i+1]),beta=.008))
                    losses.append(.005*F.smooth_l1_loss(motion[:,9],torch.as_tensor([event_remaining[i]]),beta=.02))
            if losses and ((position + 1) % 32 == 0 or position == len(epoch_ids) - 1):
                torch.stack(losses).mean().backward()
                torch.nn.utils.clip_grad_norm_(list(base.parameters()) + list(adapter.parameters()), 1.)
                optimizer.step()
                optimizer.zero_grad()
                losses = []
            if (position + 1) % 32 == 0:
                hidden, previous = hidden.detach(), previous.detach()
        prediction, _, validation_motion = evaluate(validation_ids)
        metrics = metrics_for(prediction, validation_ids)
        future_metrics = joint_future_metrics(
            prediction, validation_motion, validation_ids, future_pairs, motion_teacher, wh,
        )
        history.append(dict(stage="joint", epoch=epoch + 1, future=future_metrics, **metrics))
        if protected_joint:
            history[-1].update(train_frames=len(epoch_ids), visual_trainable=epoch>=2)
        progress(f"joint epoch {epoch+1}/{epochs}: current mean {metrics['mean_px']:.2f} px")
        torch.save({"adapter": adapter.state_dict(), "base": base.state_dict(),
                    "optimizer": optimizer.state_dict(), "stage": "joint",
                    "epoch": epoch + 1, "metrics": metrics, "future": future_metrics,
                    "rng": torch.get_rng_state()}, run_path / f"joint-{epoch+1:03d}.pt")
        np.savez_compressed(run_path / f"joint-{epoch+1:03d}-selection.npz",
                            prediction=prediction, motion=validation_motion)
        score = joint_selection_score(metrics, baseline, future_metrics)
        history[-1]["joint_selection_score"] = score if np.isfinite(score) else None
        write_json(run_path / "history.json", history)
        if score < best_score - 1e-5:
            best_score = score
            best_state = copy.deepcopy(adapter.state_dict())
            best_base_state = copy.deepcopy(base.state_dict())
            selected_stage, selected_epoch = "joint", epoch + 1
            last_improvement = epoch + 1
        if epoch + 1 >= 12 and epoch + 1 - last_improvement >= 8:
            break
    adapter.load_state_dict(best_state)
    base.load_state_dict(best_base_state)
    adapter.eval(); base.eval()
    prediction, gates, validation_motion = evaluate(validation_ids)
    final_metrics = metrics_for(prediction, validation_ids)
    final_future_metrics = joint_future_metrics(
        prediction, validation_motion, validation_ids, future_pairs, motion_teacher, wh,
    )
    stage_best.append({"stage": "joint", "selection_score": best_score,
                       "selected_median_px": final_metrics["median_px"],
                       "improved": selected_stage == "joint"})
    report = {"baseline": baseline, "candidate": final_metrics, "selected_stage": selected_stage,
              "selected_epoch": selected_epoch, "stages": stage_best, "history": history, "early_stopping": "minimum 12 epochs per stage; stop after 8 epochs without a selection improvement",
              "base_sha256": base_digest, "base_source": metadata.get("training", {}).get("base_source", "explicit_external"),
              "public_checkpoint_sha256": metadata.get("training", {}).get("public_checkpoint_sha256"),
              "fusion_reinitialized": metadata.get("training", {}).get("fusion_reinitialized"),
              "pursuit_lag_ms": pursuit_lag_ms,
              "split": "complete validation blocks select epochs; test blocks are evaluated once after selection",
              "incumbent_selection": incumbent_metrics, "training_directory":str(run_path.resolve()),
              "feature_dim":feature_dim, "training_cpu_threads": training_threads, "alignment_policy":ALIGNMENT_POLICY,
              "rail_constraint_frames":sum(row["constraint"] is not None for row in usable),
              "reset_supervision": "synthetic substitutions only; actual saccade reset accuracy not measured",
              "temporal_training_enabled": train_temporal,
              "protected_joint": protected_joint,
              "detach_motion_state": detach_motion_state,
              "event_motion": event_motion,
              "event_landing_frames": int(event_valid.sum()),
              "event_label_audit": event_labels["audit"],
              "future_prediction": "embedded joint motion" if train_temporal else "frozen spatial model; source-time trajectory head trained by calibration pipeline",
              "future_label_policy": "offline robust reconstruction of frozen personal-spatial observations; screen stimulus remains a weak current-position anchor, not future gaze truth",
              "future_teacher_noise_normalized": motion_teacher_noise,
              "future_pair_balance": future_pair_audit,
              "joint_future_validation": final_future_metrics,
              "joint_horizons_ms": JOINT_HORIZONS_MS,
              "joint_trainable": ["personal_eye", "binocular_fusion", "current_gaze", "gru", "future_motion"],
              "selection_rule": "middle 5-15 degrees: median improves, P75 <= 1.05x, P95 <= 1.10x; >=20 exact anchors in >=2 trials; independent test release check",
              "train_frames": len(train_ids), "selection_frames": len(validation_ids),
              "coordinate_labels": int((weights_np > 0).sum()),
              "capture_frames": len(rows), "valid_input_frames": len(usable),
              "hard_resets": int(reset.sum().item()),
              "median_valid_frame_interval_ms": float(np.median([row["dt_ms"] for row in usable])),
              "clock_sources": sorted(set(row["clock"] for row in rows)),
              "published": False}
    head_pose_coverage = metadata.get("training", {}).get("head_pose_coverage")
    # V6 deliberately records fixed head directions without turning the prompt
    # into a measured-pose publication gate.
    requires_head_pose_coverage = spatial_only and any(
        step.get("head_start") for step in session.get("plan", [])
    )
    report["head_pose_coverage"] = head_pose_coverage
    report["head_pose_coverage_required"] = requires_head_pose_coverage
    if event_motion:
        report["joint_event_validation"] = joint_event_metrics(
            validation_motion, validation_ids, event_phase, event_landing, event_remaining, event_valid, wh)
    if len(test_ids) and (weights_np[test_ids]>.5).any():
        test_prediction, test_gates, test_motion = evaluate(test_ids)
        report["independent_test"] = {
            "baseline":metrics_for(base_xy[test_ids], test_ids),
            "candidate":metrics_for(test_prediction, test_ids),
            "future":joint_future_metrics(test_prediction, test_motion, test_ids, future_pairs, motion_teacher, wh),
            "incumbent":metrics_for(incumbent_xy[test_ids], test_ids) if incumbent_xy is not None else None,
            "notice":"same-session unseen blocks; release gate only, never fitting or epoch choice"}
        if event_motion:
            report["independent_test"]["event"] = joint_event_metrics(
                test_motion, test_ids, event_phase, event_landing, event_remaining, event_valid, wh)
        np.savez_compressed(run_path / "test-predictions.npz",prediction=test_prediction,target=target_np[test_ids],weight=weights_np[test_ids],gate=test_gates)
    export = VideoInference(
        base, adapter, mean, scale, appearance=appearance,
        geometry_reference_indices=reference_indices,
        geometry_reference_values=reference_values,
        geometry_reference_tolerance=reference_tolerance,
    ).eval()
    args = (*example, torch.tensor([[33.]]), torch.ones(1, 1), torch.zeros(1, 64), torch.zeros(1, feature_dim))
    # Forward contains tensor operations only; explicit state and fixed binocular
    # shapes make tracing safe, and avoid requiring Python source in frozen apps.
    scripted = torch.jit.trace(export, args, strict=False)
    with torch.no_grad():
        expected, actual = export(*args), scripted(*args)
        for a, b in zip(expected, actual):
            if not torch.allclose(a, b, atol=1e-6, rtol=1e-5):
                raise ValueError("VIDEO TorchScript export differs from eager inference")
    module_name = f"conditioned-video-{path.name}-{run_path.name}.pt"
    scripted.save(str(run_path / module_name))
    output_metadata = {key: value for key, value in metadata.items() if key != "variants"}
    output_metadata.update(created_at=datetime.now(timezone.utc).isoformat(), input_source=cfg["input_source"],
                           screen={"width": int(wh[0]), "height": int(wh[1])},
                           screen_diagonal_inches=cfg["screen_diagonal_inches"],
                           camera_position_screen_cm=camera_position,
                           screen_camera_origin_cm=origin_np.tolist(),
                           camera_profile=capture_camera_profile)
    if cfg["input_source"] == "windows_camera":
        output_metadata["windows_camera"] = {"device_index": cfg["windows_camera_index"], "width": cfg["windows_camera_width"],
                                              "height": cfg["windows_camera_height"], "fov_x_degrees": cfg["windows_camera_fov_x_degrees"],
                                              "rotate": cfg["rotate"], "mirror": cfg["mirror"]}
    output_metadata["variants"] = {"conditioned_video": {"module_file": module_name, "raw_inputs": True,
                                                          "normalization": base_item["normalization"], "temporal": True,
                                                          "selected_stage": selected_stage,
                                                          "feature_dim":feature_dim,
                                                          "joint_prediction": selected_stage == "joint",
                                                          "joint_prediction_horizons_ms": JOINT_HORIZONS_MS,
                                                          "detach_motion_state": detach_motion_state,
                                                          "joint_prediction_architecture":"event_state_v2" if event_motion else "kinematic_v1",
                                                          "runtime_geometry_reference": base_item.get("runtime_geometry_reference"),
                                                          "module_sha256": hashlib.sha256((run_path / module_name).read_bytes()).hexdigest()}}
    write_json(run_path / "conditioned-video-model.json", output_metadata)
    np.savez_compressed(run_path / "selection-predictions.npz", target=target_np[validation_ids], prediction=prediction,
                        baseline=base_xy[validation_ids], gate=gates, weight=weights_np[validation_ids])
    full_retrain = metadata.get("training", {}).get("base_source") == "public_pretrained"
    personal_selected = int(metadata.get("training", {}).get("best_epoch", 0)) > 0
    improves_incumbent = supported(final_metrics) and (incumbent_metrics is None or acceptance(final_metrics, incumbent_metrics))
    eligible = (personal_selected or selected_stage == "joint" if full_retrain else selected_stage != "pretrained") and improves_incumbent
    if selected_stage == "joint":
        test_future = (report.get("independent_test") or {}).get("future") or {}
        eligible = eligible and bool(test_future.get("samples") and test_future.get("improves_hold"))
    independent = report.get("independent_test") or {}
    eligible = eligible and acceptance(independent.get("candidate", {}),
                                      independent.get("incumbent") or independent.get("baseline"), improve=False)
    measured_head_coverage_ok = bool(
        head_pose_coverage and head_pose_coverage.get("by_split", {}).get("train", {}).get("adequate")
    )
    if requires_head_pose_coverage:
        eligible = eligible and measured_head_coverage_ok
    report["head_pose_coverage_passed"] = measured_head_coverage_ok if requires_head_pose_coverage else None
    report["incumbent_is_comparator_only"] = False
    report["publication_reason"] = "eligible" if eligible else "keep_existing_model"
    if spatial_only:
        eligible = True
        report.update(publication_reason="eligible", split="train for fitting; test for epoch selection",
                      selection_rule="minimum finite test weak-label loss; no accuracy release gate")
    if publish and eligible and not cancelled():
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copy2(run_path / module_name, DATA_DIR / module_name)
        write_json(DATA_DIR / "conditioned-video-model.json", output_metadata)
        report["published"] = True
    write_json(run_path / "training-report.json", report)
    # Stable aliases remain for existing UI/tools; all historical runs survive.
    for name in (module_name,"conditioned-video-model.json","selection-predictions.npz","training-report.json"):
        shutil.copy2(run_path / name, path / name)
    return report
