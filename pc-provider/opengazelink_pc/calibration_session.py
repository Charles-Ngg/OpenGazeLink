from __future__ import annotations

import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import threading
import time

import cv2
import numpy as np

from .config import ProviderConfig
from .model_registry import DATASET_PATHS, MODEL_PATHS, ModelRegistry
from .normalized_eye import (
    EYE_PLANE_HEIGHT_CM,
    EYE_PLANE_WIDTH_CM,
    NORMALIZED_EYE_HEIGHT,
    NORMALIZED_EYE_WIDTH,
    NormalizedEyeBackend,
    SCREEN_CAMERA_MOUNT,
    eye_in_head_angles,
    screen_camera_origin,
    target_camera_point,
)
from .paths import DATA_DIR, USER_ROOT
from .shared_eye_appearance import (
    BASE_HEIGHT,
    BASE_WIDTH,
    PREPROCESSING_MODEL,
    base_eye_image,
    serialize_base_eye,
)
from .shared_eye_models import SHARED_DATASET_SCHEMA, SharedTinyCnnModel


ROOT = USER_ROOT
LIGHTING_PROFILE_LIBRARY_PATH = DATA_DIR / "lighting-profile-library.json"
LIGHTING_PROFILE_LIBRARY_SCHEMA = "eyetracing-lighting-profile-library-v1"
_LIGHTING_LIBRARY_STATUS_CACHE: tuple[tuple, dict[str, dict]] | None = None

STATIC_SAMPLES_PER_TARGET = 9
STATIC_TARGET_DURATION_MS = 1100
STATIC_TARGET_MARGIN_PX = 18.0
POSE_SAMPLES_PER_TARGET = 9
POSE_TARGET_DURATION_MS = 1100
POSE_MINIMUM_SAMPLES = 6
CALIBRATION_SCHEMA = "crossed-head-pose-4x5-v1"
LIGHT_ANCHOR_LEVELS = (("dark", 0.22), ("bright", 0.68))


def project_relative_path(path: Path) -> str:
    """Store artifacts relative to the writable data root so it stays movable."""
    return path.resolve().relative_to(ROOT).as_posix()


def calibration_targets(width: int, height: int) -> list[dict]:
    margin_x = min(STATIC_TARGET_MARGIN_PX, max(0.0, 0.5 * (width - 1)))
    margin_y = min(STATIC_TARGET_MARGIN_PX, max(0.0, 0.5 * (height - 1)))
    xs = np.linspace(margin_x, float(width - 1) - margin_x, 5)
    ys = np.linspace(margin_y, float(height - 1) - margin_y, 5)
    grid = [
        {
            "x": float(x), "y": float(y), "row": row, "column": column,
            "grid_index": row * 5 + column,
        }
        for row, y in enumerate(ys) for column, x in enumerate(xs)
    ]
    order = [
        12, 0, 24, 4, 20, 10, 14, 2, 22, 6, 18, 8, 16,
        1, 23, 3, 21, 5, 19, 9, 15, 7, 17, 11, 13,
    ]
    targets = []
    for sequence_index, grid_index in enumerate(order):
        target = dict(grid[grid_index])
        target["sequence_index"] = sequence_index
        target["phase"] = "static"
        target["lighting"] = static_lighting_profile(sequence_index)
        targets.append(target)
    return targets


def static_lighting_profile(sequence_index: int) -> dict:
    return {
        "mode": "steady", "name": "reference-mid",
        "start": 0.42, "end": 0.42, "level": 0.42,
        "duration_ms": STATIC_TARGET_DURATION_MS,
    }


def light_anchor_targets(width: int, height: int) -> list[dict]:
    margin_x = min(STATIC_TARGET_MARGIN_PX, max(0.0, 0.5 * (width - 1)))
    margin_y = min(STATIC_TARGET_MARGIN_PX, max(0.0, 0.5 * (height - 1)))
    xs = np.linspace(margin_x, float(width - 1) - margin_x, 5)
    ys = np.linspace(margin_y, float(height - 1) - margin_y, 5)
    anchors = (
        (12, "center"), (0, "upper-left"), (4, "upper-right"),
        (24, "lower-right"), (20, "lower-left"),
    )
    targets = []
    for light_name, level in LIGHT_ANCHOR_LEVELS:
        for grid_index, name in anchors:
            row, column = divmod(grid_index, 5)
            targets.append({
                "x": float(xs[column]), "y": float(ys[row]),
                "row": row, "column": column, "grid_index": grid_index,
                "group": f"grid-{grid_index:02d}",
                "sequence_index": len(targets), "phase": "light_anchor",
                "name": name, "light_name": light_name,
                "lighting": {
                    "mode": "steady", "name": light_name,
                    "start": level, "end": level, "level": level,
                    "duration_ms": STATIC_TARGET_DURATION_MS,
                },
            })
    return targets


def pose_targets(width: int, height: int) -> list[dict]:
    margin_x = min(STATIC_TARGET_MARGIN_PX, max(0.0, 0.5 * (width - 1)))
    margin_y = min(STATIC_TARGET_MARGIN_PX, max(0.0, 0.5 * (height - 1)))
    xs = np.linspace(margin_x, float(width - 1) - margin_x, 5)
    ys = np.linspace(margin_y, float(height - 1) - margin_y, 5)
    gaze_targets = (
        (12, "center"), (0, "upper-left"), (4, "upper-right"),
        (24, "lower-right"), (20, "lower-left"),
    )
    pose_conditions = (
        ("left", "头向左转约 15°"),
        ("right", "头向右转约 15°"),
        ("up", "头向上抬约 12°"),
        ("down", "头向下低约 12°"),
    )
    steady_light = {
        "mode": "steady", "name": "pose-mid", "start": 0.42,
        "end": 0.42, "level": 0.42, "duration_ms": POSE_TARGET_DURATION_MS,
    }
    targets = []
    for pose_name, instruction in pose_conditions:
        for grid_index, gaze_name in gaze_targets:
            row, column = divmod(grid_index, 5)
            targets.append({
                "x": float(xs[column]), "y": float(ys[row]),
                "row": row, "column": column, "grid_index": grid_index,
                "pose_index": len(targets), "pose_condition": pose_name,
                "pose_instruction": instruction, "gaze_name": gaze_name,
                "name": gaze_name, "phase": "head_pose",
                "group": f"pose-{pose_name}-grid-{grid_index:02d}",
                "lighting": dict(steady_light),
            })
    return targets


def sample_payload(
    observation, target: dict, target_camera: np.ndarray,
    condition: str, pass_id: str, raw_frame: str,
    lighting: dict | None = None, pose_bin: tuple[int, int] | None = None,
) -> dict:
    group = target.get("group")
    if group is None:
        group = f"grid-{int(target['grid_index']):02d}"
    right_angles = eye_in_head_angles(
        target_camera, observation.right.eye_center_camera, observation.rotation,
    )
    left_angles = eye_in_head_angles(
        target_camera, observation.left.eye_center_camera, observation.rotation,
    )
    return {
        "t_ms": float(observation.t_ms), "target": [target["x"], target["y"]],
        "target_camera_cm": target_camera.tolist(),
        "group": group,
        "grid_index": target.get("grid_index"), "row": target.get("row"), "column": target.get("column"),
        "phase": target.get("phase", "static"), "pose_index": target.get("pose_index"),
        "pose_condition": target.get("pose_condition"),
        "gaze_name": target.get("gaze_name"),
        "pose_bin": list(pose_bin) if pose_bin is not None else None,
        "condition": condition, "pass_id": pass_id, "raw_frame": raw_frame,
        "lighting": lighting or target.get("lighting") or {"mode": "unknown"},
        "right_base_eye": serialize_base_eye(base_eye_image(observation.right, True)),
        "left_base_eye": serialize_base_eye(base_eye_image(observation.left, False)),
        "right_angles": list(right_angles), "left_angles": list(left_angles),
        "right_eye_center_camera": list(observation.right.eye_center_camera),
        "left_eye_center_camera": list(observation.left.eye_center_camera),
        "right_source_eye_size": list(observation.right.source_eye_size),
        "left_source_eye_size": list(observation.left.source_eye_size),
        "right_center_residual_px": list(observation.right.center_residual_px),
        "left_center_residual_px": list(observation.left.center_residual_px),
        "right_valid_fraction": observation.right.valid_fraction,
        "left_valid_fraction": observation.left.valid_fraction,
        "right_aperture_ratio": observation.right.aperture_ratio,
        "left_aperture_ratio": observation.left.aperture_ratio,
        "rotation": [list(row) for row in observation.rotation],
        "translation": list(observation.translation),
        "head_yaw": observation.head_yaw, "head_pitch": observation.head_pitch,
        "head_roll": observation.head_roll,
        "pnp_reprojection_error_px": observation.pnp_reprojection_error_px,
        "camera_model": observation.camera_model,
        "detection_ms": observation.detection_ms,
        "normalization_ms": observation.normalization_ms,
        "landmarker_backend": observation.landmarker_backend,
    }


def _new_dataset(config: ProviderConfig, backend: str) -> dict:
    screen_origin = screen_camera_origin(
        config.screen_width, config.screen_height,
        config.screen_diagonal_inches, config.camera_position_screen_cm,
    )
    return {
        "schema": SHARED_DATASET_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "screen": {"width": config.screen_width, "height": config.screen_height},
        "screen_diagonal_inches": config.screen_diagonal_inches,
        "screen_camera_mount": SCREEN_CAMERA_MOUNT,
        "camera_position_screen_cm": list(config.camera_position_screen_cm),
        "screen_camera_origin_cm": screen_origin.tolist(),
        "input_source": config.input_source,
        "windows_camera": {
            "device_index": config.windows_camera_index,
            "width": config.windows_camera_width,
            "height": config.windows_camera_height,
            "fps": config.windows_camera_fps,
            "backend": config.windows_camera_backend,
            "fov_x_degrees": config.windows_camera_fov_x_degrees,
            "rotate": config.rotate,
            "mirror": config.mirror,
        } if config.input_source == "windows_camera" else None,
        "target_layout": {
            "name": "dense_5x5_plus_light_anchors_plus_crossed_head_pose_v4",
            "columns": 5, "rows": 5,
            "x_bounds_px": [
                min(STATIC_TARGET_MARGIN_PX, 0.5 * (config.screen_width - 1)),
                max(STATIC_TARGET_MARGIN_PX, config.screen_width - 1 - STATIC_TARGET_MARGIN_PX),
            ],
            "y_bounds_px": [
                min(STATIC_TARGET_MARGIN_PX, 0.5 * (config.screen_height - 1)),
                max(STATIC_TARGET_MARGIN_PX, config.screen_height - 1 - STATIC_TARGET_MARGIN_PX),
            ],
            "light_anchor_targets": ["center", "upper-left", "upper-right", "lower-right", "lower-left"],
            "light_anchor_levels": {name: level for name, level in LIGHT_ANCHOR_LEVELS},
            "pose_design": {
                "conditions": ["left", "right", "up", "down"],
                "gaze_targets": ["center", "upper-left", "upper-right", "lower-right", "lower-left"],
                "samples_per_target": POSE_SAMPLES_PER_TARGET,
            },
        },
        "normalization": {
            "model": PREPROCESSING_MODEL,
            "output_size": [NORMALIZED_EYE_WIDTH, NORMALIZED_EYE_HEIGHT],
            "stored_base_eye_size": [BASE_WIDTH, BASE_HEIGHT],
            "shared_model_input_size": [BASE_WIDTH, BASE_HEIGHT],
            "photometric_feature": "masked_grayscale_shared_eye_v1",
            "plane_size_cm": [EYE_PLANE_WIDTH_CM, EYE_PLANE_HEIGHT_CM],
            "face_landmarker": (
                "MediaPipe legacy Face Mesh" if backend == "legacy"
                else "MediaPipe Tasks Face Landmarker"
            ),
            "landmarker_backend": backend,
            "landmarker_delegate": "cpu",
            "iris_landmarks_used": False,
            "eye_unification": "left image mirrored; left yaw negated during training and restored at inference",
        },
        "passes": [], "samples": [],
    }


def _save(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _commit_artifacts(artifacts: list[tuple[Path, Path]], backup_dir: Path) -> None:
    """Replace a complete trained artifact set, rolling back a partial commit."""
    backup_dir.mkdir(parents=True, exist_ok=False)
    existed = {}
    for _, destination in artifacts:
        existed[destination] = destination.exists()
        if destination.exists():
            shutil.copy2(destination, backup_dir / destination.name)
    replaced = []
    try:
        for staged, destination in artifacts:
            destination.parent.mkdir(parents=True, exist_ok=True)
            staged.replace(destination)
            replaced.append(destination)
    except Exception:
        for destination in replaced:
            backup = backup_dir / destination.name
            if existed[destination]:
                shutil.copy2(backup, destination)
            elif destination.exists():
                destination.unlink()
        raise


BUILTIN_LIGHTING_PROFILES = frozenset(("reference", "dark", "bright"))


def _sample_lighting_name(sample: dict) -> str:
    lighting = sample.get("lighting") or {}
    value = lighting.get("name") if isinstance(lighting, dict) else lighting
    return str(value or "")


def _dataset_geometry_matches(dataset: dict, config: ProviderConfig) -> bool:
    if dataset.get("input_source", "phone_udp") != config.input_source:
        return False
    if config.input_source == "windows_camera":
        calibrated_camera = dataset.get("windows_camera") or {}
        expected_camera = {
            "device_index": config.windows_camera_index,
            "width": config.windows_camera_width,
            "height": config.windows_camera_height,
            "fov_x_degrees": config.windows_camera_fov_x_degrees,
            "rotate": config.rotate,
            "mirror": config.mirror,
        }
        for key, value in expected_camera.items():
            actual = calibrated_camera.get(key)
            if isinstance(value, float):
                if actual is None or abs(float(actual) - value) > 0.01:
                    return False
            elif actual != value:
                return False
    if dataset.get("screen") != {
        "width": config.screen_width, "height": config.screen_height,
    }:
        return False
    if not np.isclose(
        float(dataset.get("screen_diagonal_inches") or 0.0),
        config.screen_diagonal_inches, atol=0.01,
    ):
        return False
    origin = dataset.get("screen_camera_origin_cm")
    expected = screen_camera_origin(
        config.screen_width, config.screen_height,
        config.screen_diagonal_inches, config.camera_position_screen_cm,
    )
    return (
        isinstance(origin, list) and len(origin) == 3
        and np.allclose(np.asarray(origin, dtype=np.float64), expected, atol=0.05)
    )


def _retained_lighting_samples(
    config: ProviderConfig, profile_names: list[str] | tuple[str, ...],
) -> dict[str, dict[str, list[dict]]]:
    names = list(dict.fromkeys(str(name) for name in profile_names if str(name)))
    if any(name in BUILTIN_LIGHTING_PROFILES for name in names):
        raise ValueError("only user-added lighting profiles can be retained")
    if not names:
        return {name: {} for name in DATASET_PATHS}
    retained: dict[str, dict[str, list[dict]]] = {}
    for backend, path in DATASET_PATHS.items():
        if not path.exists():
            raise RuntimeError("cannot retain lighting profiles before a full calibration")
        dataset = json.loads(path.read_text(encoding="utf-8"))
        if not _dataset_geometry_matches(dataset, config):
            raise RuntimeError(
                f"lighting profile data for {backend} uses different screen/camera geometry"
            )
        grouped = {name: [] for name in names}
        for sample in dataset.get("samples") or []:
            profile_name = _sample_lighting_name(sample)
            if sample.get("condition") == "lighting_anchor" and profile_name in grouped:
                grouped[profile_name].append(copy.deepcopy(sample))
        missing = [name for name, samples in grouped.items() if len(samples) < 10]
        if missing:
            raise RuntimeError(
                f"lighting profile data is incomplete for {backend}: {', '.join(missing)}"
            )
        retained[backend] = grouped
    return retained


def _profile_records_from_datasets(
    datasets: dict[str, dict], config: ProviderConfig,
) -> dict[str, dict]:
    if set(datasets) != set(DATASET_PATHS):
        return {}
    if any(not _dataset_geometry_matches(dataset, config) for dataset in datasets.values()):
        return {}
    grouped: dict[str, dict[str, list[dict]]] = {
        backend: {} for backend in DATASET_PATHS
    }
    for backend, dataset in datasets.items():
        for sample in dataset.get("samples") or []:
            profile_name = _sample_lighting_name(sample)
            if (
                sample.get("condition") == "lighting_anchor"
                and profile_name not in BUILTIN_LIGHTING_PROFILES
                and profile_name
            ):
                grouped[backend].setdefault(profile_name, []).append(copy.deepcopy(sample))
    names = set.intersection(*(set(values) for values in grouped.values()))
    reference = datasets["legacy"]
    records = {}
    for name in sorted(names):
        if any(len(grouped[backend][name]) < 10 for backend in DATASET_PATHS):
            continue
        records[name] = {
            "screen": copy.deepcopy(reference.get("screen")),
            "screen_diagonal_inches": reference.get("screen_diagonal_inches"),
            "screen_camera_origin_cm": copy.deepcopy(reference.get("screen_camera_origin_cm")),
            "camera_position_screen_cm": copy.deepcopy(reference.get("camera_position_screen_cm")),
            "input_source": reference.get("input_source", "phone_udp"),
            "windows_camera": copy.deepcopy(reference.get("windows_camera")),
            "samples": {
                backend: grouped[backend][name] for backend in DATASET_PATHS
            },
        }
    return records


def _load_dataset_pair(paths: dict[str, Path]) -> dict[str, dict] | None:
    if any(not path.exists() for path in paths.values()):
        return None
    try:
        return {
            backend: json.loads(path.read_text(encoding="utf-8"))
            for backend, path in paths.items()
        }
    except (OSError, ValueError, TypeError):
        return None


def _seed_lighting_profile_library(config: ProviderConfig) -> dict:
    profiles: dict[str, dict] = {}
    sources = [dict(DATASET_PATHS)]
    calibration_root = DATA_DIR / "calibration-frames"
    if calibration_root.exists():
        backups = sorted(
            calibration_root.glob("*/pre-commit-backup"),
            key=lambda path: path.stat().st_mtime_ns,
            reverse=True,
        )
        sources.extend({
            backend: backup / path.name for backend, path in DATASET_PATHS.items()
        } for backup in backups)
    for source in sources:
        datasets = _load_dataset_pair(source)
        if datasets is None:
            continue
        for name, record in _profile_records_from_datasets(datasets, config).items():
            profiles.setdefault(name, record)
    payload = {
        "schema": LIGHTING_PROFILE_LIBRARY_SCHEMA,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "profiles": profiles,
    }
    _save(LIGHTING_PROFILE_LIBRARY_PATH, payload)
    return payload


def _load_lighting_profile_library(config: ProviderConfig) -> dict:
    if not LIGHTING_PROFILE_LIBRARY_PATH.exists():
        return _seed_lighting_profile_library(config)
    payload = json.loads(LIGHTING_PROFILE_LIBRARY_PATH.read_text(encoding="utf-8"))
    if payload.get("schema") != LIGHTING_PROFILE_LIBRARY_SCHEMA:
        raise RuntimeError("unsupported lighting profile library")
    payload.setdefault("profiles", {})
    return payload


def _library_record_matches(record: dict, config: ProviderConfig) -> bool:
    return _dataset_geometry_matches({
        "screen": record.get("screen"),
        "screen_diagonal_inches": record.get("screen_diagonal_inches"),
        "screen_camera_origin_cm": record.get("screen_camera_origin_cm"),
        "input_source": record.get("input_source", "phone_udp"),
        "windows_camera": record.get("windows_camera"),
    }, config)


def lighting_profile_library_status(config: ProviderConfig) -> dict[str, dict]:
    global _LIGHTING_LIBRARY_STATUS_CACHE
    if not LIGHTING_PROFILE_LIBRARY_PATH.exists():
        _seed_lighting_profile_library(config)
    stat = LIGHTING_PROFILE_LIBRARY_PATH.stat()
    cache_key = (
        stat.st_mtime_ns, stat.st_size,
        config.screen_width, config.screen_height, config.screen_diagonal_inches,
        *config.camera_position_screen_cm,
        config.input_source, config.windows_camera_index,
        config.windows_camera_width, config.windows_camera_height,
        config.windows_camera_fov_x_degrees, config.rotate, config.mirror,
    )
    if (
        _LIGHTING_LIBRARY_STATUS_CACHE is not None
        and _LIGHTING_LIBRARY_STATUS_CACHE[0] == cache_key
    ):
        return copy.deepcopy(_LIGHTING_LIBRARY_STATUS_CACHE[1])
    payload = _load_lighting_profile_library(config)
    status = {}
    for name, record in (payload.get("profiles") or {}).items():
        samples = record.get("samples") or {}
        geometry_matches = _library_record_matches(record, config)
        status[name] = {
            "sample_frames": {
                backend: len(samples.get(backend) or []) for backend in DATASET_PATHS
            },
            "geometry_matches": geometry_matches,
            "reusable": bool(
                geometry_matches
                and all(len(samples.get(backend) or []) >= 10 for backend in DATASET_PATHS)
            ),
        }
    _LIGHTING_LIBRARY_STATUS_CACHE = (cache_key, copy.deepcopy(status))
    return status


def _update_lighting_profile_library(
    config: ProviderConfig, datasets: dict[str, dict], profile_names: list[str],
) -> None:
    payload = _load_lighting_profile_library(config)
    records = _profile_records_from_datasets(datasets, config)
    profiles = dict(payload.get("profiles") or {})
    for name in profile_names:
        if name in records:
            profiles[name] = records[name]
    payload.update({
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "profiles": profiles,
    })
    _save(LIGHTING_PROFILE_LIBRARY_PATH, payload)


def delete_lighting_profile(profile_name: str) -> dict:
    name = str(profile_name or "").strip()
    if not name or name == "reference":
        raise ValueError("reference lighting profile cannot be deleted")
    operation_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S-%fZ-delete")
    operation_dir = DATA_DIR / "lighting-profile-operations" / operation_id
    staging_dir = operation_dir / "staged"
    staging_dir.mkdir(parents=True, exist_ok=False)
    artifacts: list[tuple[Path, Path]] = []
    removed_samples: dict[str, int] = {}
    found = False
    for backend, dataset_path in DATASET_PATHS.items():
        if not dataset_path.exists():
            continue
        dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
        original_samples = list(dataset.get("samples") or [])
        kept_samples = [
            sample for sample in original_samples
            if not (
                sample.get("condition") == "lighting_anchor"
                and _sample_lighting_name(sample) == name
            )
        ]
        removed_samples[backend] = len(original_samples) - len(kept_samples)
        if removed_samples[backend]:
            found = True
        dataset["samples"] = kept_samples
        dataset["passes"] = [
            item for item in dataset.get("passes") or []
            if not (
                item.get("condition") == "lighting_adaptation"
                and str(item.get("profile_name") or "") == name
            )
        ]
        staged = staging_dir / dataset_path.name
        _save(staged, dataset)
        artifacts.append((staged, dataset_path))

    for backend in ("legacy", "tasks"):
        metadata_path, module_path = MODEL_PATHS[backend]
        if not metadata_path.exists():
            continue
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        profiles = dict(metadata.get("lighting_profiles") or {})
        diagnostics = dict((metadata.get("diagnostics") or {}).get("lighting_adapters") or {})
        if name in profiles:
            found = True
            profiles.pop(name, None)
            diagnostics.pop(name, None)
            metadata["lighting_profiles"] = profiles
            metadata.setdefault("diagnostics", {})["lighting_adapters"] = diagnostics
            staged = staging_dir / metadata_path.name
            _save(staged, metadata)
            artifacts.append((staged, metadata_path))
    if LIGHTING_PROFILE_LIBRARY_PATH.exists():
        library = json.loads(LIGHTING_PROFILE_LIBRARY_PATH.read_text(encoding="utf-8"))
        profiles = dict(library.get("profiles") or {})
        if name in profiles:
            found = True
            profiles.pop(name, None)
            library["profiles"] = profiles
            library["updated_at"] = datetime.now(timezone.utc).isoformat()
            staged = staging_dir / LIGHTING_PROFILE_LIBRARY_PATH.name
            _save(staged, library)
            artifacts.append((staged, LIGHTING_PROFILE_LIBRARY_PATH))
    if not found:
        shutil.rmtree(operation_dir, ignore_errors=True)
        raise ValueError(f"lighting profile not found: {name}")
    _commit_artifacts(artifacts, operation_dir / "backup")
    _save(operation_dir / "result.json", {
        "schema": "eyetracing-lighting-profile-delete-v1",
        "profile_name": name,
        "removed_samples": removed_samples,
        "raw_capture_archives_retained": True,
        "finished_at": datetime.now(timezone.utc).isoformat(),
    })
    return {"profile_name": name, "removed_samples": removed_samples}


def retrain_lighting_profiles(
    config: ProviderConfig, profile_names: list[str],
) -> dict:
    selected = list(dict.fromkeys(str(name) for name in profile_names if str(name)))
    if not selected:
        return {"profile_names": [], "diagnostics": {}}
    library = _load_lighting_profile_library(config)
    profiles = library.get("profiles") or {}
    unknown = [
        name for name in selected
        if name not in profiles or not _library_record_matches(profiles[name], config)
    ]
    if unknown:
        raise ValueError(f"lighting profiles are not reusable: {', '.join(unknown)}")
    operation_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S-%fZ-retrain")
    operation_dir = DATA_DIR / "lighting-profile-operations" / operation_id
    staging_dir = operation_dir / "staged"
    staging_dir.mkdir(parents=True, exist_ok=False)
    artifacts: list[tuple[Path, Path]] = []
    diagnostics = {}
    try:
        for backend, dataset_path in DATASET_PATHS.items():
            if not dataset_path.exists():
                raise RuntimeError("run a full calibration before restoring lighting profiles")
            dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
            if not _dataset_geometry_matches(dataset, config):
                raise RuntimeError(
                    f"lighting profile data for {backend} uses different screen/camera geometry"
                )
            dataset["samples"] = [
                sample for sample in dataset.get("samples") or []
                if not (
                    sample.get("condition") == "lighting_anchor"
                    and _sample_lighting_name(sample) in selected
                )
            ]
            dataset["passes"] = [
                item for item in dataset.get("passes") or []
                if not (
                    item.get("condition") in (
                        "lighting_adaptation", "retained_lighting_adaptation",
                    )
                    and str(item.get("profile_name") or "") in selected
                )
            ]
            for profile_name in selected:
                samples = copy.deepcopy(
                    ((profiles[profile_name].get("samples") or {}).get(backend) or [])
                )
                if len(samples) < 10:
                    raise RuntimeError(
                        f"lighting profile {profile_name} is incomplete for {backend}"
                    )
                dataset["samples"].extend(samples)
                dataset["passes"].append({
                    "id": f"{operation_id}-{profile_name}",
                    "condition": "retained_lighting_adaptation",
                    "profile_name": profile_name,
                    "source_samples": len(samples),
                    "started_at": datetime.now(timezone.utc).isoformat(),
                    "finished_at": datetime.now(timezone.utc).isoformat(),
                })
            staged_dataset = staging_dir / dataset_path.name
            _save(staged_dataset, dataset)
            artifacts.append((staged_dataset, dataset_path))

            metadata_path, module_path = MODEL_PATHS[backend]
            if not metadata_path.exists() or module_path is None or not module_path.exists():
                raise RuntimeError(f"CNN model is missing for {backend}")
            staged_metadata = staging_dir / metadata_path.name
            staged_module = staging_dir / module_path.name
            shutil.copy2(metadata_path, staged_metadata)
            shutil.copy2(module_path, staged_module)
            diagnostics[backend] = {}
            for profile_name in selected:
                _, profile_diagnostics = SharedTinyCnnModel.fit_lighting_profile(
                    dataset, staged_metadata, staged_module, profile_name,
                )
                diagnostics[backend][profile_name] = profile_diagnostics
            SharedTinyCnnModel.load(staged_metadata, staged_module)
            artifacts.append((staged_metadata, metadata_path))
        _commit_artifacts(artifacts, operation_dir / "backup")
        _save(operation_dir / "result.json", {
            "schema": "eyetracing-lighting-profile-retrain-v1",
            "profile_names": selected,
            "diagnostics": diagnostics,
            "finished_at": datetime.now(timezone.utc).isoformat(),
        })
        return {"profile_names": selected, "diagnostics": diagnostics}
    except Exception:
        if not artifacts:
            shutil.rmtree(operation_dir, ignore_errors=True)
        raise


class CalibrationSession:
    def __init__(
        self, camera, config: ProviderConfig, registry: ModelRegistry,
        samples_per_target: int = STATIC_SAMPLES_PER_TARGET,
    ) -> None:
        self.camera = camera
        self.config = config
        self.registry = registry
        self.samples_per_target = samples_per_target
        self.targets = calibration_targets(config.screen_width, config.screen_height)
        self.light_targets = light_anchor_targets(config.screen_width, config.screen_height)
        self.pose_targets = pose_targets(config.screen_width, config.screen_height)
        self.index = 0
        self.light_index = 0
        self.pose_index = 0
        self.phase = "static"
        self.state = "collecting"
        self.error = ""
        self._lock = threading.Lock()
        self._last_sequence = -1
        self.pass_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S-%fZ")
        self.capture_dir = DATA_DIR / "calibration-frames" / self.pass_id
        self.raw_dir = self.capture_dir / "raw-frames"
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        library = _load_lighting_profile_library(config)
        self._lighting_profile_candidates = {
            name: copy.deepcopy(record)
            for name, record in (library.get("profiles") or {}).items()
            if _library_record_matches(record, config)
            and all(
                len((record.get("samples") or {}).get(backend) or []) >= 10
                for backend in DATASET_PATHS
            )
        }
        self._staging_dir: Path | None = None
        self._staged_artifacts: list[tuple[Path, Path]] = []
        self._diagnostics: dict = {}
        self.backends = {name: NormalizedEyeBackend(name) for name in ("legacy", "tasks")}
        self.datasets = {name: _new_dataset(config, name) for name in self.backends}
        for dataset in self.datasets.values():
            dataset["passes"].append({
                "id": self.pass_id, "condition": "fixed_head",
                "started_at": datetime.now(timezone.utc).isoformat(),
                "samples_per_target": samples_per_target,
                "completed_targets": 0,
                "paired_backends": ["legacy", "tasks"],
            })
            dataset["passes"].append({
                "id": self.pass_id + "-light-anchors", "condition": "lighting_anchors",
                "started_at": None, "completed_targets": 0,
                "target_count": len(self.light_targets),
                "lighting_levels": {name: level for name, level in LIGHT_ANCHOR_LEVELS},
            })
            dataset["passes"].append({
                "id": self.pass_id + "-head-pose", "condition": "crossed_head_pose_gaze",
                "started_at": None, "completed_targets": 0,
                "pose_conditions": ["left", "right", "up", "down"],
                "gaze_targets": ["center", "upper-left", "upper-right", "lower-right", "lower-left"],
                "target_count": len(self.pose_targets),
                "samples_per_target": POSE_SAMPLES_PER_TARGET,
            })

    def _candidate_status(self) -> list[dict]:
        return [
            {
                "name": name,
                "sample_frames": {
                    backend: len((record.get("samples") or {}).get(backend) or [])
                    for backend in DATASET_PATHS
                },
            }
            for name, record in sorted(self._lighting_profile_candidates.items())
        ]

    def _partial_path(self, backend: str) -> Path:
        return self.capture_dir / f"partial-{backend}-dataset.json"

    def status(self) -> dict:
        return {
            "active": self.state in ("collecting", "training", "awaiting_profiles"),
            "state": self.state, "phase": self.phase,
            "index": self.index, "total": len(self.targets),
            "light_index": self.light_index, "light_total": len(self.light_targets),
            "pose_index": self.pose_index, "pose_total": len(self.pose_targets),
            "available_lighting_profiles": self._candidate_status(),
            "samples_per_target": self.samples_per_target,
            "error": self.error,
        }

    @staticmethod
    def _opening(observations: dict) -> float:
        return float(min(
            observations["legacy"].right.aperture_ratio,
            observations["legacy"].left.aperture_ratio,
            observations["tasks"].right.aperture_ratio,
            observations["tasks"].left.aperture_ratio,
        ))

    def _blink_like(self, observations: dict, history: list[float]) -> bool:
        current = self._opening(observations)
        if current < 0.18:
            return True
        if len(history) < 8:
            history.append(current)
            return False
        baseline = float(np.median(history[-30:]))
        if current < max(0.03, baseline * 0.55):
            return True
        history.append(current)
        del history[:-30]
        return False

    @staticmethod
    def _candidate_score(observations: dict, history: list[float]) -> float:
        opening = CalibrationSession._opening(observations)
        baseline = float(np.median(history[-20:])) if history else opening
        opening_score = -abs(opening / max(baseline, 1e-6) - 1.0)
        reprojection = sum(float(value.pnp_reprojection_error_px) for value in observations.values())
        return opening_score - 0.01 * reprojection

    def _predict(self, frame, t_ms: float, camera_model: dict) -> dict | None:
        observations = {
            name: backend.predict(frame, t_ms, camera_model)
            for name, backend in self.backends.items()
        }
        return None if any(value is None for value in observations.values()) else observations

    def _save_observation(
        self, frame, observations: dict, target: dict, suffix: str,
        condition: str, lighting: dict | None = None,
        pose_bin: tuple[int, int] | None = None,
    ) -> None:
        raw_path = self.raw_dir / f"{suffix}.png"
        if not cv2.imwrite(str(raw_path), frame, [cv2.IMWRITE_PNG_COMPRESSION, 1]):
            raise RuntimeError(f"failed to save {raw_path}")
        target_camera = target_camera_point(
            (target["x"], target["y"]), self.config.screen_width,
            self.config.screen_height, self.config.screen_diagonal_inches,
            screen_camera_origin(
                self.config.screen_width, self.config.screen_height,
                self.config.screen_diagonal_inches,
                self.config.camera_position_screen_cm,
            ),
        )
        for name, observation in observations.items():
            self.datasets[name]["samples"].append(sample_payload(
                observation, target, target_camera, condition, self.pass_id,
                project_relative_path(raw_path), lighting, pose_bin,
            ))

    def _save_partials(self) -> None:
        for name, dataset in self.datasets.items():
            dataset["passes"][0]["completed_targets"] = self.index
            dataset["passes"][1]["completed_targets"] = self.light_index
            dataset["passes"][2]["completed_targets"] = self.pose_index
            _save(self._partial_path(name), dataset)

    def capture_next(self, expected_index: int) -> dict:
        with self._lock:
            if self.state != "collecting" or self.phase not in ("static", "light_anchor"):
                raise RuntimeError(f"calibration phase is {self.phase}")
            is_static = self.phase == "static"
            current_index = self.index if is_static else self.light_index
            targets = self.targets if is_static else self.light_targets
            if expected_index != current_index:
                raise ValueError(f"expected target {current_index}, received {expected_index}")
            target = targets[current_index]
            profile = target["lighting"]
            duration_s = float(profile["duration_ms"]) / 1000.0
            started = time.monotonic()
            deadline = started + duration_s
            slots: list[tuple[np.ndarray, dict, float] | None] = [None] * self.samples_per_target
            opening_history: list[float] = []
            rejected_blinks = 0
            while time.monotonic() < deadline:
                ok, frame, t_ms, sequence = self.camera.read_latest(
                    self._last_sequence, timeout_s=0.25,
                )
                if not ok or frame is None:
                    continue
                self._last_sequence = sequence
                camera_model = self.camera.camera_model()
                if camera_model.get("source") == "estimated_frame_center":
                    continue
                observations = self._predict(frame, t_ms, camera_model)
                if observations is None:
                    continue
                if self._blink_like(observations, opening_history):
                    rejected_blinks += 1
                    continue
                progress = min(0.999999, max(0.0, (time.monotonic() - started) / duration_s))
                slot = min(self.samples_per_target - 1, int(progress * self.samples_per_target))
                score = self._candidate_score(observations, opening_history)
                current = slots[slot]
                if current is None or score > current[2]:
                    slots[slot] = (frame.copy(), observations, score)
            selected = [value for value in slots if value is not None]
            minimum_samples = max(8, self.samples_per_target - 2)
            if len(selected) < minimum_samples:
                raise RuntimeError(
                    f"not enough paired landmark frames: {len(selected)}/{minimum_samples}"
                )
            for sample_index, (frame, observations, _) in enumerate(selected):
                prefix = "grid" if is_static else f"light-{target['light_name']}"
                self._save_observation(
                    frame, observations, target,
                    f"{prefix}-grid-{target['grid_index']:02d}-sample-{sample_index:02d}",
                    "fixed_head" if is_static else "lighting_anchor", profile,
                )
            if is_static:
                self.index += 1
            else:
                self.light_index += 1
            if is_static and self.index == len(self.targets):
                self.phase = "light_anchor"
                for dataset in self.datasets.values():
                    dataset["passes"][1]["started_at"] = datetime.now(timezone.utc).isoformat()
            elif not is_static and self.light_index == len(self.light_targets):
                self.phase = "head_pose"
                for dataset in self.datasets.values():
                    dataset["passes"][2]["started_at"] = datetime.now(timezone.utc).isoformat()
            self._save_partials()
            return {
                "ok": True, "phase": self.phase, "index": self.index,
                "light_index": self.light_index,
                "accepted": len(selected), "rejected_blinks": rejected_blinks,
            }

    def capture_pose(self, expected_index: int) -> dict:
        with self._lock:
            if self.state != "collecting" or self.phase != "head_pose":
                raise RuntimeError(f"calibration phase is {self.phase}")
            if expected_index != self.pose_index:
                raise ValueError(f"expected pose target {self.pose_index}, received {expected_index}")
            target = {
                **self.pose_targets[self.pose_index],
            }
            duration_s = float(target["lighting"]["duration_ms"]) / 1000.0
            started = time.monotonic()
            deadline = started + duration_s
            slots: list[tuple[np.ndarray, dict, float, tuple[int, int]] | None] = [
                None
            ] * POSE_SAMPLES_PER_TARGET
            opening_history: list[float] = []
            rejected_blinks = 0
            while time.monotonic() < deadline:
                ok, frame, t_ms, sequence = self.camera.read_latest(
                    self._last_sequence, timeout_s=0.25,
                )
                if not ok or frame is None:
                    continue
                self._last_sequence = sequence
                camera_model = self.camera.camera_model()
                if camera_model.get("source") == "estimated_frame_center":
                    continue
                observations = self._predict(frame, t_ms, camera_model)
                if observations is None:
                    continue
                if self._blink_like(observations, opening_history):
                    rejected_blinks += 1
                    continue
                reference = observations["tasks"]
                pose_bin = (
                    int(round(float(np.degrees(reference.head_yaw)) / 5.0)),
                    int(round(float(np.degrees(reference.head_pitch)) / 4.0)),
                )
                progress = min(
                    0.999999,
                    max(0.0, (time.monotonic() - started) / duration_s),
                )
                slot = min(
                    POSE_SAMPLES_PER_TARGET - 1,
                    int(progress * POSE_SAMPLES_PER_TARGET),
                )
                score = self._candidate_score(observations, opening_history)
                current = slots[slot]
                if current is None or score > current[2]:
                    slots[slot] = (frame.copy(), observations, score, pose_bin)
            selected = [value for value in slots if value is not None]
            minimum_samples = min(POSE_SAMPLES_PER_TARGET, POSE_MINIMUM_SAMPLES)
            if len(selected) < minimum_samples:
                raise RuntimeError(
                    f"not enough crossed head-pose frames: {len(selected)}/{minimum_samples}"
                )
            measured_pose = []
            for sample_index, (frame, observations, _, pose_bin) in enumerate(selected):
                self._save_observation(
                    frame, observations, target,
                    f"pose-{target['pose_condition']}-grid-{target['grid_index']:02d}-sample-{sample_index:02d}",
                    "head_pose_cross", target["lighting"], pose_bin,
                )
                reference = observations["tasks"]
                measured_pose.append([
                    float(np.degrees(reference.head_yaw)),
                    float(np.degrees(reference.head_pitch)),
                    float(np.degrees(reference.head_roll)),
                ])
            self.pose_index += 1
            completed_pose_group = (
                self.pose_index == len(self.pose_targets)
                or self.pose_targets[self.pose_index]["pose_condition"] != target["pose_condition"]
            )
            if completed_pose_group:
                self._save_partials()
            median_pose = np.median(np.asarray(measured_pose, dtype=np.float64), axis=0)
            return {
                "ok": True, "phase": self.phase, "pose_index": self.pose_index,
                "accepted": len(selected), "rejected_blinks": rejected_blinks,
                "pose_condition": target["pose_condition"],
                "measured_head_pose_deg": median_pose.tolist(),
            }

    def finish(self) -> dict:
        with self._lock:
            if self.index != len(self.targets):
                raise RuntimeError(f"calibration has only {self.index}/{len(self.targets)} targets")
            if self.light_index != len(self.light_targets):
                raise RuntimeError(
                    f"lighting calibration has only {self.light_index}/{len(self.light_targets)} targets"
                )
            if self.pose_index != len(self.pose_targets):
                raise RuntimeError(
                    f"head-pose calibration has only {self.pose_index}/{len(self.pose_targets)} targets"
                )
            self.phase = "training"
            self.state = "training"
        try:
            staging_dir = self.capture_dir / "trained-artifacts"
            staging_dir.mkdir(parents=True, exist_ok=False)
            artifacts = []
            for name, dataset in self.datasets.items():
                finished_at = datetime.now(timezone.utc).isoformat()
                dataset["passes"][0].update({
                    "finished_at": finished_at,
                    "cancelled": False,
                })
                dataset["passes"][1].update({
                    "finished_at": finished_at,
                    "cancelled": False,
                })
                dataset["passes"][2].update({
                    "finished_at": finished_at,
                    "cancelled": False,
                })
                _save(self._partial_path(name), dataset)
                dataset_path = DATASET_PATHS[name]
                cnn_path, module_path = MODEL_PATHS[name]
                staged_dataset = staging_dir / dataset_path.name
                staged_cnn = staging_dir / cnn_path.name
                staged_module = staging_dir / module_path.name
                _save(staged_dataset, dataset)
                _, cnn_diagnostics = SharedTinyCnnModel.fit_dataset(
                    dataset, staged_cnn, staged_module,
                )
                SharedTinyCnnModel.load(staged_cnn, staged_module)
                artifacts.extend((
                    (staged_dataset, dataset_path),
                    (staged_cnn, cnn_path),
                    (staged_module, module_path),
                ))
                self._diagnostics[name] = cnn_diagnostics
            self._staging_dir = staging_dir
            self._staged_artifacts = artifacts
            self._close_backends()
            self.phase = "lighting_profile_selection"
            self.state = "awaiting_profiles"
            if self._lighting_profile_candidates:
                return {
                    "ok": True,
                    "requires_lighting_profile_selection": True,
                    "available_lighting_profiles": self._candidate_status(),
                    "diagnostics": self._diagnostics,
                }
            return self.complete_lighting_profiles([])
        except Exception as error:
            self.state = "error"
            self.error = str(error)
            raise
        finally:
            if self.state not in ("awaiting_profiles", "complete"):
                self._close_backends()

    def complete_lighting_profiles(self, selected_profiles: list[str]) -> dict:
        selected = list(dict.fromkeys(str(name) for name in selected_profiles if str(name)))
        with self._lock:
            if self.state != "awaiting_profiles":
                raise RuntimeError(f"calibration phase is {self.phase}")
            unknown = [
                name for name in selected if name not in self._lighting_profile_candidates
            ]
            if unknown:
                raise ValueError(f"lighting profiles are not reusable: {', '.join(unknown)}")
            if self._staging_dir is None or not self._staged_artifacts:
                raise RuntimeError("trained calibration artifacts are unavailable")
            self.phase = "lighting_profile_training"
            self.state = "training"
        try:
            adapter_diagnostics = {}
            for backend, dataset in self.datasets.items():
                for profile_name in selected:
                    record = self._lighting_profile_candidates[profile_name]
                    samples = copy.deepcopy((record.get("samples") or {}).get(backend) or [])
                    dataset["samples"].extend(samples)
                    dataset["passes"].append({
                        "id": f"{self.pass_id}-retained-{profile_name}",
                        "condition": "retained_lighting_adaptation",
                        "profile_name": profile_name,
                        "source_samples": len(samples),
                        "started_at": datetime.now(timezone.utc).isoformat(),
                        "finished_at": datetime.now(timezone.utc).isoformat(),
                    })
                dataset_path = DATASET_PATHS[backend]
                staged_dataset = self._staging_dir / dataset_path.name
                _save(staged_dataset, dataset)
                cnn_path, module_path = MODEL_PATHS[backend]
                staged_cnn = self._staging_dir / cnn_path.name
                staged_module = self._staging_dir / module_path.name
                adapter_diagnostics[backend] = {}
                for profile_name in selected:
                    _, profile_diagnostics = SharedTinyCnnModel.fit_lighting_profile(
                        dataset, staged_cnn, staged_module, profile_name,
                    )
                    adapter_diagnostics[backend][profile_name] = profile_diagnostics
                SharedTinyCnnModel.load(staged_cnn, staged_module)

            library_payload = {
                "schema": LIGHTING_PROFILE_LIBRARY_SCHEMA,
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "profiles": {
                    name: self._lighting_profile_candidates[name] for name in selected
                },
            }
            staged_library = self._staging_dir / LIGHTING_PROFILE_LIBRARY_PATH.name
            _save(staged_library, library_payload)
            artifacts = [
                *self._staged_artifacts,
                (staged_library, LIGHTING_PROFILE_LIBRARY_PATH),
            ]
            _commit_artifacts(artifacts, self.capture_dir / "pre-commit-backup")
            self.registry.clear()
            self._diagnostics["retained_lighting_adapters"] = adapter_diagnostics
            self.phase = "complete"
            self.state = "complete"
            result = {
                "ok": True,
                "retained_lighting_profiles": selected,
                "diagnostics": self._diagnostics,
            }
            _save(self.capture_dir / "completed.json", {
                "schema": "eyetracing-full-calibration-result-v1",
                "pass_id": self.pass_id,
                "finished_at": datetime.now(timezone.utc).isoformat(),
                **result,
            })
            return result
        except Exception as error:
            self.state = "error"
            self.error = str(error)
            raise

    def cancel(self) -> None:
        self.phase = "cancelled"
        self.state = "cancelled"
        for name, dataset in self.datasets.items():
            dataset["passes"][0].update({
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "cancelled": True,
            })
            dataset["passes"][1].update({
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "cancelled": True,
            })
            dataset["passes"][2].update({
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "cancelled": True,
            })
            _save(self._partial_path(name), dataset)
        self._close_backends()

    def _close_backends(self) -> None:
        for backend in self.backends.values():
            backend.close()
        self.backends.clear()


class LightingAdaptationSession(CalibrationSession):
    def __init__(
        self, camera, config: ProviderConfig, registry: ModelRegistry,
        profile_name: str, screen_level: float = 0.42,
    ) -> None:
        self.camera = camera
        self.config = config
        self.registry = registry
        cleaned = "".join(
            character if character.isalnum() or character in "-_" else "-"
            for character in profile_name.strip()
        ).strip("-")
        self.profile_name = cleaned or datetime.now(timezone.utc).strftime("light-%Y%m%d-%H%M%S")
        self.screen_level = float(np.clip(screen_level, 0.05, 1.0))
        self.samples_per_target = 6
        self.targets = []
        for target in light_anchor_targets(config.screen_width, config.screen_height)[:5]:
            item = dict(target)
            item["phase"] = "light_adaptation"
            item["light_name"] = self.profile_name
            item["lighting"] = {
                "mode": "steady", "name": self.profile_name,
                "start": self.screen_level, "end": self.screen_level,
                "level": self.screen_level, "duration_ms": 900,
            }
            self.targets.append(item)
        self.index = 0
        self.phase = "light_adaptation"
        self.state = "collecting"
        self.error = ""
        self._lock = threading.Lock()
        self._last_sequence = -1
        self.pass_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S-%fZ-light")
        self.capture_dir = DATA_DIR / "lighting-adaptation-frames" / self.pass_id
        self.raw_dir = self.capture_dir / "raw-frames"
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.backends = {name: NormalizedEyeBackend(name) for name in ("legacy", "tasks")}
        self.datasets = {}
        for name, path in DATASET_PATHS.items():
            if not path.exists():
                self._close_backends()
                raise RuntimeError("run a full calibration before adding a lighting profile")
            dataset = json.loads(path.read_text(encoding="utf-8"))
            dataset.setdefault("passes", []).append({
                "id": self.pass_id, "condition": "lighting_adaptation",
                "profile_name": self.profile_name, "screen_level": self.screen_level,
                "started_at": datetime.now(timezone.utc).isoformat(),
                "samples_per_target": self.samples_per_target,
                "completed_targets": 0,
            })
            self.datasets[name] = dataset

    def status(self) -> dict:
        return {
            "active": self.state in ("collecting", "training"),
            "state": self.state, "phase": self.phase,
            "index": self.index, "total": len(self.targets),
            "profile_name": self.profile_name,
            "samples_per_target": self.samples_per_target,
            "error": self.error,
        }

    def _save_partials(self) -> None:
        for name, dataset in self.datasets.items():
            dataset["passes"][-1]["completed_targets"] = self.index
            _save(self.capture_dir / f"partial-{name}-dataset.json", dataset)

    def capture_next(self, expected_index: int) -> dict:
        with self._lock:
            if self.state != "collecting" or self.phase != "light_adaptation":
                raise RuntimeError(f"lighting adaptation phase is {self.phase}")
            if expected_index != self.index:
                raise ValueError(f"expected target {self.index}, received {expected_index}")
            target = self.targets[self.index]
            duration_s = float(target["lighting"]["duration_ms"]) / 1000.0
            started = time.monotonic()
            deadline = started + duration_s
            slots: list[tuple[np.ndarray, dict, float] | None] = [None] * self.samples_per_target
            opening_history: list[float] = []
            rejected_blinks = 0
            while time.monotonic() < deadline:
                ok, frame, t_ms, sequence = self.camera.read_latest(
                    self._last_sequence, timeout_s=0.25,
                )
                if not ok or frame is None:
                    continue
                self._last_sequence = sequence
                camera_model = self.camera.camera_model()
                if camera_model.get("source") == "estimated_frame_center":
                    continue
                observations = self._predict(frame, t_ms, camera_model)
                if observations is None:
                    continue
                if self._blink_like(observations, opening_history):
                    rejected_blinks += 1
                    continue
                progress = min(0.999999, max(0.0, (time.monotonic() - started) / duration_s))
                slot = min(self.samples_per_target - 1, int(progress * self.samples_per_target))
                score = self._candidate_score(observations, opening_history)
                current = slots[slot]
                if current is None or score > current[2]:
                    slots[slot] = (frame.copy(), observations, score)
            selected = [value for value in slots if value is not None]
            if len(selected) < 4:
                raise RuntimeError(f"not enough paired landmark frames: {len(selected)}/4")
            for sample_index, (frame, observations, _) in enumerate(selected):
                self._save_observation(
                    frame, observations, target,
                    f"{self.profile_name}-grid-{target['grid_index']:02d}-sample-{sample_index:02d}",
                    "lighting_anchor", target["lighting"],
                )
            self.index += 1
            if self.index == len(self.targets):
                self._save_partials()
            return {
                "ok": True, "phase": self.phase, "index": self.index,
                "accepted": len(selected), "rejected_blinks": rejected_blinks,
            }

    def finish(self) -> dict:
        with self._lock:
            if self.index != len(self.targets):
                raise RuntimeError(
                    f"lighting adaptation has only {self.index}/{len(self.targets)} targets"
                )
            self.phase = "training"
            self.state = "training"
        diagnostics = {}
        try:
            for name, dataset in self.datasets.items():
                finished_at = datetime.now(timezone.utc).isoformat()
                dataset["passes"][-1].update({
                    "finished_at": finished_at, "cancelled": False,
                })
                _save(DATASET_PATHS[name], dataset)
                cnn_path, module_path = MODEL_PATHS[name]
                _, profile_diagnostics = SharedTinyCnnModel.fit_lighting_profile(
                    dataset, cnn_path, module_path, self.profile_name,
                )
                diagnostics[name] = profile_diagnostics
            _update_lighting_profile_library(
                self.config, self.datasets, [self.profile_name],
            )
            self.registry.clear()
            self.phase = "complete"
            self.state = "complete"
            result = {
                "ok": True, "profile_name": self.profile_name,
                "diagnostics": diagnostics,
            }
            _save(self.capture_dir / "completed.json", {
                "schema": "eyetracing-lighting-adaptation-result-v1",
                "pass_id": self.pass_id,
                "profile_name": self.profile_name,
                "screen_level": self.screen_level,
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "diagnostics": diagnostics,
            })
            return result
        except Exception as error:
            self.state = "error"
            self.error = str(error)
            raise
        finally:
            self._close_backends()

    def cancel(self) -> None:
        self.phase = "cancelled"
        self.state = "cancelled"
        for name, dataset in self.datasets.items():
            dataset["passes"][-1].update({
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "cancelled": True,
            })
            _save(self.capture_dir / f"cancelled-{name}-dataset.json", dataset)
        self._close_backends()
