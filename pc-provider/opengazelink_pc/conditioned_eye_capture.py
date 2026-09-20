from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Mapping

import numpy as np


CONDITIONED_EYE_CAPTURE_SCHEMA = "opengazelink-conditioned-eye-capture-v1"
CONDITIONED_EYE_RUNNING_MODE = "VIDEO"
CONDITIONED_EYE_ORDER = ("right", "left")

_ARRAY_SHAPES = {
    "images": (2, 2, 2, 36, 64),
    "points": (2, 2, 52),
    "head": (2, 10),
    "crop": (2, 8),
    "targets": (2, 3),
    "rotation": (2, 3, 3),
    "center": (2, 3),
    "camera_target": (2, 3),
    "pose_degrees": (2, 3),
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_conditioned_capture(arrays: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    result = {}
    for name, shape in _ARRAY_SHAPES.items():
        if name not in arrays:
            raise ValueError(f"conditioned-eye capture is missing {name}")
        value = np.asarray(arrays[name])
        if value.shape != shape:
            raise ValueError(
                f"conditioned-eye capture {name} has shape {value.shape}, expected {shape}"
            )
        if name == "images":
            if value.dtype != np.uint8:
                raise ValueError("conditioned-eye capture images must be uint8")
        elif not np.issubdtype(value.dtype, np.floating):
            raise ValueError(f"conditioned-eye capture {name} must be floating point")
        if not np.isfinite(value).all():
            raise ValueError(f"conditioned-eye capture {name} contains non-finite values")
        result[name] = value
    norms = np.linalg.norm(result["targets"], axis=1)
    if not np.allclose(norms, 1.0, atol=1e-5):
        raise ValueError("conditioned-eye capture targets must be unit directions")
    return result


def save_conditioned_capture(path: Path, arrays: Mapping[str, np.ndarray]) -> dict:
    values = validate_conditioned_capture(arrays)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(
            stream,
            schema=np.asarray(CONDITIONED_EYE_CAPTURE_SCHEMA),
            mediapipe_running_mode=np.asarray(CONDITIONED_EYE_RUNNING_MODE),
            eye_order=np.asarray(CONDITIONED_EYE_ORDER),
            **values,
        )
    temporary.replace(path)
    return {
        "schema": CONDITIONED_EYE_CAPTURE_SCHEMA,
        "mediapipe_running_mode": CONDITIONED_EYE_RUNNING_MODE,
        "eye_order": list(CONDITIONED_EYE_ORDER),
        "sha256": file_sha256(path),
        "bytes": path.stat().st_size,
    }


def load_conditioned_capture(path: Path, expected_sha256: str | None = None) -> dict[str, np.ndarray]:
    if expected_sha256 is not None and file_sha256(path) != str(expected_sha256):
        raise ValueError(f"conditioned-eye capture checksum mismatch: {path}")
    with np.load(path, allow_pickle=False) as stored:
        schema = str(np.asarray(stored["schema"]).item())
        running_mode = str(np.asarray(stored["mediapipe_running_mode"]).item())
        eye_order = tuple(str(value) for value in np.asarray(stored["eye_order"]).tolist())
        if schema != CONDITIONED_EYE_CAPTURE_SCHEMA:
            raise ValueError(f"unsupported conditioned-eye capture schema: {schema}")
        if running_mode != CONDITIONED_EYE_RUNNING_MODE:
            raise ValueError(
                f"conditioned-eye capture used MediaPipe {running_mode}, expected VIDEO"
            )
        if eye_order != CONDITIONED_EYE_ORDER:
            raise ValueError(f"conditioned-eye capture eye order is {eye_order}")
        arrays = {name: stored[name] for name in _ARRAY_SHAPES}
    return validate_conditioned_capture(arrays)
