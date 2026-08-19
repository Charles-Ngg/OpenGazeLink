from __future__ import annotations

import json
from pathlib import Path
import threading

from .shared_eye_models import (
    SHARED_CNN_SCHEMA,
    SHARED_DATASET_SCHEMA,
    SharedTinyCnnModel,
)
from .paths import DATA_DIR

MODEL_PATHS = {
    "tasks": (
        DATA_DIR / "shared-eye-cnn-model.json", DATA_DIR / "shared-eye-cnn-module.pt",
    ),
    "legacy": (
        DATA_DIR / "shared-eye-legacy-cnn-model.json",
        DATA_DIR / "shared-eye-legacy-cnn-module.pt",
    ),
}

DATASET_PATHS = {
    "tasks": DATA_DIR / "shared-eye-angle-calibration.json",
    "legacy": DATA_DIR / "shared-eye-legacy-angle-calibration.json",
}


class ModelRegistry:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._cache: dict[str, SharedTinyCnnModel] = {}
        self._status_cache: dict | None = None

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()
            self._status_cache = None

    def load(self, landmarker: str) -> SharedTinyCnnModel:
        key = landmarker
        with self._lock:
            cached = self._cache.get(key)
            if cached is not None:
                return cached
            metadata_path, module_path = MODEL_PATHS[key]
            model = SharedTinyCnnModel.load(metadata_path, module_path)
            if model.landmarker_backend != landmarker:
                raise ValueError(
                    f"{metadata_path.name} was trained with {model.landmarker_backend}, not {landmarker}"
                )
            self._cache[key] = model
            return model

    def status(self) -> dict:
        with self._lock:
            cached = self._status_cache
        if cached is not None:
            return cached
        result = self._read_status()
        with self._lock:
            if self._status_cache is None:
                self._status_cache = result
            return self._status_cache

    def _read_status(self) -> dict:
        models = {}
        for landmarker, (metadata_path, module_path) in MODEL_PATHS.items():
            key = f"{landmarker}_cnn"
            required = [metadata_path, module_path]
            present = all(path.exists() for path in required)
            item = {
                "landmarker": landmarker,
                "model": "cnn",
                "ready": present,
                "files": [str(path) for path in required],
            }
            if present:
                try:
                    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
                    preprocessing = payload.get("preprocessing") or {}
                    diagnostics = payload.get("diagnostics") or {}
                    backend = preprocessing.get("landmarker_backend", "tasks")
                    schema = payload.get("schema")
                    item.update({
                        "compatible": backend == landmarker and schema == SHARED_CNN_SCHEMA,
                        "trained_backend": backend,
                        "schema": schema,
                        "expected_schema": SHARED_CNN_SCHEMA,
                        "created_at": payload.get("created_at"),
                        "screen": payload.get("screen"),
                        "screen_diagonal_inches": payload.get("screen_diagonal_inches"),
                        "camera_position_screen_cm": payload.get("camera_position_screen_cm"),
                        "screen_camera_origin_cm": payload.get("screen_camera_origin_cm"),
                        "input_source": payload.get("input_source", "phone_udp"),
                        "windows_camera": payload.get("windows_camera"),
                        "training_error_deg": diagnostics.get("training_error_deg"),
                        "holdout": diagnostics.get("comparison_holdout"),
                        "lighting_profiles": sorted((payload.get("lighting_profiles") or {}).keys()),
                    })
                except Exception as error:
                    item.update({"ready": False, "compatible": False, "error": str(error)})
            models[key] = item

        datasets = {}
        for landmarker, path in DATASET_PATHS.items():
            item = {"path": str(path), "ready": path.exists(), "samples": 0, "targets": 0}
            if path.exists():
                try:
                    payload = json.loads(path.read_text(encoding="utf-8"))
                    samples = list(payload.get("samples") or [])
                    lighting_profile_samples: dict[str, int] = {}
                    for sample in samples:
                        if sample.get("condition") != "lighting_anchor":
                            continue
                        lighting = sample.get("lighting") or {}
                        profile_name = (
                            lighting.get("name") if isinstance(lighting, dict)
                            else str(lighting)
                        )
                        if profile_name:
                            lighting_profile_samples[str(profile_name)] = (
                                lighting_profile_samples.get(str(profile_name), 0) + 1
                            )
                    schema = payload.get("schema")
                    item.update({
                        "compatible": schema == SHARED_DATASET_SCHEMA,
                        "schema": schema,
                        "expected_schema": SHARED_DATASET_SCHEMA,
                        "samples": len(samples),
                        "targets": len({sample.get("group") for sample in samples}),
                        "target_layout": payload.get("target_layout"),
                        "created_at": payload.get("created_at"),
                        "passes": payload.get("passes") or [],
                        "screen": payload.get("screen"),
                        "screen_diagonal_inches": payload.get("screen_diagonal_inches"),
                        "screen_camera_origin_cm": payload.get("screen_camera_origin_cm"),
                        "input_source": payload.get("input_source", "phone_udp"),
                        "windows_camera": payload.get("windows_camera"),
                        "lighting_profile_samples": lighting_profile_samples,
                    })
                except Exception as error:
                    item.update({"ready": False, "error": str(error)})
            datasets[landmarker] = item
        return {"models": models, "datasets": datasets}
