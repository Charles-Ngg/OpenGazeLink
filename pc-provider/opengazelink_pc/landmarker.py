from __future__ import annotations

import time
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import cv2
import mediapipe as mp

from .paths import LANDMARKER_MODEL_DIR


DEFAULT_MODEL_PATH = (
    LANDMARKER_MODEL_DIR
    / "face_landmarker.task"
).resolve()
TASKS_LANDMARKER_BACKEND = "tasks"
LEGACY_LANDMARKER_BACKEND = "legacy"
NORMALIZED_EYE_LANDMARKER_BACKENDS = (LEGACY_LANDMARKER_BACKEND, TASKS_LANDMARKER_BACKEND)


class FaceLandmarkerProvider:
    def __init__(
        self,
        model_path: Optional[str] = None,
        output_blendshapes: bool = True,
        output_transform_matrix: bool = True,
    ) -> None:
        self.model_path = Path(model_path).resolve() if model_path else DEFAULT_MODEL_PATH
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Face landmarker model not found: {self.model_path}. "
                "Restore pc-provider/models/face_landmarker.task from the repository."
            )

        base_options = mp.tasks.BaseOptions(
            model_asset_buffer=self.model_path.read_bytes(),
            delegate=mp.tasks.BaseOptions.Delegate.CPU,
        )
        options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=0.35,
            min_face_presence_confidence=0.35,
            min_tracking_confidence=0.35,
            output_face_blendshapes=output_blendshapes,
            output_facial_transformation_matrixes=output_transform_matrix,
        )
        self.landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(options)

    def detect_bgr(self, frame_bgr, timestamp_ms: Optional[int] = None):
        if timestamp_ms is None:
            timestamp_ms = int(time.monotonic() * 1000)
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        return self.landmarker.detect_for_video(image, timestamp_ms)

    def close(self) -> None:
        self.landmarker.close()


class LegacyFaceMeshProvider:
    """CPU-only legacy Face Mesh adapter with the Tasks landmark interface."""

    def __init__(
        self,
        output_blendshapes: bool = False,
        output_transform_matrix: bool = False,
    ) -> None:
        if output_blendshapes or output_transform_matrix:
            raise ValueError("legacy Face Mesh does not provide Tasks blendshapes or transform matrices")
        self.landmarker = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.35,
            min_tracking_confidence=0.35,
        )

    def detect_bgr(self, frame_bgr, timestamp_ms: Optional[int] = None):
        del timestamp_ms
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        result = self.landmarker.process(frame_rgb)
        faces = getattr(result, "multi_face_landmarks", None) or []
        return SimpleNamespace(
            face_landmarks=[face.landmark for face in faces],
            face_blendshapes=[],
            facial_transformation_matrixes=[],
        )

    def close(self) -> None:
        self.landmarker.close()


def create_normalized_eye_landmarker(backend: str):
    if backend == LEGACY_LANDMARKER_BACKEND:
        return LegacyFaceMeshProvider()
    if backend == TASKS_LANDMARKER_BACKEND:
        return FaceLandmarkerProvider(
            output_blendshapes=False,
            output_transform_matrix=False,
        )
    raise ValueError(f"unsupported normalized-eye landmarker backend: {backend}")
