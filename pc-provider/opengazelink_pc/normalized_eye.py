from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Sequence

import cv2
import numpy as np

from .features import (
    LEFT_EYE_CANONICAL_CENTER,
    LEFT_EYE_CONTOUR,
    LEFT_INNER,
    LEFT_OUTER,
    RIGHT_EYE_CANONICAL_CENTER,
    RIGHT_EYE_CONTOUR,
    RIGHT_INNER,
    RIGHT_OUTER,
    _estimate_rigid_head_pose,
)
from .landmarker import TASKS_LANDMARKER_BACKEND, create_normalized_eye_landmarker


NORMALIZED_EYE_WIDTH = 64
NORMALIZED_EYE_HEIGHT = 36
EYE_PLANE_WIDTH_CM = 4.2
EYE_PLANE_HEIGHT_CM = 2.4


@dataclass(frozen=True)
class NormalizedEyePatch:
    subject_eye: str
    image_bgr: np.ndarray
    validity_mask: np.ndarray
    source_quad: tuple[tuple[float, float], ...]
    source_eye_box: tuple[int, int, int, int]
    source_eye_size: tuple[int, int]
    eye_center_head: tuple[float, float, float]
    eye_center_camera: tuple[float, float, float]
    projected_center: tuple[float, float]
    observed_corner_center: tuple[float, float]
    observed_inner_corner: tuple[float, float]
    observed_outer_corner: tuple[float, float]
    observed_contour: tuple[tuple[float, float], ...]
    center_residual_px: tuple[float, float]
    valid_fraction: float
    aperture_ratio: float


@dataclass(frozen=True)
class NormalizedEyeObservation:
    t_ms: float
    right: NormalizedEyePatch
    left: NormalizedEyePatch
    rotation: tuple[tuple[float, float, float], ...]
    translation: tuple[float, float, float]
    head_yaw: float
    head_pitch: float
    head_roll: float
    pnp_reprojection_error_px: float
    camera_model: dict
    detection_ms: float
    normalization_ms: float
    landmarks_count: int
    landmarker_backend: str


def _camera_matrix(camera_model: dict, frame_shape: tuple[int, int]) -> np.ndarray:
    height, width = frame_shape
    fx = float(camera_model.get("fx", max(width, height)))
    fy = float(camera_model.get("fy", max(width, height)))
    cx = float(camera_model.get("cx", (width - 1) * 0.5))
    cy = float(camera_model.get("cy", (height - 1) * 0.5))
    return np.asarray([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)


def _project_points(points_head: np.ndarray, rotation: np.ndarray, translation: np.ndarray, camera_matrix: np.ndarray) -> np.ndarray:
    points_camera = (rotation @ points_head.T).T + translation
    if np.any(points_camera[:, 2] <= 1e-6):
        raise ValueError("normalized eye plane is behind the camera")
    homogeneous = (camera_matrix @ points_camera.T).T
    return homogeneous[:, :2] / homogeneous[:, 2:3]


def _eye_box(landmarks: Sequence[object], contour: Sequence[int], frame_shape: tuple[int, int]) -> tuple[int, int, int, int]:
    height, width = frame_shape
    points = np.asarray(
        [[float(landmarks[index].x) * width, float(landmarks[index].y) * height] for index in contour],
        dtype=np.float64,
    )
    x0, y0 = np.floor(np.min(points, axis=0)).astype(int)
    x1, y1 = np.ceil(np.max(points, axis=0)).astype(int)
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(width, x1 + 1), min(height, y1 + 1)
    return int(x0), int(y0), int(x1), int(y1)


def normalize_eye_patch(
    frame_bgr: np.ndarray,
    landmarks: Sequence[object],
    rigid_head: dict,
    camera_model: dict,
    subject_eye: str,
    output_size: tuple[int, int] = (NORMALIZED_EYE_WIDTH, NORMALIZED_EYE_HEIGHT),
    plane_size_cm: tuple[float, float] = (EYE_PLANE_WIDTH_CM, EYE_PLANE_HEIGHT_CM),
) -> NormalizedEyePatch:
    if subject_eye == "right":
        center_head = np.asarray(RIGHT_EYE_CANONICAL_CENTER, dtype=np.float64)
        outer_index, inner_index, contour = RIGHT_OUTER, RIGHT_INNER, RIGHT_EYE_CONTOUR
    elif subject_eye == "left":
        center_head = np.asarray(LEFT_EYE_CANONICAL_CENTER, dtype=np.float64)
        outer_index, inner_index, contour = LEFT_OUTER, LEFT_INNER, LEFT_EYE_CONTOUR
    else:
        raise ValueError(f"unknown subject eye: {subject_eye}")

    rotation = np.asarray(rigid_head["rotation"], dtype=np.float64).reshape(3, 3)
    translation = np.asarray(rigid_head["translation"], dtype=np.float64).reshape(3)
    camera_matrix = _camera_matrix(camera_model, frame_bgr.shape[:2])
    half_width = float(plane_size_cm[0]) * 0.5
    half_height = float(plane_size_cm[1]) * 0.5
    # Canonical +Y points toward the forehead. The destination is expressed in
    # head coordinates, so perspective and camera-facing mirroring are removed.
    plane = np.asarray([
        center_head + [-half_width, half_height, 0.0],
        center_head + [half_width, half_height, 0.0],
        center_head + [half_width, -half_height, 0.0],
        center_head + [-half_width, -half_height, 0.0],
    ], dtype=np.float64)
    source_quad = _project_points(plane, rotation, translation, camera_matrix).astype(np.float32)
    output_width, output_height = output_size
    destination_quad = np.asarray([
        [0.0, 0.0], [output_width - 1.0, 0.0],
        [output_width - 1.0, output_height - 1.0], [0.0, output_height - 1.0],
    ], dtype=np.float32)
    transform = cv2.getPerspectiveTransform(source_quad, destination_quad)
    normalized = cv2.warpPerspective(
        frame_bgr, transform, (output_width, output_height),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0),
    )
    source_valid = np.full(frame_bgr.shape[:2], 255, dtype=np.uint8)
    validity = cv2.warpPerspective(
        source_valid, transform, (output_width, output_height),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    )
    valid_fraction = float(np.mean(validity >= 250))

    center_camera = rotation @ center_head + translation
    projected_center = _project_points(center_head.reshape(1, 3), rotation, translation, camera_matrix)[0]
    height, width = frame_bgr.shape[:2]
    observed = np.asarray([
        0.5 * (float(landmarks[outer_index].x) + float(landmarks[inner_index].x)) * width,
        0.5 * (float(landmarks[outer_index].y) + float(landmarks[inner_index].y)) * height,
    ], dtype=np.float64)
    observed_inner = np.asarray([
        float(landmarks[inner_index].x) * width,
        float(landmarks[inner_index].y) * height,
    ], dtype=np.float64)
    observed_outer = np.asarray([
        float(landmarks[outer_index].x) * width,
        float(landmarks[outer_index].y) * height,
    ], dtype=np.float64)
    observed_contour = np.asarray([
        [float(landmarks[index].x) * width, float(landmarks[index].y) * height]
        for index in contour
    ], dtype=np.float64)
    contour_width = max(float(np.ptp(observed_contour[:, 0])), 1.0)
    aperture_ratio = float(np.ptp(observed_contour[:, 1]) / contour_width)
    residual = projected_center - observed
    box = _eye_box(landmarks, contour, frame_bgr.shape[:2])
    return NormalizedEyePatch(
        subject_eye=subject_eye,
        image_bgr=normalized,
        validity_mask=validity,
        source_quad=tuple(tuple(float(value) for value in point) for point in source_quad),
        source_eye_box=box,
        source_eye_size=(box[2] - box[0], box[3] - box[1]),
        eye_center_head=tuple(float(value) for value in center_head),
        eye_center_camera=tuple(float(value) for value in center_camera),
        projected_center=tuple(float(value) for value in projected_center),
        observed_corner_center=tuple(float(value) for value in observed),
        observed_inner_corner=tuple(float(value) for value in observed_inner),
        observed_outer_corner=tuple(float(value) for value in observed_outer),
        observed_contour=tuple(
            tuple(float(value) for value in point) for point in observed_contour
        ),
        center_residual_px=tuple(float(value) for value in residual),
        valid_fraction=valid_fraction,
        aperture_ratio=aperture_ratio,
    )


def target_camera_point(
    target_xy: Sequence[float], screen_width: int, screen_height: int,
    screen_diagonal_inches: float,
    screen_origin_camera: Sequence[float] | None = None,
) -> np.ndarray:
    diagonal_cm = float(screen_diagonal_inches) * 2.54
    aspect = float(screen_width) / max(float(screen_height), 1.0)
    physical_height = diagonal_cm / math.sqrt(aspect * aspect + 1.0)
    physical_width = physical_height * aspect
    screen_origin = np.asarray(
        screen_origin_camera
        if screen_origin_camera is not None
        else screen_camera_origin(screen_width, screen_height, screen_diagonal_inches),
        dtype=np.float64,
    )
    return np.asarray([
        screen_origin[0] - (float(target_xy[0]) - (screen_width - 1) * 0.5) * physical_width / max(screen_width - 1, 1),
        screen_origin[1] + (float(target_xy[1]) - (screen_height - 1) * 0.5) * physical_height / max(screen_height - 1, 1),
        screen_origin[2],
    ], dtype=np.float64)


SCREEN_CAMERA_MOUNT = "configured_screen_camera_position_v2"


def screen_camera_origin(
    screen_width: int, screen_height: int, screen_diagonal_inches: float,
    camera_position_screen_cm: Sequence[float] | None = None,
) -> np.ndarray:
    """Return screen centre in camera coordinates.

    ``camera_position_screen_cm`` is user-facing: camera position relative to
    screen centre, with right, down, and toward-user positive. The legacy
    default retains the original bottom-bezel test rig for callers that do not
    yet pass an explicit position.
    """
    if camera_position_screen_cm is not None:
        x_right, y_down, z_toward_user = (
            float(value) for value in camera_position_screen_cm
        )
        return np.asarray([
            x_right,
            -y_down,
            -z_toward_user,
        ], dtype=np.float64)
    diagonal_cm = float(screen_diagonal_inches) * 2.54
    aspect = float(screen_width) / max(float(screen_height), 1.0)
    physical_height = diagonal_cm / math.sqrt(aspect * aspect + 1.0)
    return np.asarray([0.0, -0.5 * physical_height, 0.0], dtype=np.float64)


def eye_in_head_angles(
    target_camera: np.ndarray,
    eye_center_camera: Sequence[float],
    rotation: Sequence[Sequence[float]],
) -> tuple[float, float]:
    direction_camera = np.asarray(target_camera, dtype=np.float64) - np.asarray(eye_center_camera, dtype=np.float64)
    direction_camera /= max(float(np.linalg.norm(direction_camera)), 1e-12)
    local = np.asarray(rotation, dtype=np.float64).reshape(3, 3).T @ direction_camera
    local /= max(float(np.linalg.norm(local)), 1e-12)
    if local[2] <= 1e-9:
        raise ValueError("screen target is outside the head-forward hemisphere")
    yaw = math.atan2(float(local[0]), float(local[2]))
    pitch = math.atan2(-float(local[1]), math.hypot(float(local[0]), float(local[2])))
    return float(yaw), float(pitch)


def angles_to_camera_direction(
    yaw: float, pitch: float, rotation: Sequence[Sequence[float]],
) -> np.ndarray:
    local_x = math.tan(float(yaw))
    local_y = -math.tan(float(pitch)) * math.sqrt(1.0 + local_x * local_x)
    local = np.asarray([local_x, local_y, 1.0], dtype=np.float64)
    local /= max(float(np.linalg.norm(local)), 1e-12)
    camera = np.asarray(rotation, dtype=np.float64).reshape(3, 3) @ local
    return camera / max(float(np.linalg.norm(camera)), 1e-12)


def intersect_screen_plane(
    eye_center_camera: Sequence[float], camera_direction: Sequence[float],
    screen_origin_camera: Sequence[float] = (0.0, 0.0, 0.0),
) -> np.ndarray:
    origin = np.asarray(eye_center_camera, dtype=np.float64)
    direction = np.asarray(camera_direction, dtype=np.float64)
    screen_origin = np.asarray(screen_origin_camera, dtype=np.float64)
    if direction[2] >= -1e-9:
        raise ValueError("predicted gaze ray does not point toward the screen plane")
    distance = (screen_origin[2] - origin[2]) / direction[2]
    if distance <= 0.0:
        raise ValueError("predicted gaze intersection is behind the eye")
    return origin + distance * direction


def screen_point_to_pixels(
    point_camera: Sequence[float], screen_width: int, screen_height: int,
    screen_diagonal_inches: float,
    screen_origin_camera: Sequence[float] | None = None,
) -> tuple[float, float]:
    diagonal_cm = float(screen_diagonal_inches) * 2.54
    aspect = float(screen_width) / max(float(screen_height), 1.0)
    physical_height = diagonal_cm / math.sqrt(aspect * aspect + 1.0)
    physical_width = physical_height * aspect
    if screen_origin_camera is None:
        screen_origin_camera = screen_camera_origin(
            screen_width, screen_height, screen_diagonal_inches,
        )
    point = np.asarray(point_camera, dtype=np.float64) - np.asarray(
        screen_origin_camera, dtype=np.float64,
    )
    return (
        float((-point[0] / physical_width + 0.5) * max(screen_width - 1, 1)),
        float((point[1] / physical_height + 0.5) * max(screen_height - 1, 1)),
    )


class NormalizedEyeBackend:
    def __init__(self, landmarker_backend: str = TASKS_LANDMARKER_BACKEND) -> None:
        self.landmarker_backend = landmarker_backend
        self._landmarker = create_normalized_eye_landmarker(landmarker_backend)
        self._last_timestamp_ms = -1

    def close(self) -> None:
        self._landmarker.close()

    def predict(self, frame_bgr: np.ndarray, t_ms: float, camera_model: dict) -> NormalizedEyeObservation | None:
        timestamp_ms = max(self._last_timestamp_ms + 1, int(round(t_ms)))
        self._last_timestamp_ms = timestamp_ms
        detect_started = time.perf_counter()
        result = self._landmarker.detect_bgr(frame_bgr, timestamp_ms)
        detection_ms = (time.perf_counter() - detect_started) * 1000.0
        faces = getattr(result, "face_landmarks", None) or []
        if not faces:
            return None
        landmarks = faces[0]
        rigid = _estimate_rigid_head_pose(landmarks, frame_bgr.shape[:2], camera_model)
        if not rigid.get("valid", False):
            return None
        normalize_started = time.perf_counter()
        right = normalize_eye_patch(frame_bgr, landmarks, rigid, camera_model, "right")
        left = normalize_eye_patch(frame_bgr, landmarks, rigid, camera_model, "left")
        normalization_ms = (time.perf_counter() - normalize_started) * 1000.0
        rotation = np.asarray(rigid["rotation"], dtype=np.float64).reshape(3, 3)
        translation = np.asarray(rigid["translation"], dtype=np.float64).reshape(3)
        return NormalizedEyeObservation(
            t_ms=float(t_ms), right=right, left=left,
            rotation=tuple(tuple(float(value) for value in row) for row in rotation),
            translation=tuple(float(value) for value in translation),
            head_yaw=float(rigid["yaw"]), head_pitch=float(rigid["pitch"]), head_roll=float(rigid["roll"]),
            pnp_reprojection_error_px=float(rigid["reprojectionErrorPx"]),
            camera_model=dict(camera_model), detection_ms=detection_ms,
            normalization_ms=normalization_ms, landmarks_count=len(landmarks),
            landmarker_backend=self.landmarker_backend,
        )
