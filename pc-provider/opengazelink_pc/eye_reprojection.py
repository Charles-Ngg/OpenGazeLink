"""Opt-in, single-resampling eye normalization.

The deployed two-stage preprocessing remains available in shared_eye_appearance.
These functions are deliberately separate: a model trained on the old images
must not silently switch to a different sampling or masking convention.
"""
from __future__ import annotations

from typing import Sequence

import cv2
import numpy as np

from .shared_eye_appearance import (
    BASE_HEIGHT,
    BASE_WIDTH,
    MASK_FEATHER_BASE_PX,
    BaseEyeImage,
    CanonicalEyeImage,
    _destination,
    _map_points,
    _similarity_from_two_points,
)


SINGLE_WARP_PREPROCESSING = "pnp_single_warp_canthus_aperture_validity_v1_64x36"


def source_to_base_transform(
    source_quad: Sequence[Sequence[float]], canonical_side: bool,
) -> np.ndarray:
    """Map source pixels to the stored base-eye coordinates, including reflection."""
    transform = cv2.getPerspectiveTransform(
        np.asarray(source_quad, dtype=np.float32).reshape(4, 2), _destination(),
    )
    if not canonical_side:
        reflection = np.asarray([
            [-1.0, 0.0, BASE_WIDTH - 1.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ])
        transform = reflection @ transform
    return transform


def canonicalize_source_eye(
    frame_bgr: np.ndarray,
    base: BaseEyeImage,
    source_to_base: np.ndarray,
    corner_targets: Sequence[Sequence[float]],
) -> CanonicalEyeImage:
    """Align directly from the original frame and propagate source validity.

Only base.corners and base.contour are used; base.gray_base is not resampled.
source_to_base must include the left-eye reflection if the base was mirrored.
Keeping the existing aperture rasterizer isolates the sampling change in an
experiment. This validity mask detects frame boundaries, not 3D occlusion.
"""
    if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
        raise ValueError("source eye normalization requires a BGR frame")
    source_to_base = np.asarray(source_to_base, dtype=np.float64).reshape(3, 3)
    if not np.isfinite(source_to_base).all():
        raise ValueError("source-to-base transform must be finite")
    target = np.asarray(corner_targets, dtype=np.float64).reshape(2, 2)
    alignment = _similarity_from_two_points(base.corners, target)
    source_to_output = alignment @ source_to_base
    aligned = cv2.warpPerspective(
        frame_bgr, source_to_output, (BASE_WIDTH, BASE_HEIGHT),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    )
    validity = cv2.warpPerspective(
        np.ones(frame_bgr.shape[:2], dtype=np.float32), source_to_output,
        (BASE_WIDTH, BASE_HEIGHT), flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    )
    polygon = _map_points(alignment, base.contour)
    aperture = np.zeros((BASE_HEIGHT, BASE_WIDTH), dtype=np.uint8)
    cv2.fillPoly(
        aperture, [np.rint(polygon).astype(np.int32).reshape(-1, 1, 2)],
        255, lineType=cv2.LINE_AA,
    )
    distance = cv2.distanceTransform((aperture >= 128).astype(np.uint8), cv2.DIST_L2, 3)
    alpha = np.clip(distance / MASK_FEATHER_BASE_PX, 0.0, 1.0) * validity
    corners = _map_points(alignment, base.corners)
    return CanonicalEyeImage(
        gray_base=cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY).astype(np.float32),
        alpha_base=np.clip(alpha, 0.0, 1.0).astype(np.float32),
        mapped_inner=tuple(float(value) for value in corners[0]),
        mapped_outer=tuple(float(value) for value in corners[1]),
    )


def canonicalize_source_patch(
    frame_bgr: np.ndarray, patch, canonical_side: bool,
    corner_targets: Sequence[Sequence[float]],
) -> CanonicalEyeImage:
    """Runtime counterpart of the archived-frame experiment, without eye resampling."""
    source_to_base = source_to_base_transform(patch.source_quad, canonical_side)
    corners = _map_points(source_to_base, np.asarray([
        patch.observed_inner_corner, patch.observed_outer_corner,
    ], dtype=np.float64))
    contour = _map_points(source_to_base, np.asarray(patch.observed_contour, dtype=np.float64))
    base = BaseEyeImage(
        gray_base=np.empty((0, 0), dtype=np.float32), corners=corners, contour=contour,
    )
    return canonicalize_source_eye(frame_bgr, base, source_to_base, corner_targets)


def eye_view_geometry(
    rotation: Sequence[Sequence[float]],
    eye_center_camera: Sequence[float],
    canonical_side: bool,
) -> np.ndarray:
    """Camera viewing yaw/pitch in head coordinates and log distance in cm.

Unlike head Euler angles alone, the viewing direction also changes when the
head translates. Reflect horizontal geometry along with the left image/label.
No target coordinates, gaze labels or calibration group IDs enter this vector.
"""
    origin = np.asarray(eye_center_camera, dtype=np.float64).reshape(3)
    distance = float(np.linalg.norm(origin))
    if not np.isfinite(distance) or distance <= 1e-9:
        raise ValueError("eye centre must be finite and away from the camera")
    local = np.asarray(rotation, dtype=np.float64).reshape(3, 3).T @ (-origin / distance)
    if not np.isfinite(local).all() or local[2] <= 0.0:
        raise ValueError("camera must be in the head-forward hemisphere")
    yaw = np.arctan2(local[0], local[2])
    pitch = np.arctan2(-local[1], np.hypot(local[0], local[2]))
    return np.asarray([
        yaw if canonical_side else -yaw, pitch, np.log(distance),
    ], dtype=np.float32)
