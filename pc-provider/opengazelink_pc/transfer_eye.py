"""Shared raw-image eye preprocessing for opt-in public/personal experiments.

This representation has its own version; existing deployment checkpoints must
keep their original PnP/aperture preprocessing. Both data sources call the same
function, with observed corners and eyelid contours in source-image pixels.
"""
from __future__ import annotations

import cv2
import numpy as np

from .shared_eye_appearance import (
    BASE_HEIGHT, BASE_WIDTH, CanonicalEyeImage, _map_points,
    _similarity_from_two_points, cnn_eye_input,
)

TRANSFER_PREPROCESSING = "raw_similarity_canthus_aperture_v1_64x36"
# Fixed before observing either data split. Inner corner is on the right in
# the canonical right-eye image. No personal test images fit this template.
TRANSFER_CORNERS = np.asarray([[49.0, 18.0], [15.0, 18.0]], dtype=np.float64)


def align_transfer_eye(frame_bgr, corners, contour, side):
    """One warp from raw pixels; explicit left reflection preserves up/down."""
    if side not in {"right", "left"}:
        raise ValueError("eye side must be right or left")
    if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
        raise ValueError("expected a BGR source frame")
    corners = np.asarray(corners, dtype=np.float64)
    contour = np.asarray(contour, dtype=np.float64)
    if corners.shape != (2, 2) or contour.ndim != 2 or contour.shape[1] != 2 or len(contour) < 3:
        raise ValueError("expected two corners and an eyelid polygon")
    if not np.isfinite(corners).all() or not np.isfinite(contour).all():
        raise ValueError("nonfinite eye landmarks")
    reflection = np.eye(3)
    if side == "left":
        reflection[0] = [-1.0, 0.0, frame_bgr.shape[1] - 1.0]
    matrix = _similarity_from_two_points(_map_points(reflection, corners), TRANSFER_CORNERS) @ reflection
    warped = cv2.warpPerspective(frame_bgr, matrix, (BASE_WIDTH, BASE_HEIGHT),
                                 flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    # A small source bounding box could otherwise turn a missing pixel into
    # valid black eye texture. Propagate actual source support separately.
    validity = cv2.warpPerspective(np.ones(frame_bgr.shape[:2], np.float32), matrix,
                                  (BASE_WIDTH, BASE_HEIGHT), flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_CONSTANT)
    polygon = _map_points(matrix, contour)
    mask = np.zeros((BASE_HEIGHT, BASE_WIDTH), np.uint8)
    cv2.fillPoly(mask, [np.rint(polygon).astype(np.int32)], 255, lineType=cv2.LINE_AA)
    distance = cv2.distanceTransform((mask >= 128).astype(np.uint8), cv2.DIST_L2, 3)
    alpha = np.clip(distance, 0, 1).astype(np.float32) * validity
    return CanonicalEyeImage(cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY).astype(np.float32),
                             alpha, tuple(TRANSFER_CORNERS[0]), tuple(TRANSFER_CORNERS[1]))


def transfer_eye_input(frame_bgr, corners, contour, side):
    eye = align_transfer_eye(frame_bgr, corners, contour, side)
    return cnn_eye_input(eye.gray_base, eye.alpha_base)
