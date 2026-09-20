"""Eye-centred virtual-camera images with consistently rotated gaze labels.

Independent implementation of projective camera rectification. The experiment
is motivated by Zhang et al., Revisiting Data Normalization for Appearance-Based
Gaze Estimation, ETRA 2018 (rotation, not image depth scaling, acts on gaze).
No external source code or pretrained weights are included.
"""
from __future__ import annotations

import numpy as np

from .eye_reprojection import canonicalize_source_eye
from .shared_eye_appearance import BASE_HEIGHT, BASE_WIDTH, BaseEyeImage, _map_points


VIRTUAL_CAMERA_MODEL = "observed_eye_center_virtual_camera_rotation_labels_v1"


def virtual_eye_transform(source_corners, eye_origin, head_rotation, intrinsics, canonical_side):
    """Return image homography and camera-to-gaze rotation.

    Aim at the observed eye-corner midpoint to avoid generic-centre crop drift.
    Use the recorded eye distance for scale and head horizontal axis for roll.
    Perspective foreshortening is retained; no frontal eye-plane expansion or
    post-warp eye-width normalization is applied.
    """
    source_corners = np.asarray(source_corners, dtype=np.float64).reshape(2, 2)
    camera = np.asarray([
        [intrinsics["fx"], 0.0, intrinsics["cx"]],
        [0.0, intrinsics["fy"], intrinsics["cy"]],
        [0.0, 0.0, 1.0],
    ])
    distance = float(np.linalg.norm(eye_origin))
    if not np.isfinite(distance) or distance < 1e-6:
        raise ValueError("invalid eye-camera distance")
    direction = np.linalg.solve(camera, np.r_[source_corners.mean(axis=0), 1.0])
    direction /= np.linalg.norm(direction)
    horizontal = np.asarray(head_rotation, dtype=np.float64)[:, 0].copy()
    horizontal -= direction * np.dot(horizontal, direction)
    norm = float(np.linalg.norm(horizontal))
    if norm < 0.25:
        raise ValueError("head horizontal axis nearly parallel to the viewing ray")
    horizontal /= norm
    down = np.cross(direction, horizontal)
    view_rotation = np.stack((horizontal, down, direction))
    virtual_distance = 60.0
    virtual_intrinsics = np.asarray([
        [(BASE_WIDTH - 1) * virtual_distance / 4.2, 0, (BASE_WIDTH - 1) / 2],
        [0, (BASE_HEIGHT - 1) * virtual_distance / 2.4, (BASE_HEIGHT - 1) / 2],
        [0, 0, 1.0],
    ])
    # Projectively scaling depth changes crop magnification, not ray orientation.
    image_map = virtual_intrinsics @ np.diag([1.0, 1.0, virtual_distance / distance]) @ view_rotation @ np.linalg.inv(camera)
    if not canonical_side:
        image_map = np.asarray([[-1.0, 0, BASE_WIDTH - 1], [0, 1.0, 0], [0, 0, 1.0]]) @ image_map
    # Gaze has +Z towards the camera/user-facing halfspace, and +Y up.
    # Keeping a proper rotation allows an exact inverse with transpose.
    camera_to_gaze = np.diag([1.0, -1.0, -1.0]) @ view_rotation
    return image_map, camera_to_gaze


def canonicalize_virtual_eye(frame, source_corners, source_contour, image_map):
    corners = _map_points(image_map, np.asarray(source_corners))
    contour = _map_points(image_map, np.asarray(source_contour))
    base = BaseEyeImage(np.empty((0, 0), dtype=np.float32), corners, contour)
    # Equal source/target corners produce identity alignment: rasterize directly
    # through the virtual-camera map, retaining its actual apparent eye width.
    return canonicalize_source_eye(frame, base, image_map, corners)


def direction_angles(direction, canonical_side=True):
    direction = np.asarray(direction, dtype=np.float64)
    if not np.isfinite(direction).all() or direction[2] <= 0:
        raise ValueError("gaze direction must be finite and in the forward hemisphere")
    yaw = np.arctan2(direction[0], direction[2])
    pitch = np.arctan2(-direction[1], np.hypot(direction[0], direction[2]))
    return np.asarray([yaw if canonical_side else -yaw, pitch])


def virtual_gaze_label(target_camera, eye_origin, camera_to_gaze, canonical_side):
    direction = np.asarray(target_camera) - np.asarray(eye_origin)
    return direction_angles(np.asarray(camera_to_gaze) @ direction, canonical_side)


def restore_head_angles(virtual_angles, camera_to_gaze, head_rotation, canonical_side):
    yaw, pitch = np.asarray(virtual_angles, dtype=np.float64)
    if not canonical_side:
        yaw = -yaw
    direction = np.asarray([np.sin(yaw) * np.cos(pitch), -np.sin(pitch), np.cos(yaw) * np.cos(pitch)])
    camera_direction = np.asarray(camera_to_gaze).T @ direction
    head_direction = np.asarray(head_rotation).T @ camera_direction
    return direction_angles(head_direction, canonical_side)
