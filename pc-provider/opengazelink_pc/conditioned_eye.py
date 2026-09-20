"""Experimental Tasks eye inputs; independent of installed model preprocessing.

Both sources must supply undistorted original pixels and their matching K.
No aperture stretching/masking. Left images and head-vector x are reflected.
"""
from __future__ import annotations

import cv2
import numpy as np
from dataclasses import dataclass

from . import features as f
from .normalized_eye import normalize_eye_patch, eye_in_head_angles
from .shared_eye_appearance import _map_points, _similarity_from_two_points

VERSION = "tasks_conditioned_full_texture_v1"
SIZE = (64, 36)
TARGET_CORNERS = np.array([[49., 18.], [15., 18.]])  # inner, outer


def vector_from_angles(a):
    a = np.asarray(a)
    return np.stack((np.sin(a[..., 0]) * np.cos(a[..., 1]),
                     -np.sin(a[..., 1]), np.cos(a[..., 0]) * np.cos(a[..., 1])), -1)


@dataclass(frozen=True)
class EyeSamplingPlan:
    """Source-geometry-only work, reusable with new pixels of the same camera."""
    matrices: tuple
    masks: tuple
    points: np.ndarray
    head: np.ndarray
    crop: np.ndarray
    rotation: np.ndarray
    center: np.ndarray
    patch: object
    perspective_only: bool


def build_eye_sampling_plan(frame, landmarks, pose, camera, side, *,
                            support_frame=None, perspective_only=False, xy=None):
    """Precompute warps/features without reading image intensities.

    Offline callers retain both historical views; live callers need only view 1.
    Do not change interpolation, reflections or the support mask at boundaries.
    """
    prefix = "RIGHT" if side == "right" else "LEFT"
    inner, outer = getattr(f, prefix + "_INNER"), getattr(f, prefix + "_OUTER")
    indices = list(getattr(f, prefix + "_EYE_CONTOUR")) + list(getattr(f, prefix + "_IRIS_RING")) + list(getattr(f, prefix + "_EYEBROW"))
    h, w = frame.shape[:2]
    if xy is None:
        xy = np.array([[p.x * w, p.y * h] for p in landmarks])
    corners = xy[[inner, outer]]
    distance = np.linalg.norm(corners[0] - corners[1])
    if distance < 3 or not np.isfinite(xy).all():
        raise ValueError("collapsed or nonfinite landmarks")
    if not perspective_only:
        reflection = np.eye(3)
        if side == "left":
            reflection[0] = [-1, 0, w - 1]
        similarity = _similarity_from_two_points(_map_points(reflection, corners), TARGET_CORNERS) @ reflection
    patch = normalize_eye_patch(frame, landmarks, pose, camera, side, warp_image=False)
    plane = cv2.getPerspectiveTransform(np.asarray(patch.source_quad, np.float32),
                                       np.array([[0, 0], [63, 0], [63, 35], [0, 35]], np.float32))
    flip = np.eye(3)
    if side == "left":
        flip[0] = [-1, 0, 63]
    plane = flip @ plane
    perspective = _similarity_from_two_points(_map_points(plane, corners), TARGET_CORNERS) @ plane
    support = np.ones((h, w), np.uint8) * 255 if support_frame is None else support_frame
    matrices = (perspective,) if perspective_only else (similarity, perspective)
    masks, points = [], []
    for matrix in matrices:
        masks.append(cv2.warpPerspective(support, matrix, SIZE, flags=cv2.INTER_LINEAR))
        points.append((_map_points(matrix, xy[indices]) / [63., 35.] - .5).ravel())
    rotation = np.asarray(pose["rotation"])
    center = np.asarray(patch.eye_center_camera)
    midpoint = corners.mean(0)
    face_width = np.linalg.norm(xy[356] - xy[127])
    crop = np.array([(midpoint[0]-camera['cx'])/camera['fx'],
                     (midpoint[1]-camera['cy'])/camera['fy'],
                     distance/camera['fx'], center[2]/60.,
                     camera['fx']/w, camera['fy']/h,
                     pose['reprojectionErrorPx']/max(face_width, 1.),
                     np.linalg.norm(xy[getattr(f,prefix+'_TOP')]-xy[getattr(f,prefix+'_BOTTOM')])/distance])
    # Global R plus side flag avoids silently mixing mirrored coordinate frames.
    head = np.r_[rotation.ravel(), float(side == "left")]
    plan = EyeSamplingPlan(matrices, tuple(masks), np.asarray(points, np.float32),
                           head.astype(np.float32), crop.astype(np.float32),
                           rotation.astype(np.float32), center.astype(np.float32),
                           patch, perspective_only)
    for value in (*plan.matrices, *plan.masks, plan.points, plan.head, plan.crop,
                  plan.rotation, plan.center):
        value.setflags(write=False)
    return plan


def sample_eye_plan(gray, plan):
    images = np.asarray([np.stack((cv2.warpPerspective(gray, matrix, SIZE, flags=cv2.INTER_LINEAR), mask))
                         for matrix, mask in zip(plan.matrices, plan.masks)], np.uint8)
    # Keep the model's established view-1 contract without generating view 0.
    points = plan.points
    if plan.perspective_only:
        images, points = (None, images[0]), (None, points[0])
    return dict(images=images, points=points, head=plan.head, crop=plan.crop,
                rotation=plan.rotation, center=plan.center), plan.patch


def runtime_eye_inputs(frame, landmarks, pose, camera, side, *, gray_frame=None, support_frame=None):
    """Same-frame, two-view preprocessing for capture/training/offline replay."""
    plan = build_eye_sampling_plan(frame, landmarks, pose, camera, side, support_frame=support_frame)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if gray_frame is None else gray_frame
    return sample_eye_plan(gray, plan)


def eye_inputs(frame, landmarks, pose, camera, target_cm, side):
    values, _ = runtime_eye_inputs(frame, landmarks, pose, camera, side)
    rotation = values["rotation"]
    center = values["center"]
    angles = np.array(eye_in_head_angles(target_cm, center, rotation))
    direction = vector_from_angles(angles)
    camera_direction = np.asarray(target_cm) - center
    camera_direction /= np.linalg.norm(camera_direction)
    if not np.allclose(rotation @ direction, camera_direction, atol=1e-6):
        raise ValueError("gaze label round-trip failed")
    if side == "left":
        direction[0] *= -1
    return {**values, "targets": direction.astype(np.float32),
            "camera_target": camera_direction.astype(np.float32)}
