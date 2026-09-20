"""Fit a small personal eye-region model from multi-pose corner observations.

This estimates four rigid canthi in the existing PnP head coordinate system.
It does not estimate an anatomical eyeball centre, absolute metric face scale,
eyelid depth, gaze direction, or a complete face surface.
"""
from __future__ import annotations

import cv2
import numpy as np


PERSONAL_PLANE_MODEL = "multiview_rigid_canthi_plane_v1"


def project_head_points(points, rotation, translation, intrinsics):
    camera = (np.asarray(rotation, dtype=np.float64) @ np.asarray(points, dtype=np.float64).T).T + np.asarray(translation, dtype=np.float64)
    if np.any(camera[:, 2] <= 1e-6):
        raise ValueError("eye-region model projects behind the camera")
    pixels = camera[:, :2] / camera[:, 2:3]
    return pixels * [intrinsics["fx"], intrinsics["fy"]] + [intrinsics["cx"], intrinsics["cy"]]


def fit_rigid_point(observations, rotations, translations, intrinsics, weights):
    """Robustly triangulate one rigid point; input contains no gaze labels.

    Geometry inherits the coordinate frame and scale of the supplied head poses.
    Independent per-frame depth estimates are neither required nor generated.
    """
    from scipy.optimize import least_squares

    observations = np.asarray(observations, dtype=np.float64)
    rotations = np.asarray(rotations, dtype=np.float64)
    translations = np.asarray(translations, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    n = len(observations)
    if n < 8 or observations.shape != (n, 2) or rotations.shape != (n, 3, 3) or translations.shape != (n, 3):
        raise ValueError("need at least eight aligned corner/pose observations")
    if weights.shape != (n,) or not np.isfinite(weights).all() or np.any(weights <= 0):
        raise ValueError("geometry weights must be positive and finite")
    if not all(np.isfinite(x).all() for x in (observations, rotations, translations)):
        raise ValueError("corner/pose observations must be finite")
    weights = weights / weights.mean()
    focal = np.asarray([[k["fx"], k["fy"]] for k in intrinsics], dtype=np.float64)
    principal = np.asarray([[k["cx"], k["cy"]] for k in intrinsics], dtype=np.float64)
    camera_rays = np.column_stack(((observations - principal) / focal, np.ones(n)))
    rays = np.einsum("nji,nj->ni", rotations, camera_rays)
    rays /= np.linalg.norm(rays, axis=1, keepdims=True)
    origins = -np.einsum("nji,nj->ni", rotations, translations)
    perpendicular = np.eye(3)[None] - rays[:, :, None] * rays[:, None, :]
    matrix = (perpendicular * np.sqrt(weights)[:, None, None]).reshape(-1, 3)
    rhs = (np.einsum("nij,nj->ni", perpendicular, origins) * np.sqrt(weights)[:, None]).reshape(-1)
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    condition = float(singular_values[0] / max(singular_values[-1], 1e-12))
    if condition > 100:
        raise ValueError("insufficient viewing-angle diversity to fit a rigid 3D corner")
    initial = np.linalg.lstsq(matrix, rhs, rcond=None)[0]

    def residual(point):
        camera = np.einsum("nij,j->ni", rotations, point) + translations
        if np.any(camera[:, 2] <= 1e-6):
            raise ValueError("fitted corner moved behind the camera")
        projected = camera[:, :2] / camera[:, 2:3] * focal + principal
        return ((projected - observations) * np.sqrt(weights)[:, None]).reshape(-1)

    result = least_squares(residual, initial, loss="soft_l1", f_scale=1.0, max_nfev=100)
    if not result.success or not np.isfinite(result.x).all():
        raise ValueError("rigid eye-corner fit did not converge")
    errors = np.linalg.norm(residual(result.x).reshape(-1, 2) / np.sqrt(weights)[:, None], axis=1)
    return result.x, {
        "observations": n, "ray_system_condition": condition,
        "training_reprojection_median_px": float(np.median(errors)),
        "training_reprojection_p95_px": float(np.percentile(errors, 95)),
        "optimizer_evaluations": int(result.nfev),
    }


def eye_plane_basis(inner, outer):
    """Define a local plane from canthi and the projected head-up direction.

    Two corners identify horizontal tilt, not full surface orientation. The
    vertical tangent therefore retains the head-up prior instead of inventing
    eyelid or eyeball depth from dynamic image structure.
    """
    inner, outer = np.asarray(inner, dtype=np.float64), np.asarray(outer, dtype=np.float64)
    horizontal = outer - inner
    width = float(np.linalg.norm(horizontal))
    if not 1.0 < width < 6.0:
        raise ValueError("fitted canthus separation is implausible in the PnP cm scale")
    if horizontal[0] < 0:
        horizontal = -horizontal
    horizontal /= width
    vertical = np.asarray([0.0, 1.0, 0.0])
    vertical -= horizontal * np.dot(vertical, horizontal)
    length = float(np.linalg.norm(vertical))
    if length < 0.5:
        raise ValueError("fitted eye horizontal direction is nearly vertical")
    vertical /= length
    normal = np.cross(horizontal, vertical)
    if normal[2] < 0.5:
        raise ValueError("fitted eye plane is implausibly oblique to the face")
    return 0.5 * (inner + outer), np.column_stack((horizontal, vertical, normal))


def personal_plane_quad(inner, outer, rotation, translation, intrinsics, plane_size_cm):
    centre, basis = eye_plane_basis(inner, outer)
    half_width, half_height = np.asarray(plane_size_cm, dtype=np.float64) * 0.5
    offsets = np.asarray([
        [-half_width, half_height, 0], [half_width, half_height, 0],
        [half_width, -half_height, 0], [-half_width, -half_height, 0],
    ])
    points = centre + offsets @ basis.T
    return project_head_points(points, rotation, translation, intrinsics).astype(np.float32)
