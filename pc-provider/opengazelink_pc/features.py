from __future__ import annotations

import math
from typing import Iterable, Optional, Sequence, Tuple

import cv2
import numpy as np

from .types import FeatureSample, HeadPose


FEATURE_NAMES = [
    "iris_left_x",
    "iris_left_y",
    "iris_right_x",
    "iris_right_y",
    "yaw",
    "pitch",
    "head_x",
    "head_y",
    "head_z",
]

TPS_INPUT_NAMES = ["gaze_proxy_x", "gaze_proxy_y"]
PHYSICAL_PROXY_NAMES = ["physical_proxy_x", "physical_proxy_y"]

RIGHT_OUTER = 33
RIGHT_INNER = 133
RIGHT_TOP = 159
RIGHT_BOTTOM = 145
RIGHT_IRIS = 468
RIGHT_IRIS_RING = (468, 469, 470, 471, 472)

LEFT_INNER = 362
LEFT_OUTER = 263
LEFT_TOP = 386
LEFT_BOTTOM = 374
LEFT_IRIS = 473
LEFT_IRIS_RING = (473, 474, 475, 476, 477)

# MediaPipe canonical-face coordinates (centimeters). The midpoint is used only
# as the linearization point of the camera projection; user-specific offsets
# are still removed by calibration's neutral eye reference.
RIGHT_EYE_CANONICAL_CENTER = np.array([-3.1511455, 2.6246180, 4.0654325], dtype=np.float64)
LEFT_EYE_CANONICAL_CENTER = np.array([3.1511455, 2.6246180, 4.0654325], dtype=np.float64)
# The corner midpoint in the official canonical mesh is z=3.465663. Canonical
# +Z points out of the face, so an eyeball centre must move toward -Z. Keep the
# historical centres above for the Jacobian baseline and ray origin; the sphere
# diagnostic uses the physically directed centres below.
CANONICAL_EYE_CORNER_MIDPOINT_Z = 3.465663
# Current per-user horizontal diagnostic estimate from a blue-target yaw sweep.
# This must eventually be calibration data, not a population constant.
EYEBALL_CENTER_BEHIND_CORNERS_CM = 1.03684387
HORIZONTAL_EYE_SEPARATION_SCALE = 0.96909896
RIGHT_EYEBALL_SPHERE_CENTER = np.array([
    -3.1511455 * HORIZONTAL_EYE_SEPARATION_SCALE, 2.6246180,
    CANONICAL_EYE_CORNER_MIDPOINT_Z - EYEBALL_CENTER_BEHIND_CORNERS_CM,
], dtype=np.float64)
LEFT_EYEBALL_SPHERE_CENTER = np.array([
    3.1511455 * HORIZONTAL_EYE_SEPARATION_SCALE, 2.6246180,
    CANONICAL_EYE_CORNER_MIDPOINT_Z - EYEBALL_CENTER_BEHIND_CORNERS_CM,
], dtype=np.float64)
CANONICAL_FACE_WIDTH = abs(7.743095 - (-7.743095))

NOSE_TIP = 1

RIGHT_EYE_CONTOUR = (33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246)
LEFT_EYE_CONTOUR = (263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466)
RIGHT_EYEBROW = (70, 63, 105, 66, 107)
LEFT_EYEBROW = (336, 296, 334, 293, 300)

RIGID_HEAD_MODEL = {
    168: (0.0, 3.271027, 5.236015),
    6: (0.0, 2.473255, 5.788627),
    1: (0.0, -1.126865, 7.475604),
    2: (0.0, -2.089024, 6.058267),
    4: (0.0, -0.46317, 7.58658),
    5: (0.0, 0.365669, 7.24287),
    197: (0.0, 1.728369, 6.31675),
    98: (-1.405627, -1.714196, 5.241087),
    327: (1.405627, -1.714196, 5.241087),
    64: (-1.592294, -1.257709, 5.456949),
    294: (1.592294, -1.257709, 5.456949),
    127: (-7.743095, 2.364999, -2.005167),
    356: (7.743095, 2.364999, -2.005167),
    172: (-5.940524, -6.223629, -0.631468),
    397: (5.940524, -6.223629, -0.631468),
}

SELECTED_LANDMARKS = {
    "forehead": 10,
    "chin": 152,
    "nose_bridge": 168,
    "nose_root": 6,
    "nose_tip": NOSE_TIP,
    "nose_bottom": 2,
    "nose_center": 4,
    "nose_upper": 5,
    "nose_lower_bridge": 197,
    "right_nose_side": 98,
    "left_nose_side": 327,
    "right_nose_wing": 64,
    "left_nose_wing": 294,
    "right_temple": 127,
    "left_temple": 356,
    "right_jaw_side": 172,
    "left_jaw_side": 397,
    "right_cheek": 234,
    "left_cheek": 454,
    "right_outer": RIGHT_OUTER,
    "right_inner": RIGHT_INNER,
    "right_top": RIGHT_TOP,
    "right_bottom": RIGHT_BOTTOM,
    "right_iris": RIGHT_IRIS,
    "right_iris_0": 468,
    "right_iris_1": 469,
    "right_iris_2": 470,
    "right_iris_3": 471,
    "right_iris_4": 472,
    "left_inner": LEFT_INNER,
    "left_outer": LEFT_OUTER,
    "left_top": LEFT_TOP,
    "left_bottom": LEFT_BOTTOM,
    "left_iris": LEFT_IRIS,
    "left_iris_0": 473,
    "left_iris_1": 474,
    "left_iris_2": 475,
    "left_iris_3": 476,
    "left_iris_4": 477,
}
SELECTED_LANDMARKS.update({f"right_eye_contour_{i}": index for i, index in enumerate(RIGHT_EYE_CONTOUR)})
SELECTED_LANDMARKS.update({f"left_eye_contour_{i}": index for i, index in enumerate(LEFT_EYE_CONTOUR)})
SELECTED_LANDMARKS.update({f"right_eyebrow_{i}": index for i, index in enumerate(RIGHT_EYEBROW)})
SELECTED_LANDMARKS.update({f"left_eyebrow_{i}": index for i, index in enumerate(LEFT_EYEBROW)})


class GazeProxyExtractor:
    def __init__(self, head_center_weight: float = 0.25, eyeball_depth: float = 0.62) -> None:
        self.head_center_weight = head_center_weight
        self.eyeball_depth = eyeball_depth

    def extract(
        self,
        landmarks: Sequence,
        head: HeadPose,
        frame_shape: Optional[Tuple[int, int]] = None,
        transform_matrix=None,
        blendshapes: Optional[dict] = None,
    ) -> Optional[dict]:
        normalized_points = np.array([_as_xyz(item) for item in landmarks], dtype=np.float64)
        points = _isotropic_points(normalized_points, frame_shape)
        right_center = 0.5 * (points[RIGHT_OUTER] + points[RIGHT_INNER])
        left_center = 0.5 * (points[LEFT_OUTER] + points[LEFT_INNER])
        interocular = float(np.linalg.norm((left_center - right_center)[:2]))
        if interocular <= 1e-6:
            return None

        face_x = _unit((left_center - right_center)[:2])
        face_y = np.array([-face_x[1], face_x[0]], dtype=np.float64)
        if face_y[1] < 0.0:
            face_y = -face_y

        face_width, face_height = self._face_dimensions(points, face_x, face_y)
        if face_width <= 1e-6 or face_height <= 1e-6:
            return None
        rigid_head = _estimate_rigid_head_pose(landmarks, frame_shape)
        stable_face_width = _pnp_frontal_face_width(rigid_head, frame_shape)
        if stable_face_width <= 1e-6:
            return None

        right_eye = self._eye_proxy(
            points,
            RIGHT_OUTER,
            RIGHT_INNER,
            RIGHT_TOP,
            RIGHT_BOTTOM,
            RIGHT_IRIS,
            RIGHT_IRIS_RING,
            face_x,
            face_y,
            interocular,
            face_width,
            face_height,
            stable_face_width,
            rigid_head,
            frame_shape,
            RIGHT_EYE_CANONICAL_CENTER,
        )
        left_eye = self._eye_proxy(
            points,
            LEFT_OUTER,
            LEFT_INNER,
            LEFT_TOP,
            LEFT_BOTTOM,
            LEFT_IRIS,
            LEFT_IRIS_RING,
            face_x,
            face_y,
            interocular,
            face_width,
            face_height,
            stable_face_width,
            rigid_head,
            frame_shape,
            LEFT_EYE_CANONICAL_CENTER,
        )
        if right_eye is None or left_eye is None:
            return None

        right_quality = right_eye["quality"]
        left_quality = left_eye["quality"]
        total_quality = max(right_quality + left_quality, 1e-6)
        # Keep the aggregate independent of visibility changes. Runtime ray
        # projection uses the two eyes separately; eye2 is diagnostic-only.
        eye2 = (
            0.5 * (right_eye["point"][0] + left_eye["point"][0]),
            0.5 * (right_eye["point"][1] + left_eye["point"][1]),
        )
        iris_diameter = float(np.mean([right_eye["irisDiameter"], left_eye["irisDiameter"]]))

        face_center = 0.5 * (points[:, :2].min(axis=0) + points[:, :2].max(axis=0))
        center_proxy = (float(face_center[0]) / interocular, float(face_center[1]) / interocular)
        head2 = (
            float(head.yaw) + self.head_center_weight * center_proxy[0],
            float(head.pitch) + self.head_center_weight * center_proxy[1],
        )
        eye_diff = (
            float(left_eye["point"][0] - right_eye["point"][0]),
            float(left_eye["point"][1] - right_eye["point"][1]),
        )
        eye_agreement = max(0.0, 1.0 - min(1.0, float(np.hypot(*eye_diff)) / 0.45))
        quality = max(0.0, min(1.0, 0.65 * (total_quality * 0.5) + 0.35 * eye_agreement))
        eye_openness = 0.5 * (float(right_eye["openness"]) + float(left_eye["openness"]))

        rotation = np.array(head.rotation or np.eye(3), dtype=np.float64).reshape(3, 3)
        head_right3 = rotation @ np.array([1.0, 0.0, 0.0], dtype=np.float64)
        head_down3 = rotation @ np.array([0.0, 1.0, 0.0], dtype=np.float64)
        head_forward3 = rotation @ np.array([0.0, 0.0, 1.0], dtype=np.float64)
        frame_height = int(frame_shape[0]) if frame_shape else 0
        frame_width = int(frame_shape[1]) if frame_shape else 0
        transform = None
        if transform_matrix is not None:
            transform = np.array(transform_matrix, dtype=np.float64).reshape(4, 4).tolist()
        return {
            "eye2": (float(eye2[0]), float(eye2[1])),
            "head2": (float(head2[0]), float(head2[1])),
            "gaze2": (float(eye2[0]), float(eye2[1])),
            "rightEye2": (float(right_eye["point"][0]), float(right_eye["point"][1])),
            "leftEye2": (float(left_eye["point"][0]), float(left_eye["point"][1])),
            "rightEyeRaw": right_eye,
            "leftEyeRaw": left_eye,
            "eyeDiff": eye_diff,
            "eyeOpenness": float(eye_openness),
            "faceScale": interocular,
            "faceWidth": float(face_width),
            "stableFaceWidth": float(stable_face_width),
            "faceHeight": float(face_height),
            "faceCenter": (float(face_center[0]), float(face_center[1])),
            "irisDiameter": iris_diameter,
            "centerProxy": center_proxy,
            "faceAxisX": (float(face_x[0]), float(face_x[1])),
            "faceAxisY": (float(face_y[0]), float(face_y[1])),
            "yaw": float(head.yaw),
            "pitch": float(head.pitch),
            "rotation": head.rotation,
            "transformMatrix4x4": transform,
            "headRight3": tuple(float(v) for v in head_right3),
            "headDown3": tuple(float(v) for v in head_down3),
            "headForward3": tuple(float(v) for v in head_forward3),
            "frameShape": {
                "width": frame_width,
                "height": frame_height,
                "aspect": float(frame_width / frame_height) if frame_height > 0 else 0.0,
            },
            "blendshapes": blendshapes or {},
            "rigidHead": rigid_head,
            "eyeModel": "iris_corner_pnp_head_local_vertical_v7",
            "eyeballDepth": float(self.eyeball_depth),
            "selectedLandmarks": {
                name: (
                    float(normalized_points[index][0]),
                    float(normalized_points[index][1]),
                    float(normalized_points[index][2]),
                )
                for name, index in SELECTED_LANDMARKS.items()
            },
            "selectedLandmarkMeta": {
                name: _landmark_meta(landmarks[index])
                for name, index in SELECTED_LANDMARKS.items()
            },
            "quality": float(quality),
        }

    @staticmethod
    def _face_dimensions(points: np.ndarray, face_x: np.ndarray, face_y: np.ndarray) -> Tuple[float, float]:
        projected_x = points[:, :2] @ face_x
        projected_y = points[:, :2] @ face_y
        return float(projected_x.max() - projected_x.min()), float(projected_y.max() - projected_y.min())

    @staticmethod
    def _eye_proxy(
        points: np.ndarray,
        outer_index: int,
        inner_index: int,
        top_index: int,
        bottom_index: int,
        iris_index: int,
        iris_ring: Sequence[int],
        face_x: np.ndarray,
        face_y: np.ndarray,
        face_scale: float,
        face_width: float,
        face_height: float,
        stable_face_width: float,
        rigid_head: dict,
        frame_shape: Optional[Tuple[int, int]],
        canonical_eye_center: np.ndarray,
    ) -> Optional[dict]:
        outer = points[outer_index]
        inner = points[inner_index]
        top = points[top_index]
        bottom = points[bottom_index]
        iris_points = points[list(iris_ring)]
        iris = points[iris_index]
        corner_center = 0.5 * (outer + inner)
        lid_center = 0.5 * (top + bottom)
        box_center = 0.25 * (outer + inner + top + bottom)
        eye_width = float(np.linalg.norm((inner - outer)[:2]))
        eye_height = float(np.linalg.norm((bottom - top)[:2]))
        if (
            eye_width <= 1e-6
            or eye_height <= 1e-6
            or face_scale <= 1e-6
            or face_width <= 1e-6
            or face_height <= 1e-6
            or stable_face_width <= 1e-6
        ):
            return None
        offset = (iris - corner_center)[:2]
        legacy_point = (
            float(np.dot(offset, face_x) / stable_face_width),
            float(np.dot(offset, face_y) / stable_face_width),
        )
        head_local_point = _head_local_eye_offset(
            iris, corner_center, canonical_eye_center, rigid_head, frame_shape
        )
        point = (
            legacy_point[0],
            head_local_point[1] if head_local_point is not None else legacy_point[1],
        )
        image_point = (
            float(np.dot(offset, face_x) / face_scale),
            float(np.dot(offset, face_y) / face_scale),
        )
        iris_diameter = _iris_diameter(iris_points)
        openness = eye_height / max(eye_width, 1e-6)
        quality = max(0.0, min(1.0, (openness - 0.05) / 0.12))
        return {
            "point": point,
            "legacyPoint": legacy_point,
            "headLocalPoint": head_local_point,
            "quality": float(quality),
            "center": (float(corner_center[0]), float(corner_center[1]), float(corner_center[2])),
            "cornerCenter": (float(corner_center[0]), float(corner_center[1]), float(corner_center[2])),
            "lidCenter": (float(lid_center[0]), float(lid_center[1]), float(lid_center[2])),
            "offset": (float(offset[0]), float(offset[1])),
            "imagePoint": image_point,
            "irisCenter": (float(iris[0]), float(iris[1]), float(iris[2])),
            "irisDiameter": float(iris_diameter),
            "eyeWidth": float(eye_width),
            "eyeHeight": float(eye_height),
            "boxCenter": (float(box_center[0]), float(box_center[1]), float(box_center[2])),
            "coordinateDenominator": "isotropicFaceWidth",
            "coordinateScale": float(stable_face_width),
            "imageCoordinateDenominator": "faceScale",
            "imageCoordinateScale": float(face_scale),
            "coordinateModel": "horizontalImageOffset_verticalPnpHeadLocalJacobianV7",
            "openness": float(openness),
        }


def _unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-9:
        return np.array([1.0, 0.0], dtype=np.float64)
    return vector / norm


def _head_local_eye_offset(
    iris: np.ndarray,
    corner_center: np.ndarray,
    canonical_eye_center: np.ndarray,
    rigid_head: dict,
    frame_shape: Optional[Tuple[int, int]],
) -> Optional[Tuple[float, float]]:
    """Invert the local perspective Jacobian into the PnP head frame."""
    if not rigid_head.get("valid", False) or not frame_shape:
        return None
    rotation_value = rigid_head.get("rotation")
    translation_value = rigid_head.get("translation")
    if rotation_value is None or translation_value is None:
        return None
    height, width = int(frame_shape[0]), int(frame_shape[1])
    if height <= 0 or width <= 0:
        return None
    rotation = np.array(rotation_value, dtype=np.float64).reshape(3, 3)
    translation = np.array(translation_value, dtype=np.float64).reshape(3)
    eye_camera = rotation @ canonical_eye_center + translation
    x_camera, y_camera, z_camera = (float(value) for value in eye_camera)
    if not np.isfinite(eye_camera).all() or z_camera <= 1e-6:
        return None
    focal_x = float(rigid_head.get("focalX", rigid_head.get("focalPx", max(width, height))))
    focal_y = float(rigid_head.get("focalY", rigid_head.get("focalPx", max(width, height))))
    if focal_x <= 0.0 or focal_y <= 0.0:
        return None
    projection = np.array([
        [
            focal_x * (rotation[0, 0] * z_camera - x_camera * rotation[2, 0]) / (z_camera * z_camera),
            focal_x * (rotation[0, 1] * z_camera - x_camera * rotation[2, 1]) / (z_camera * z_camera),
        ],
        [
            focal_y * (rotation[1, 0] * z_camera - y_camera * rotation[2, 0]) / (z_camera * z_camera),
            focal_y * (rotation[1, 1] * z_camera - y_camera * rotation[2, 1]) / (z_camera * z_camera),
        ],
    ], dtype=np.float64)
    if not np.isfinite(projection).all() or abs(float(np.linalg.det(projection))) < 1e-8:
        return None
    # Isotropic points use frame-height units on both axes.
    pixel_offset = (iris[:2] - corner_center[:2]) * float(height)
    local_xy, *_ = np.linalg.lstsq(projection, pixel_offset, rcond=None)
    if not np.isfinite(local_xy).all():
        return None
    return (
        float(local_xy[0] / CANONICAL_FACE_WIDTH),
        float(-local_xy[1] / CANONICAL_FACE_WIDTH),
    )


def _as_xyz(point) -> np.ndarray:
    z = point.z if hasattr(point, "z") else 0.0
    return np.array([float(point.x), float(point.y), float(z)], dtype=np.float64)


def _isotropic_points(points: np.ndarray, frame_shape: Optional[Tuple[int, int]]) -> np.ndarray:
    if not frame_shape or frame_shape[0] <= 0 or frame_shape[1] <= 0:
        return points.copy()
    height, width = int(frame_shape[0]), int(frame_shape[1])
    aspect = float(width / height)
    out = points.copy()
    out[:, 0] = (out[:, 0] - 0.5) * aspect
    out[:, 1] = out[:, 1] - 0.5
    out[:, 2] = out[:, 2] * aspect
    return out


def _estimate_rigid_head_pose(
    landmarks: Sequence,
    frame_shape: Optional[Tuple[int, int]],
    camera_model: Optional[dict] = None,
) -> dict:
    if not frame_shape or frame_shape[0] <= 0 or frame_shape[1] <= 0:
        return {"valid": False, "error": "missing frame shape"}
    height, width = int(frame_shape[0]), int(frame_shape[1])
    indices = list(RIGID_HEAD_MODEL.keys())
    object_points = np.array([RIGID_HEAD_MODEL[index] for index in indices], dtype=np.float64)
    image_points = np.array(
        [[float(landmarks[index].x) * width, float(landmarks[index].y) * height] for index in indices],
        dtype=np.float64,
    )
    camera_model = camera_model or {}
    fx = float(camera_model.get("fx", max(width, height)))
    fy = float(camera_model.get("fy", max(width, height)))
    cx = float(camera_model.get("cx", (width - 1) * 0.5))
    cy = float(camera_model.get("cy", (height - 1) * 0.5))
    if not all(np.isfinite(value) for value in (fx, fy, cx, cy)) or fx <= 0.0 or fy <= 0.0:
        return {"valid": False, "error": "invalid camera intrinsics"}
    camera_matrix = np.array(
        [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    distortion = np.zeros((4, 1), dtype=np.float64)
    try:
        ok, rvec, tvec = cv2.solvePnP(
            object_points,
            image_points,
            camera_matrix,
            distortion,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not ok or rvec is None or tvec is None:
            return {"valid": False, "error": "solvePnP failed"}
        if hasattr(cv2, "solvePnPRefineLM"):
            rvec, tvec = cv2.solvePnPRefineLM(
                object_points,
                image_points,
                camera_matrix,
                distortion,
                rvec,
                tvec,
            )
        rotation, _ = cv2.Rodrigues(rvec)
        outward_normal = rotation @ np.array([0.0, 0.0, 1.0], dtype=np.float64)
        forward_denominator = -float(outward_normal[2])
        yaw = math.atan2(float(outward_normal[0]), forward_denominator)
        pitch = math.atan2(float(outward_normal[1]), forward_denominator)
        roll = math.atan2(float(rotation[1, 0]), float(rotation[0, 0]))
        projected, _ = cv2.projectPoints(object_points, rvec, tvec, camera_matrix, distortion)
        errors = np.linalg.norm(projected[:, 0, :] - image_points, axis=1)
        tz = float(tvec[2, 0])
        if abs(tz) < 1e-6:
            return {"valid": False, "error": "invalid PnP depth"}
        return {
            "valid": True,
            "yaw": float(yaw),
            "pitch": float(pitch),
            "roll": float(roll),
            "rotation": [[float(v) for v in row] for row in rotation],
            "translation": [float(tvec[0, 0]), float(tvec[1, 0]), tz],
            "translation2": [float(tvec[0, 0] / tz), float(tvec[1, 0] / tz)],
            "reprojectionErrorPx": float(np.median(errors)),
            "maxReprojectionErrorPx": float(np.max(errors)),
            "inliers": int(len(indices)),
            "points": int(len(indices)),
            "focalPx": float(0.5 * (fx + fy)),
            "focalX": fx,
            "focalY": fy,
            "principalPoint": [cx, cy],
            "cameraModel": dict(camera_model),
        }
    except Exception as error:
        return {"valid": False, "error": str(error)}


def _pnp_frontal_face_width(rigid_head: dict, frame_shape: Optional[Tuple[int, int]]) -> float:
    if not rigid_head.get("valid", False) or not frame_shape or frame_shape[0] <= 0:
        return 0.0
    translation = rigid_head.get("translation") or [0.0, 0.0, 0.0]
    tz = abs(float(translation[2]))
    focal = float(rigid_head.get("focalX", rigid_head.get("focalPx", 0.0)))
    if tz <= 1e-6 or focal <= 0.0:
        return 0.0
    canonical_width = abs(float(RIGID_HEAD_MODEL[356][0] - RIGID_HEAD_MODEL[127][0]))
    return float(canonical_width * focal / tz / float(frame_shape[0]))


def _landmark_meta(point) -> dict:
    visibility = getattr(point, "visibility", None)
    presence = getattr(point, "presence", None)
    return {
        "visibility": float(visibility) if visibility is not None else None,
        "presence": float(presence) if presence is not None else None,
    }


def _blendshape_scores(result) -> dict:
    faces = getattr(result, "face_blendshapes", None) or []
    if not faces:
        return {}
    scores = {}
    for category in faces[0]:
        name = getattr(category, "category_name", None) or getattr(category, "display_name", None)
        if name:
            scores[str(name)] = float(getattr(category, "score", 0.0))
    return scores


def _iris_diameter(iris_points: np.ndarray) -> float:
    if iris_points.shape[0] < 5:
        return 0.0
    center = iris_points[0, :2]
    ring = iris_points[1:, :2]
    radial = float(np.mean(np.linalg.norm(ring - center, axis=1)) * 2.0)
    if iris_points.shape[0] >= 5:
        horizontal = float(np.linalg.norm(iris_points[1, :2] - iris_points[3, :2]))
        vertical = float(np.linalg.norm(iris_points[2, :2] - iris_points[4, :2]))
        candidates = [value for value in (horizontal, vertical, radial) if value > 1e-6]
        return float(np.mean(candidates)) if candidates else 0.0
    return radial


def _project_ratio(point: np.ndarray, origin: np.ndarray, target: np.ndarray) -> float:
    axis = target[:2] - origin[:2]
    denom = float(np.dot(axis, axis))
    if denom <= 1e-10:
        return 0.5
    return float(np.dot(point[:2] - origin[:2], axis) / denom)


def _eye_feature(
    landmarks: Sequence,
    inner_index: int,
    outer_index: int,
    top_index: int,
    bottom_index: int,
    iris_index: int,
) -> Tuple[float, float, float]:
    inner = _as_xyz(landmarks[inner_index])
    outer = _as_xyz(landmarks[outer_index])
    top = _as_xyz(landmarks[top_index])
    bottom = _as_xyz(landmarks[bottom_index])
    iris = _as_xyz(landmarks[iris_index])

    iris_x = _project_ratio(iris, inner, outer)
    iris_y = _project_ratio(iris, top, bottom)
    width = float(np.linalg.norm((outer - inner)[:2]))
    height = float(np.linalg.norm((bottom - top)[:2]))
    openness = height / max(width, 1e-6)
    return max(-0.5, min(1.5, iris_x)), max(-0.5, min(1.5, iris_y)), openness


def _matrix_to_head_pose(matrix: Optional[Iterable[Iterable[float]]], landmarks: Sequence) -> HeadPose:
    if matrix is not None:
        arr = np.array(matrix, dtype=np.float64).reshape(4, 4)
        rotation = arr[:3, :3]
        yaw = math.atan2(rotation[0, 2], rotation[2, 2])
        pitch = math.atan2(-rotation[1, 2], math.hypot(rotation[1, 0], rotation[1, 1]))
        return HeadPose(
            yaw=float(yaw),
            pitch=float(pitch),
            x=float(arr[0, 3]),
            y=float(arr[1, 3]),
            z=float(arr[2, 3]),
            rotation=tuple(tuple(float(v) for v in row) for row in rotation),
        )

    points = np.array([_as_xyz(item) for item in landmarks], dtype=np.float64)
    min_xy = points[:, :2].min(axis=0)
    max_xy = points[:, :2].max(axis=0)
    center = (min_xy + max_xy) * 0.5
    size = np.maximum(max_xy - min_xy, 1e-6)
    nose = _as_xyz(landmarks[NOSE_TIP])
    yaw_proxy = float((nose[0] - center[0]) / size[0])
    pitch_proxy = float((nose[1] - center[1]) / size[1])
    return HeadPose(
        yaw=yaw_proxy,
        pitch=pitch_proxy,
        x=float(center[0] - 0.5),
        y=float(center[1] - 0.5),
        z=float(1.0 / max(size[0], 1e-6)),
    )


def extract_feature_sample(result, t_ms: float, frame_shape: Optional[Tuple[int, int]] = None) -> Optional[FeatureSample]:
    face_landmarks = getattr(result, "face_landmarks", None)
    if not face_landmarks:
        return None
    landmarks = face_landmarks[0]
    if len(landmarks) <= max(max(LEFT_IRIS_RING), max(RIGHT_IRIS_RING)):
        return None

    left_x, left_y, left_open = _eye_feature(
        landmarks, LEFT_INNER, LEFT_OUTER, LEFT_TOP, LEFT_BOTTOM, LEFT_IRIS
    )
    right_x, right_y, right_open = _eye_feature(
        landmarks, RIGHT_INNER, RIGHT_OUTER, RIGHT_TOP, RIGHT_BOTTOM, RIGHT_IRIS
    )

    matrices = getattr(result, "facial_transformation_matrixes", None) or []
    transform_matrix = matrices[0] if matrices else None
    head = _matrix_to_head_pose(transform_matrix, landmarks)
    features = (
        left_x,
        left_y,
        right_x,
        right_y,
        head.yaw,
        head.pitch,
        head.x,
        head.y,
        head.z,
    )

    points = np.array([_as_xyz(item) for item in landmarks], dtype=np.float64)
    face_size = (points[:, 0].max() - points[:, 0].min()) * (points[:, 1].max() - points[:, 1].min())
    eye_open = (left_open + right_open) * 0.5
    face_score = max(0.0, min(1.0, face_size / 0.08))
    eye_score = max(0.0, min(1.0, (eye_open - 0.05) / 0.12))
    confidence = max(0.0, min(1.0, 0.65 * face_score + 0.35 * eye_score))
    proxy = GazeProxyExtractor().extract(
        landmarks,
        head,
        frame_shape=frame_shape,
        transform_matrix=transform_matrix,
        blendshapes=_blendshape_scores(result),
    )
    virtual_point = proxy["gaze2"] if proxy else None
    if proxy:
        confidence = max(0.0, min(1.0, 0.55 * confidence + 0.45 * float(proxy["quality"])))
    return FeatureSample(t_ms=t_ms, features=features, confidence=confidence, head=head, virtual_point=virtual_point, proxy=proxy)
