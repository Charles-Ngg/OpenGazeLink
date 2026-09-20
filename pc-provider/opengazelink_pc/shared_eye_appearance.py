from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterable, Sequence

import cv2
import numpy as np

if TYPE_CHECKING:
    from .normalized_eye import NormalizedEyePatch


BASE_WIDTH = 64
BASE_HEIGHT = 36
CNN_MODEL_WIDTH = BASE_WIDTH
CNN_MODEL_HEIGHT = BASE_HEIGHT
CNN_INPUT_CHANNELS = 2
AUGMENT_REFERENCE_WIDTH = 20.0
MASK_FEATHER_BASE_PX = 1.0
PREPROCESSING_MODEL = "pnp_two_canthus_aperture_shared_eye_v2_64x36"


@dataclass(frozen=True)
class CanonicalEyeImage:
    gray_base: np.ndarray
    alpha_base: np.ndarray
    mapped_inner: tuple[float, float]
    mapped_outer: tuple[float, float]


@dataclass(frozen=True)
class BaseEyeImage:
    gray_base: np.ndarray
    corners: np.ndarray
    contour: np.ndarray


def _destination() -> np.ndarray:
    return np.asarray([
        [0.0, 0.0], [BASE_WIDTH - 1.0, 0.0],
        [BASE_WIDTH - 1.0, BASE_HEIGHT - 1.0], [0.0, BASE_HEIGHT - 1.0],
    ], dtype=np.float32)


def _map_points(transform: np.ndarray, points: np.ndarray) -> np.ndarray:
    homogeneous = np.column_stack([points, np.ones(len(points), dtype=np.float64)])
    mapped = (transform @ homogeneous.T).T
    return mapped[:, :2] / mapped[:, 2:3]


def _similarity_from_two_points(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    source_delta = source[1] - source[0]
    target_delta = target[1] - target[0]
    denominator = float(source_delta @ source_delta)
    if denominator <= 1e-9:
        raise ValueError("eye corners collapsed in normalized coordinates")
    real = float(source_delta @ target_delta) / denominator
    imag = float(source_delta[0] * target_delta[1] - source_delta[1] * target_delta[0]) / denominator
    linear = np.asarray([[real, -imag], [imag, real]], dtype=np.float64)
    translation = target[0] - linear @ source[0]
    return np.asarray([
        [linear[0, 0], linear[0, 1], translation[0]],
        [linear[1, 0], linear[1, 1], translation[1]],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)


def _base_geometry(patch: NormalizedEyePatch, canonical_side: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    base = cv2.getPerspectiveTransform(np.asarray(patch.source_quad, np.float32), _destination())
    corners = _map_points(base, np.asarray([
        patch.observed_inner_corner, patch.observed_outer_corner,
    ], dtype=np.float64))
    contour = _map_points(base, np.asarray(patch.observed_contour, dtype=np.float64))
    image = patch.image_bgr
    if not canonical_side:
        image = cv2.flip(image, 1)
        corners[:, 0] = (BASE_WIDTH - 1.0) - corners[:, 0]
        contour[:, 0] = (BASE_WIDTH - 1.0) - contour[:, 0]
    return image, corners, contour, base


def base_eye_image(patch: NormalizedEyePatch, canonical_side: bool) -> BaseEyeImage:
    image, corners, contour, _ = _base_geometry(patch, canonical_side)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    return BaseEyeImage(gray_base=gray, corners=corners, contour=contour)


def serialize_base_eye(base: BaseEyeImage) -> dict:
    return {
        "size": [BASE_WIDTH, BASE_HEIGHT],
        "gray": np.rint(base.gray_base).clip(0, 255).astype(np.uint8).reshape(-1).tolist(),
        "corners": base.corners.astype(float).tolist(),
        "contour": base.contour.astype(float).tolist(),
    }


def deserialize_base_eye(payload: dict) -> BaseEyeImage:
    size = tuple(int(value) for value in payload.get("size", ()))
    if size != (BASE_WIDTH, BASE_HEIGHT):
        raise ValueError(
            f"base eye image is {size or 'legacy'}, expected {(BASE_WIDTH, BASE_HEIGHT)}"
        )
    gray = np.asarray(payload["gray"], dtype=np.float32).reshape(BASE_HEIGHT, BASE_WIDTH)
    corners = np.asarray(payload["corners"], dtype=np.float64).reshape(2, 2)
    contour = np.asarray(payload["contour"], dtype=np.float64).reshape(-1, 2)
    return BaseEyeImage(gray_base=gray, corners=corners, contour=contour)


def canonical_corner_targets(patches: Iterable[tuple[NormalizedEyePatch, bool]]) -> np.ndarray:
    mapped = []
    for patch, canonical_side in patches:
        _, corners, _, _ = _base_geometry(patch, canonical_side)
        mapped.append(corners)
    if not mapped:
        raise ValueError("no eye patches are available to estimate canonical corners")
    target = np.median(np.asarray(mapped, dtype=np.float64), axis=0)
    if float(np.linalg.norm(target[1] - target[0])) <= 2.0:
        raise ValueError("canonical eye corners are too close")
    return target


def canonical_corner_targets_from_base(images: Iterable[BaseEyeImage]) -> np.ndarray:
    corners = [image.corners for image in images]
    if not corners:
        raise ValueError("no base eye images are available to estimate canonical corners")
    target = np.median(np.asarray(corners, dtype=np.float64), axis=0)
    if float(np.linalg.norm(target[1] - target[0])) <= 2.0:
        raise ValueError("canonical eye corners are too close")
    return target


def canonicalize_patch(
    patch: NormalizedEyePatch,
    canonical_side: bool,
    corner_targets: Sequence[Sequence[float]],
) -> CanonicalEyeImage:
    image, corners, contour, _ = _base_geometry(patch, canonical_side)
    target = np.asarray(corner_targets, dtype=np.float64).reshape(2, 2)
    transform = _similarity_from_two_points(corners, target)
    aligned = cv2.warpPerspective(
        image, transform, (BASE_WIDTH, BASE_HEIGHT), flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0),
    )
    polygon = _map_points(transform, contour)
    mask = np.zeros((BASE_HEIGHT, BASE_WIDTH), dtype=np.uint8)
    cv2.fillPoly(mask, [np.rint(polygon).astype(np.int32).reshape(-1, 1, 2)], 255, lineType=cv2.LINE_AA)
    inside = (mask >= 128).astype(np.uint8)
    distance = cv2.distanceTransform(inside, cv2.DIST_L2, 3)
    alpha = np.clip(distance / MASK_FEATHER_BASE_PX, 0.0, 1.0).astype(np.float32)
    gray = cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY).astype(np.float32)
    mapped_corners = _map_points(transform, corners)
    return CanonicalEyeImage(
        gray_base=gray,
        alpha_base=alpha,
        mapped_inner=tuple(float(value) for value in mapped_corners[0]),
        mapped_outer=tuple(float(value) for value in mapped_corners[1]),
    )


def canonicalize_base_eye(
    base: BaseEyeImage,
    corner_targets: Sequence[Sequence[float]],
) -> CanonicalEyeImage:
    target = np.asarray(corner_targets, dtype=np.float64).reshape(2, 2)
    transform = _similarity_from_two_points(base.corners, target)
    aligned = cv2.warpPerspective(
        base.gray_base, transform, (BASE_WIDTH, BASE_HEIGHT), flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    ).astype(np.float32)
    polygon = _map_points(transform, base.contour)
    mask = np.zeros((BASE_HEIGHT, BASE_WIDTH), dtype=np.uint8)
    cv2.fillPoly(mask, [np.rint(polygon).astype(np.int32).reshape(-1, 1, 2)], 255, lineType=cv2.LINE_AA)
    inside = (mask >= 128).astype(np.uint8)
    distance = cv2.distanceTransform(inside, cv2.DIST_L2, 3)
    alpha = np.clip(distance / MASK_FEATHER_BASE_PX, 0.0, 1.0).astype(np.float32)
    mapped_corners = _map_points(transform, base.corners)
    return CanonicalEyeImage(
        gray_base=aligned,
        alpha_base=alpha,
        mapped_inner=tuple(float(value) for value in mapped_corners[0]),
        mapped_outer=tuple(float(value) for value in mapped_corners[1]),
    )


def resize_model_arrays(
    gray_base: np.ndarray,
    alpha_base: np.ndarray,
    width: int = CNN_MODEL_WIDTH,
    height: int = CNN_MODEL_HEIGHT,
) -> tuple[np.ndarray, np.ndarray]:
    gray = cv2.resize(gray_base, (width, height), interpolation=cv2.INTER_AREA).astype(np.float32)
    alpha = cv2.resize(alpha_base, (width, height), interpolation=cv2.INTER_AREA).astype(np.float32)
    return gray, np.clip(alpha, 0.0, 1.0)


def cnn_eye_input(gray: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Build illumination-normalized intensity and validity-mask channels."""
    if gray.shape != alpha.shape or gray.ndim != 2:
        raise ValueError("gray and alpha must be matching 2D arrays")
    clipped_alpha = np.clip(np.asarray(alpha, dtype=np.float32), 0.0, 1.0)
    valid = clipped_alpha >= 0.5
    if int(np.count_nonzero(valid)) < 8:
        raise ValueError("eye aperture contains too few valid pixels")
    pixels = np.asarray(gray, dtype=np.float32)[valid]
    center = float(np.median(pixels))
    low, high = np.percentile(pixels, [10.0, 90.0])
    scale = max(float(high - low), 8.0)
    intensity = np.clip((np.asarray(gray, dtype=np.float32) - center) / scale, -3.0, 3.0)
    intensity *= clipped_alpha
    return np.stack([intensity, clipped_alpha], axis=0).astype(np.float32)


def runtime_cnn_input(
    patch: NormalizedEyePatch,
    canonical_side: bool,
    corner_targets: Sequence[Sequence[float]],
) -> np.ndarray:
    canonical = canonicalize_patch(patch, canonical_side, corner_targets)
    gray, alpha = resize_model_arrays(
        canonical.gray_base, canonical.alpha_base, CNN_MODEL_WIDTH, CNN_MODEL_HEIGHT,
    )
    return cnn_eye_input(gray, alpha)


def _augment_eye_arrays(
    gray: np.ndarray,
    alpha: np.ndarray,
    rng: np.random.Generator,
    strength: str,
) -> tuple[np.ndarray, np.ndarray]:
    height, width = gray.shape
    if alpha.shape != (height, width):
        raise ValueError("gray and alpha must have the same shape")
    resolution_scale = width / AUGMENT_REFERENCE_WIDTH
    if strength in {"cnn_geometry", "cnn_photometric"}:
        shift_x = rng.uniform(-0.55, 0.55) * resolution_scale
        shift_y = rng.uniform(-0.35, 0.35) * resolution_scale
        scale = rng.uniform(0.985, 1.015)
        angle = rng.uniform(-0.75, 0.75)
    else:
        raise ValueError(f"unknown augmentation strength: {strength}")
    center = ((width - 1.0) * 0.5, (height - 1.0) * 0.5)
    matrix = cv2.getRotationMatrix2D(center, angle, scale)
    matrix[:, 2] += [shift_x, shift_y]
    warped_gray = cv2.warpAffine(
        gray, matrix, (width, height), flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    )
    warped_alpha = cv2.warpAffine(
        alpha, matrix, (width, height), flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    )
    if strength in {"cnn_geometry", "cnn_photometric"}:
        if rng.random() < 0.7:
            top_inset = float(rng.uniform(0.0, 0.30)) * resolution_scale
            bottom_inset = float(rng.uniform(0.0, 0.25)) * resolution_scale
            adjusted_alpha = warped_alpha.copy()
            rows = np.arange(height, dtype=np.float32)
            for column in range(width):
                visible = np.flatnonzero(warped_alpha[:, column] > 0.05)
                if len(visible) < 2:
                    continue
                top = float(visible[0]) + top_inset
                bottom = float(visible[-1]) - bottom_inset
                top_weight = np.clip(rows - top + 1.0, 0.0, 1.0)
                bottom_weight = np.clip(bottom - rows + 1.0, 0.0, 1.0)
                adjusted_alpha[:, column] *= np.minimum(top_weight, bottom_weight)
            warped_alpha = adjusted_alpha
        if rng.random() < 0.5:
            kernel = np.ones((3, 3), dtype=np.uint8)
            operator = cv2.erode if rng.random() < 0.5 else cv2.dilate
            changed = operator(warped_alpha, kernel, iterations=1)
            amount = float(rng.uniform(0.08, 0.20))
            warped_alpha = warped_alpha * (1.0 - amount) + changed * amount
        if rng.random() < 0.5:
            warped_alpha = cv2.GaussianBlur(
                warped_alpha, (0, 0), sigmaX=float(rng.uniform(0.20, 0.45)),
            )
    if strength == "cnn_photometric":
        values = np.clip(warped_gray / 255.0, 0.0, 1.0)
        gamma = float(rng.uniform(0.85, 1.20))
        values = np.power(values, gamma)
        yy, xx = np.mgrid[-1.0:1.0:complex(height), -1.0:1.0:complex(width)]
        shade_x = float(rng.uniform(-0.12, 0.12))
        shade_y = float(rng.uniform(-0.12, 0.12))
        shade_xy = float(rng.uniform(-0.04, 0.04))
        shading = np.clip(1.0 + shade_x * xx + shade_y * yy + shade_xy * xx * yy, 0.8, 1.2)
        gain = float(rng.uniform(0.80, 1.20))
        offset = float(rng.uniform(-0.03, 0.03))
        values = values * shading * gain + offset
        if rng.random() < 0.15:
            black = float(rng.uniform(0.0, 0.03))
            white = float(rng.uniform(0.97, 1.0))
            values = (values - black) / max(white - black, 0.5)
        values = np.clip(values, 0.0, 1.0)
        photometric = (values * 255.0).astype(np.float32)
        if rng.random() < 0.03:
            photometric = cv2.GaussianBlur(
                photometric, (0, 0), sigmaX=float(rng.uniform(0.15, 0.35)),
            )
        if rng.random() < 0.05:
            sigma = float(rng.uniform(0.0, 0.8))
            photometric = np.clip(
                photometric + rng.normal(0.0, sigma, photometric.shape), 0.0, 255.0,
            )
    else:
        photometric = np.clip(warped_gray, 0.0, 255.0)
    return photometric.astype(np.float32), np.clip(warped_alpha, 0.0, 1.0).astype(np.float32)


def augment_cnn_eye_input(
    gray: np.ndarray,
    alpha: np.ndarray,
    rng: np.random.Generator,
    mode: str = "photometric",
) -> np.ndarray:
    strengths = {
        "geometry": "cnn_geometry",
        "photometric": "cnn_photometric",
    }
    try:
        strength = strengths[mode]
    except KeyError as error:
        raise ValueError(f"unknown CNN augmentation mode: {mode}") from error
    photometric, warped_alpha = _augment_eye_arrays(gray, alpha, rng, strength)
    return cnn_eye_input(photometric, warped_alpha)
