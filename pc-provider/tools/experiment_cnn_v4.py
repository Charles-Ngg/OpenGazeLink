from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from opengazelink_pc.shared_eye_appearance import (
    CNN_INPUT_CHANNELS,
    CNN_MODEL_HEIGHT,
    CNN_MODEL_WIDTH,
    augment_cnn_eye_input,
    canonical_corner_targets_from_base,
    cnn_eye_input,
)
from opengazelink_pc.shared_eye_models import (
    _balanced_source_weights,
    _with_geometry,
    cnn_validation_grid_indices,
    prepare_shared_dataset,
    require_torch,
)


@dataclass(frozen=True)
class Geometry:
    origins: np.ndarray
    rotations: np.ndarray
    target_pixels: np.ndarray
    yaw_signs: np.ndarray
    sides: tuple[str, ...]

    def select(self, indices: np.ndarray) -> "Geometry":
        return Geometry(
            origins=self.origins[indices], rotations=self.rotations[indices],
            target_pixels=self.target_pixels[indices], yaw_signs=self.yaw_signs[indices],
            sides=tuple(self.sides[index] for index in indices.tolist()),
        )


def dataset_geometry(dataset: dict) -> Geometry:
    origins = []
    rotations = []
    target_pixels = []
    yaw_signs = []
    sides = []
    for sample in dataset["samples"]:
        rotation = np.asarray(sample["rotation"], dtype=np.float32).reshape(3, 3)
        target = np.asarray(sample["target"], dtype=np.float32)
        for side in ("right", "left"):
            origins.append(sample[f"{side}_eye_center_camera"])
            rotations.append(rotation)
            target_pixels.append(target)
            yaw_signs.append(1.0 if side == "right" else -1.0)
            sides.append(side)
    return Geometry(
        origins=np.asarray(origins, dtype=np.float32),
        rotations=np.asarray(rotations, dtype=np.float32),
        target_pixels=np.asarray(target_pixels, dtype=np.float32),
        yaw_signs=np.asarray(yaw_signs, dtype=np.float32),
        sides=tuple(sides),
    )


def physical_screen_size(dataset: dict) -> tuple[float, float]:
    screen = dataset["screen"]
    aspect = float(screen["width"]) / max(float(screen["height"]), 1.0)
    diagonal_cm = float(dataset["screen_diagonal_inches"]) * 2.54
    height = diagonal_cm / math.sqrt(aspect * aspect + 1.0)
    return height * aspect, height


def _network_class(nn, architecture: str):
    if architecture in {"dense-full", "dense-full-deep", "dense-full-wide"}:
        expanded = architecture != "dense-full"
        wide = architecture == "dense-full-wide"

        class DenseFullResolutionEyeCnn(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                if not expanded:
                    self.features = nn.Sequential(
                        nn.Conv2d(CNN_INPUT_CHANNELS, 8, 3, padding=1), nn.ReLU(),
                        nn.Conv2d(8, 8, 3, padding=1, groups=8),
                        nn.Conv2d(8, 12, 1), nn.ReLU(),
                        nn.Conv2d(12, 12, 3, padding=1, groups=12),
                        nn.Conv2d(12, 16, 1), nn.ReLU(),
                        nn.Conv2d(16, 8, 1), nn.ReLU(),
                    )
                    output_channels = 8
                else:
                    first, middle, work = (16, 24, 32) if wide else (12, 16, 24)
                    self.features = nn.Sequential(
                        nn.Conv2d(CNN_INPUT_CHANNELS, first, 3, padding=1), nn.ReLU(),
                        nn.Conv2d(first, middle, 3, padding=1), nn.ReLU(),
                        nn.Conv2d(middle, work, 3, padding=1), nn.ReLU(),
                        nn.Conv2d(work, 12, 3, padding=1), nn.ReLU(),
                    )
                    output_channels = 12
                self.head = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(output_channels * CNN_MODEL_HEIGHT * CNN_MODEL_WIDTH, 16),
                    nn.ReLU(), nn.Linear(16, 2),
                )

            def forward(self, values, sides):
                del sides
                return self.head(self.features(values))

        return DenseFullResolutionEyeCnn

    if architecture in {"dense-conv-deep", "dense-conv-wide"}:
        wide = architecture == "dense-conv-wide"

        class DenseConvolutionalEyeCnn(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                first, middle, work = (16, 24, 32) if wide else (12, 16, 24)
                self.features = nn.Sequential(
                    nn.Conv2d(CNN_INPUT_CHANNELS, first, 3, padding=1), nn.ReLU(),
                    nn.Conv2d(first, middle, 3, padding=1), nn.ReLU(),
                    nn.AvgPool2d(2),
                    nn.Conv2d(middle, work, 3, padding=1), nn.ReLU(),
                    nn.Conv2d(work, 16, 3, padding=1), nn.ReLU(),
                )
                self.head = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(16 * (CNN_MODEL_HEIGHT // 2) * (CNN_MODEL_WIDTH // 2), 16),
                    nn.ReLU(), nn.Linear(16, 2),
                )

            def forward(self, values, sides):
                del sides
                return self.head(self.features(values))

        return DenseConvolutionalEyeCnn

    if architecture in {"dense-deep", "dense-wide"}:
        wide = architecture == "dense-wide"

        class DenseExpandedEyeCnn(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                if wide:
                    first, middle, final, latent = 16, 24, 32, 32
                else:
                    first, middle, final, latent = 8, 12, 24, 24
                self.features = nn.Sequential(
                    nn.Conv2d(CNN_INPUT_CHANNELS, first, 3, padding=1), nn.ReLU(),
                    nn.Conv2d(first, first, 3, padding=1, groups=first),
                    nn.Conv2d(first, middle, 1), nn.ReLU(), nn.AvgPool2d(2),
                    nn.Conv2d(middle, middle, 3, padding=1, groups=middle),
                    nn.Conv2d(middle, final, 1), nn.ReLU(),
                )
                if not wide:
                    self.features = nn.Sequential(
                        *self.features,
                        nn.Conv2d(final, final, 3, padding=1, groups=final),
                        nn.Conv2d(final, final, 1), nn.ReLU(),
                    )
                feature_count = final * (CNN_MODEL_HEIGHT // 2) * (CNN_MODEL_WIDTH // 2)
                self.head = nn.Sequential(
                    nn.Flatten(), nn.Linear(feature_count, latent),
                    nn.ReLU(), nn.Linear(latent, 2),
                )

            def forward(self, values, sides):
                del sides
                return self.head(self.features(values))

        return DenseExpandedEyeCnn

    if architecture in {"dense", "dense-template-gray", "dense-template-all"}:
        class DenseEyeCnn(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.features = nn.Sequential(
                    nn.Conv2d(CNN_INPUT_CHANNELS, 8, 3, padding=1), nn.ReLU(),
                    nn.Conv2d(8, 8, 3, padding=1, groups=8),
                    nn.Conv2d(8, 12, 1), nn.ReLU(), nn.AvgPool2d(2),
                    nn.Conv2d(12, 12, 3, padding=1, groups=12),
                    nn.Conv2d(12, 16, 1), nn.ReLU(),
                )
                self.head = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(16 * (CNN_MODEL_HEIGHT // 2) * (CNN_MODEL_WIDTH // 2), 16),
                    nn.ReLU(), nn.Linear(16, 2),
                )

            def forward(self, values, sides):
                del sides
                return self.head(self.features(values))

        return DenseEyeCnn

    if architecture in {"dense-side-output", "dense-side-heads"}:
        separate_full_heads = architecture == "dense-side-heads"

        class DenseSideEyeCnn(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.features = nn.Sequential(
                    nn.Conv2d(CNN_INPUT_CHANNELS, 8, 3, padding=1), nn.ReLU(),
                    nn.Conv2d(8, 8, 3, padding=1, groups=8),
                    nn.Conv2d(8, 12, 1), nn.ReLU(), nn.AvgPool2d(2),
                    nn.Conv2d(12, 12, 3, padding=1, groups=12),
                    nn.Conv2d(12, 16, 1), nn.ReLU(), nn.Flatten(),
                )
                feature_count = 16 * (CNN_MODEL_HEIGHT // 2) * (CNN_MODEL_WIDTH // 2)
                if separate_full_heads:
                    self.shared = nn.Identity()
                    head_inputs = feature_count
                else:
                    self.shared = nn.Sequential(nn.Linear(feature_count, 16), nn.ReLU())
                    head_inputs = 16
                self.right_head = nn.Sequential(
                    nn.Linear(head_inputs, 16), nn.ReLU(), nn.Linear(16, 2),
                ) if separate_full_heads else nn.Linear(head_inputs, 2)
                self.left_head = nn.Sequential(
                    nn.Linear(head_inputs, 16), nn.ReLU(), nn.Linear(16, 2),
                ) if separate_full_heads else nn.Linear(head_inputs, 2)

            def forward(self, values, sides):
                shared = self.shared(self.features(values))
                right = self.right_head(shared)
                left = self.left_head(shared)
                choose_right = (sides >= 0.0).to(values.dtype).reshape(-1, 1)
                return right * choose_right + left * (1.0 - choose_right)

        return DenseSideEyeCnn

    if architecture in {"dense-split", "dense-split-side", "dense-split-conditional"}:
        use_side = architecture == "dense-split-side"
        conditional_pitch = architecture == "dense-split-conditional"

        class DenseSplitEyeCnn(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.features = nn.Sequential(
                    nn.Conv2d(CNN_INPUT_CHANNELS, 8, 3, padding=1), nn.ReLU(),
                    nn.Conv2d(8, 8, 3, padding=1, groups=8),
                    nn.Conv2d(8, 12, 1), nn.ReLU(), nn.AvgPool2d(2),
                    nn.Conv2d(12, 12, 3, padding=1, groups=12),
                    nn.Conv2d(12, 16, 1), nn.ReLU(), nn.Flatten(),
                )
                feature_count = 16 * (CNN_MODEL_HEIGHT // 2) * (CNN_MODEL_WIDTH // 2)
                self.yaw_features = nn.Sequential(nn.Linear(feature_count, 8), nn.SiLU())
                self.pitch_features = nn.Sequential(nn.Linear(feature_count, 8), nn.SiLU())
                self.yaw = nn.Linear(8 + (1 if use_side else 0), 1)
                self.pitch = nn.Linear(
                    8 + (1 if use_side else 0) + (1 if conditional_pitch else 0), 1,
                )

            def forward(self, values, sides):
                features = self.features(values)
                yaw_features = self.yaw_features(features)
                pitch_features = self.pitch_features(features)
                if use_side:
                    side_column = sides.reshape(-1, 1)
                    yaw_features = require_torch()[0].cat([yaw_features, side_column], dim=1)
                    pitch_features = require_torch()[0].cat([pitch_features, side_column], dim=1)
                yaw = self.yaw(yaw_features)
                if conditional_pitch:
                    pitch_features = require_torch()[0].cat([pitch_features, yaw], dim=1)
                return require_torch()[0].cat([yaw, self.pitch(pitch_features)], dim=1)

        return DenseSplitEyeCnn

    use_side = architecture in {"coord-side", "coord-side-conditional"}
    conditional_pitch = architecture in {"coord-side-conditional", "coord-grid-conditional"}
    use_grid = architecture in {"coord-grid", "coord-grid-conditional"}
    if architecture not in {
        "coord", "coord-side", "coord-side-conditional",
        "coord-grid", "coord-grid-conditional",
    }:
        raise ValueError(f"unknown architecture: {architecture}")

    class CoordEyeCnn(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            yy, xx = np.mgrid[-1.0:1.0:complex(CNN_MODEL_HEIGHT), -1.0:1.0:complex(CNN_MODEL_WIDTH)]
            coordinates = np.stack([xx, yy], axis=0).astype(np.float32)
            self.register_buffer("coordinates", require_torch()[0].from_numpy(coordinates).unsqueeze(0))
            pool = nn.AdaptiveAvgPool2d((3, 5)) if use_grid else nn.AdaptiveAvgPool2d(1)
            self.features = nn.Sequential(
                nn.Conv2d(CNN_INPUT_CHANNELS + 2, 16, 3, padding=1, bias=False),
                nn.GroupNorm(4, 16), nn.SiLU(),
                nn.Conv2d(16, 16, 3, stride=2, padding=1, groups=16, bias=False),
                nn.Conv2d(16, 24, 1, bias=False), nn.GroupNorm(4, 24), nn.SiLU(),
                nn.Conv2d(24, 24, 3, stride=2, padding=1, groups=24, bias=False),
                nn.Conv2d(24, 32, 1, bias=False), nn.GroupNorm(4, 32), nn.SiLU(),
                nn.Conv2d(32, 32, 3, padding=1, groups=32, bias=False),
                nn.Conv2d(32, 48, 1, bias=False), nn.GroupNorm(6, 48), nn.SiLU(),
                pool, nn.Flatten(),
            )
            shared_inputs = 48 * (15 if use_grid else 1) + (1 if use_side else 0)
            self.shared = nn.Sequential(nn.Linear(shared_inputs, 32), nn.SiLU())
            self.yaw = nn.Linear(32, 1)
            pitch_inputs = 33 if conditional_pitch else 32
            self.pitch = nn.Sequential(
                nn.Linear(pitch_inputs, 16), nn.SiLU(), nn.Linear(16, 1),
            )

        def forward(self, values, sides):
            coordinates = self.coordinates.expand(values.shape[0], -1, -1, -1)
            features = self.features(require_torch()[0].cat([values, coordinates], dim=1))
            if use_side:
                features = require_torch()[0].cat([features, sides.reshape(-1, 1)], dim=1)
            shared = self.shared(features)
            yaw = self.yaw(shared)
            pitch_input = require_torch()[0].cat([shared, yaw], dim=1) if conditional_pitch else shared
            return require_torch()[0].cat([yaw, self.pitch(pitch_input)], dim=1)

    return CoordEyeCnn


def torch_screen_points(
    torch, canonical_angles, geometry: Geometry, source_indices, screen_size,
):
    origins = torch.from_numpy(geometry.origins[source_indices]).to(canonical_angles)
    rotations = torch.from_numpy(geometry.rotations[source_indices]).to(canonical_angles)
    yaw_signs = torch.from_numpy(geometry.yaw_signs[source_indices]).to(canonical_angles)
    yaw = canonical_angles[:, 0] * yaw_signs
    pitch = canonical_angles[:, 1]
    local_x = torch.tan(yaw)
    local_y = -torch.tan(pitch) * torch.sqrt(1.0 + local_x.square())
    local = torch.stack([local_x, local_y, torch.ones_like(local_x)], dim=1)
    direction = torch.bmm(rotations, local.unsqueeze(2)).squeeze(2)
    denominator = torch.where(
        direction[:, 2] < -0.02,
        direction[:, 2],
        -0.02 + 0.01 * direction[:, 2],
    )
    points = origins[:, :2] - origins[:, 2:3] * direction[:, :2] / denominator.unsqueeze(1)
    width_cm, height_cm = screen_size
    return torch.stack([
        -points[:, 0] / width_cm + 0.5,
        points[:, 1] / height_cm + 0.5,
    ], dim=1)


def numpy_screen_points(angles, geometry: Geometry, screen_size, screen_resolution):
    raw = np.asarray(angles, dtype=np.float64).copy()
    raw[:, 0] *= geometry.yaw_signs
    local_x = np.tan(raw[:, 0])
    local_y = -np.tan(raw[:, 1]) * np.sqrt(1.0 + local_x * local_x)
    local = np.stack([local_x, local_y, np.ones_like(local_x)], axis=1)
    direction = np.einsum("nij,nj->ni", geometry.rotations, local)
    points = geometry.origins[:, :2] - geometry.origins[:, 2:3] * direction[:, :2] / direction[:, 2:3]
    width_cm, height_cm = screen_size
    width, height = screen_resolution
    return np.stack([
        (-points[:, 0] / width_cm + 0.5) * max(width - 1, 1),
        (points[:, 1] / height_cm + 0.5) * max(height - 1, 1),
    ], axis=1)


def affine_response(targets, predictions):
    centered_targets = targets - np.mean(targets, axis=0, keepdims=True)
    centered_predictions = predictions - np.mean(predictions, axis=0, keepdims=True)
    matrix, *_ = np.linalg.lstsq(centered_targets, centered_predictions, rcond=None)
    return {
        "yaw_gain": float(matrix[0, 0]),
        "pitch_to_yaw": float(matrix[1, 0]),
        "yaw_to_pitch": float(matrix[0, 1]),
        "pitch_gain": float(matrix[1, 1]),
    }


def evaluate(
    targets, predictions, groups, geometry: Geometry, screen_size, screen_resolution,
):
    angle_errors = np.degrees(np.linalg.norm(predictions - targets, axis=1))
    points = numpy_screen_points(predictions, geometry, screen_size, screen_resolution)
    screen_errors = np.linalg.norm(points - geometry.target_pixels, axis=1)
    per_group = {}
    for group in sorted(set(groups)):
        mask = np.asarray(groups) == group
        per_group[group] = float(np.median(screen_errors[mask]))
    corners = {"grid-00", "grid-02", "grid-12", "grid-14"}
    corner_mask = np.asarray([group in corners for group in groups])
    sides = np.asarray(geometry.sides)
    static = np.asarray([not group.startswith("pose-") for group in groups])
    responses = {}
    for condition, condition_mask in (("static", static), ("pose", ~static)):
        for side in ("right", "left"):
            mask = condition_mask & (sides == side)
            if np.count_nonzero(mask) >= 4:
                responses[f"{condition}_{side}"] = affine_response(targets[mask], predictions[mask])
    paired_separations = []
    for index in range(0, len(points) - 1, 2):
        if geometry.sides[index:index + 2] == ("right", "left"):
            paired_separations.append(float(np.linalg.norm(points[index] - points[index + 1])))
    return {
        "angle_median_deg": float(np.median(angle_errors)),
        "angle_p95_deg": float(np.percentile(angle_errors, 95.0)),
        "screen_median_px": float(np.median(screen_errors)),
        "screen_p95_px": float(np.percentile(screen_errors, 95.0)),
        "corner_median_px": float(np.median(screen_errors[corner_mask])) if np.any(corner_mask) else None,
        "worst_group_median_px": float(max(per_group.values())),
        "binocular_separation_median_px": float(np.median(paired_separations)) if paired_separations else None,
        "responses": responses,
        "per_group_median_px": per_group,
    }


def train_candidate(
    dataset, data, geometry, train_indices, validation_indices,
    architecture: str, loss_mode: str, augmentation_profile: str, seed: int, epochs: int,
    extra_validation_indices: np.ndarray | None = None,
    pose_mass: float = 0.40,
    return_runtime: bool = False,
):
    torch, nn = require_torch()
    torch.manual_seed(seed)
    torch.set_num_threads(max(1, min(8, int(torch.get_num_threads()))))
    augmentation_modes = {
        "none": (),
        "geometry": ("geometry", "geometry"),
        "current": ("geometry", "geometry", "photometric"),
    }.get(augmentation_profile)
    if augmentation_modes is None:
        raise ValueError(f"unknown augmentation profile: {augmentation_profile}")
    augmentation_rng = np.random.default_rng(1000 + seed)
    train_images = []
    train_targets = []
    for index in train_indices.tolist():
        train_images.append(cnn_eye_input(data.gray[index], data.alpha[index]))
        train_targets.append(data.targets[index])
        for mode in augmentation_modes:
            train_images.append(augment_cnn_eye_input(
                data.gray[index], data.alpha[index], augmentation_rng, mode=mode,
            ))
            train_targets.append(data.targets[index])
    train_images = np.asarray(train_images, dtype=np.float32)
    train_targets = np.asarray(train_targets, dtype=np.float32)
    views_per_source = 1 + len(augmentation_modes)
    validation_images = np.asarray([
        cnn_eye_input(data.gray[index], data.alpha[index]) for index in validation_indices
    ], dtype=np.float32)
    validation_targets = data.targets[validation_indices].astype(np.float32)
    train_side_signs = np.repeat(geometry.yaw_signs[train_indices], views_per_source)
    validation_side_signs = geometry.yaw_signs[validation_indices]
    train_side_indices = (train_side_signs < 0.0).astype(np.int64)
    validation_side_indices = (validation_side_signs < 0.0).astype(np.int64)
    template_mode = architecture in {"dense-template-gray", "dense-template-all"}
    if template_mode:
        image_templates = np.stack([
            np.mean(train_images[train_side_indices == side_index], axis=0)
            for side_index in range(2)
        ]).astype(np.float32)
        if architecture == "dense-template-gray":
            image_templates[:, 1] = 0.0
        centered_train_images = train_images - image_templates[train_side_indices]
        centered_validation_images = validation_images - image_templates[validation_side_indices]
        target_means = np.stack([
            np.mean(train_targets[train_side_indices == side_index], axis=0)
            for side_index in range(2)
        ]).astype(np.float32)
    else:
        image_templates = np.zeros(
            (2, CNN_INPUT_CHANNELS, CNN_MODEL_HEIGHT, CNN_MODEL_WIDTH), dtype=np.float32,
        )
        image_mean = np.mean(train_images, axis=(0, 2, 3), keepdims=True).astype(np.float32)
        centered_train_images = train_images - image_mean
        centered_validation_images = validation_images - image_mean
        shared_target_mean = np.mean(train_targets, axis=0).astype(np.float32)
        target_means = np.stack([shared_target_mean, shared_target_mean])
    image_scale = np.maximum(
        np.std(centered_train_images, axis=(0, 2, 3)), 1e-3,
    ).astype(np.float32)
    target_residuals = train_targets - target_means[train_side_indices]
    target_scale = np.maximum(np.std(target_residuals, axis=0), np.radians(1.0)).astype(np.float32)
    scale_view = image_scale.reshape(1, CNN_INPUT_CHANNELS, 1, 1)
    train_x = torch.from_numpy(centered_train_images / scale_view).reshape(
        len(train_indices), views_per_source,
        CNN_INPUT_CHANNELS, CNN_MODEL_HEIGHT, CNN_MODEL_WIDTH,
    )
    train_y = torch.from_numpy(target_residuals / target_scale).reshape(
        len(train_indices), views_per_source, 2,
    )
    validation_x = torch.from_numpy(centered_validation_images / scale_view)
    source_sides = torch.from_numpy(geometry.yaw_signs[train_indices]).float()
    validation_sides = torch.from_numpy(geometry.yaw_signs[validation_indices]).float()
    weights = _balanced_source_weights(data, train_indices, pose_mass=pose_mass)
    probability = weights / np.sum(weights)
    screen_size = physical_screen_size(dataset)
    screen = dataset["screen"]
    screen_resolution = (int(screen["width"]), int(screen["height"]))
    target_fraction = geometry.target_pixels[train_indices] / np.asarray([
        max(screen_resolution[0] - 1, 1), max(screen_resolution[1] - 1, 1),
    ], dtype=np.float32)
    screen_target_scale = np.maximum(np.std(target_fraction, axis=0), 0.1).astype(np.float32)
    model = _network_class(nn, architecture)()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    smooth = nn.SmoothL1Loss(beta=0.35)
    mse = nn.MSELoss()
    rng = np.random.default_rng(seed)
    best_state = None
    best_score = float("inf")
    best_epoch = 0
    best_metrics = None
    stale = 0
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        order = rng.choice(len(train_indices), len(train_indices), replace=True, p=probability)
        epoch_losses = []
        for start in range(0, len(order), 32):
            batch_np = order[start:start + 32]
            batch = torch.from_numpy(batch_np.astype(np.int64))
            optimizer.zero_grad(set_to_none=True)
            batch_x = train_x[batch]
            batch_y = train_y[batch]
            batch_sides = source_sides[batch]
            repeated_sides = batch_sides[:, None].expand(-1, views_per_source).reshape(-1)
            normalized = model(batch_x.flatten(0, 1), repeated_sides).reshape(
                len(batch), views_per_source, 2,
            )
            if loss_mode == "angle-smooth":
                supervised = smooth(normalized, batch_y)
            elif loss_mode == "angle-mse":
                supervised = mse(normalized, batch_y)
            elif loss_mode in {"screen-mse", "screen-mse-radial"}:
                repeated_side_indices = np.repeat(
                    (geometry.yaw_signs[train_indices[batch_np]] < 0.0).astype(np.int64),
                    views_per_source,
                )
                angle_means = torch.from_numpy(target_means[repeated_side_indices])
                angles = normalized.reshape(-1, 2) * torch.from_numpy(target_scale) + angle_means
                source_indices = np.repeat(train_indices[batch_np], views_per_source)
                predicted_points = torch_screen_points(
                    torch, angles, geometry, source_indices, screen_size,
                )
                target_points = torch.from_numpy(target_fraction[batch_np]).to(predicted_points)
                target_points = target_points[:, None, :].expand(
                    -1, views_per_source, -1,
                ).reshape(-1, 2)
                screen_scale_tensor = torch.from_numpy(screen_target_scale).to(predicted_points)
                screen_error = (
                    (predicted_points - target_points) / screen_scale_tensor
                ).square().mean(dim=1)
                if loss_mode == "screen-mse-radial":
                    radial = np.clip(np.sum(
                        ((target_fraction[batch_np] - 0.5) / 0.5) ** 2,
                        axis=1,
                    ) / 2.0, 0.0, 1.0).astype(np.float32)
                    radial = torch.from_numpy(radial).to(screen_error)
                    radial = radial[:, None].expand(-1, views_per_source).reshape(-1)
                    screen_error = screen_error * (1.0 + radial)
                supervised = screen_error.mean()
            else:
                raise ValueError(f"unknown loss mode: {loss_mode}")
            if views_per_source > 1:
                reference = normalized[:, :1, :].expand_as(normalized[:, 1:, :])
                consistency = smooth(normalized[:, 1:, :], reference)
            else:
                consistency = normalized.sum() * 0.0
            original = normalized[:, 0, :]
            original_target = batch_y[:, 0, :]
            delta = smooth(
                original - original.mean(dim=0, keepdim=True),
                original_target - original_target.mean(dim=0, keepdim=True),
            )
            loss = supervised + 0.05 * consistency + 0.15 * delta
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.detach()))
        model.eval()
        with torch.inference_mode():
            normalized_validation = model(validation_x, validation_sides).numpy()
        validation_prediction = (
            normalized_validation * target_scale + target_means[validation_side_indices]
        )
        metrics = evaluate(
            validation_targets, validation_prediction,
            tuple(data.groups[index] for index in validation_indices.tolist()),
            geometry.select(validation_indices), screen_size, screen_resolution,
        )
        score = metrics["screen_median_px"] + 0.25 * metrics["screen_p95_px"]
        history.append({"epoch": epoch, "loss": float(np.mean(epoch_losses)), "score": score})
        if score < best_score - 1e-4:
            best_score = score
            best_epoch = epoch
            best_metrics = metrics
            best_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= 20:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    source_side_indices = (geometry.yaw_signs[train_indices] < 0.0).astype(np.int64)
    with torch.inference_mode():
        normalized_training = model(train_x[:, 0], source_sides).numpy()
    training_prediction = (
        normalized_training * target_scale + target_means[source_side_indices]
    )
    training_metrics = evaluate(
        data.targets[train_indices].astype(np.float32), training_prediction,
        tuple(data.groups[index] for index in train_indices.tolist()),
        geometry.select(train_indices), screen_size, screen_resolution,
    )
    extra_metrics = None
    if extra_validation_indices is not None and len(extra_validation_indices) > 0:
        extra_images = np.asarray([
            cnn_eye_input(data.gray[index], data.alpha[index])
            for index in extra_validation_indices.tolist()
        ], dtype=np.float32)
        extra_side_indices = (
            geometry.yaw_signs[extra_validation_indices] < 0.0
        ).astype(np.int64)
        if template_mode:
            extra_centered = extra_images - image_templates[extra_side_indices]
        else:
            extra_centered = extra_images - image_mean
        extra_x = torch.from_numpy(extra_centered / scale_view)
        with torch.inference_mode():
            normalized_extra = model(
                extra_x, torch.from_numpy(geometry.yaw_signs[extra_validation_indices]).float(),
            ).numpy()
        extra_prediction = (
            normalized_extra * target_scale + target_means[extra_side_indices]
        )
        extra_metrics = evaluate(
            data.targets[extra_validation_indices].astype(np.float32), extra_prediction,
            tuple(data.groups[index] for index in extra_validation_indices.tolist()),
            geometry.select(extra_validation_indices), screen_size, screen_resolution,
        )
    result = {
        "architecture": architecture,
        "loss_mode": loss_mode,
        "augmentation_profile": augmentation_profile,
        "pose_mass": pose_mass,
        "seed": seed,
        "parameters": int(sum(parameter.numel() for parameter in model.parameters())),
        "best_epoch": best_epoch,
        "metrics": best_metrics,
        "training_metrics": training_metrics,
        "extra_metrics": extra_metrics,
        "history": history,
    }
    if return_runtime:
        result["runtime"] = {
            "model": model,
            "image_mean": image_mean,
            "image_scale": image_scale,
            "target_means": target_means,
            "target_scale": target_scale,
            "template_mode": template_mode,
            "image_templates": image_templates,
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare tiny CNN gaze losses and architectures")
    parser.add_argument("--dataset", type=Path, default=Path("data/shared-eye-legacy-angle-calibration.json"))
    parser.add_argument("--architectures", nargs="+", default=["dense"])
    parser.add_argument("--losses", nargs="+", default=["angle-smooth", "angle-mse", "screen-mse"])
    parser.add_argument("--augmentation-profiles", nargs="+", default=["current"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[6101, 6102, 6103])
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--output", type=Path, default=Path("data/cnn-v4-experiments.json"))
    args = parser.parse_args()
    dataset = json.loads(args.dataset.read_text(encoding="utf-8"))
    data = prepare_shared_dataset(dataset, CNN_MODEL_WIDTH, CNN_MODEL_HEIGHT)
    geometry = dataset_geometry(dataset)
    if len(data.targets) != len(geometry.origins):
        raise ValueError("prepared eye samples and geometry do not align")
    validation_labels = {
        f"grid-{value:02d}" for value in cnn_validation_grid_indices(data.groups)
    }
    validation_indices = np.flatnonzero(np.asarray([
        group in validation_labels for group in data.groups
    ], dtype=bool))
    train_indices = np.flatnonzero(np.asarray([
        group not in validation_labels for group in data.groups
    ], dtype=bool))
    corner_targets = canonical_corner_targets_from_base([
        data.base_images[index] for index in train_indices
    ])
    data = _with_geometry(data, corner_targets)
    results = []
    for architecture in args.architectures:
        for loss_mode in args.losses:
            for augmentation_profile in args.augmentation_profiles:
                for seed in args.seeds:
                    print(
                        f"Training {architecture} / {loss_mode} / "
                        f"{augmentation_profile} / {seed}...",
                        flush=True,
                    )
                    result = train_candidate(
                        dataset, data, geometry, train_indices, validation_indices,
                        architecture, loss_mode, augmentation_profile, seed, args.epochs,
                    )
                    results.append(result)
                    summary = {
                        "architecture": architecture, "loss_mode": loss_mode,
                        "augmentation_profile": augmentation_profile, "seed": seed,
                        "best_epoch": result["best_epoch"], **result["metrics"],
                    }
                    summary.pop("responses", None)
                    summary.pop("per_group_median_px", None)
                    print(json.dumps(summary, ensure_ascii=False), flush=True)
                    args.output.parent.mkdir(parents=True, exist_ok=True)
                    args.output.write_text(json.dumps({
                        "schema": "eyetracing-cnn-v4-experiments-v1",
                        "created_at": datetime.now(timezone.utc).isoformat(),
                        "dataset": str(args.dataset.resolve()),
                        "validation_groups": sorted(validation_labels),
                        "results": results,
                    }, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
