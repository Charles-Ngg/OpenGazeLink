from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from .shared_eye_appearance import (
    CNN_INPUT_CHANNELS,
    CNN_MODEL_HEIGHT,
    CNN_MODEL_WIDTH,
    PREPROCESSING_MODEL,
    augment_cnn_eye_input,
    canonical_corner_targets_from_base,
    canonicalize_base_eye,
    cnn_eye_input,
    deserialize_base_eye,
    resize_model_arrays,
    runtime_cnn_input,
)


SHARED_DATASET_SCHEMA = "eyetracing-shared-eye-angle-dataset-v2-64x36"
SHARED_CNN_SCHEMA = "eyetracing-shared-eye-tiny-cnn-v4-64x36-mask-channel"
CNN_VALIDATION_GRID_INDICES_3X5 = (0, 5, 7, 9, 14)
CNN_VALIDATION_GRID_INDICES_5X5 = (0, 6, 12, 18, 24)


def cnn_validation_grid_indices(groups: Sequence[str]) -> tuple[int, ...]:
    available = {
        int(group.split("-", 1)[1])
        for group in groups
        if group.startswith("grid-") and group.split("-", 1)[1].isdigit()
    }
    return (
        CNN_VALIDATION_GRID_INDICES_5X5
        if max(available, default=-1) >= 24
        else CNN_VALIDATION_GRID_INDICES_3X5
    )


def dataset_landmarker_backend(dataset: dict) -> str:
    normalization = dataset.get("normalization") or {}
    explicit = normalization.get("landmarker_backend")
    if explicit:
        return str(explicit)
    description = str(normalization.get("face_landmarker") or "").lower()
    return "legacy" if "legacy" in description else "tasks"


@dataclass(frozen=True)
class SharedEyePrediction:
    yaw: float
    pitch: float
    nearest_group_distance: float
    fusion_weight: float = 0.5


def fuse_screen_points(points, weights):
    """Fuse already projected eye points; weights express relative reliability."""
    points = np.asarray(points, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if points.shape != (2, 2) or weights.shape != (2,):
        raise ValueError('screen fusion requires exactly two eye points and weights')
    if not np.isfinite(weights).all() or np.any(weights < 0) or weights.sum() <= 0:
        raise ValueError('invalid binocular fusion weights')
    return tuple(np.sum(points * (weights / weights.sum())[:, None], axis=0))


@dataclass(frozen=True)
class PreparedSharedData:
    base_images: tuple[object, ...]
    gray: np.ndarray
    alpha: np.ndarray
    targets: np.ndarray
    groups: tuple[str, ...]
    sides: tuple[str, ...]
    corner_targets: np.ndarray
    lighting: tuple[str, ...] = ()
    conditions: tuple[str, ...] = ()


def prepare_shared_dataset(
    dataset: dict,
    width: int = CNN_MODEL_WIDTH,
    height: int = CNN_MODEL_HEIGHT,
) -> PreparedSharedData:
    if dataset.get("schema") != SHARED_DATASET_SCHEMA:
        raise ValueError("unsupported shared-eye dataset")
    samples = list(dataset.get("samples") or [])
    if not samples:
        raise ValueError("shared-eye dataset contains no samples")
    entries = []
    base_images = []
    for sample in samples:
        lighting = str((sample.get("lighting") or {}).get("name", "unknown"))
        condition = str(sample.get("condition", "unknown"))
        for side in ("right", "left"):
            base = deserialize_base_eye(sample[f"{side}_base_eye"])
            angles = np.asarray(sample[f"{side}_angles"], dtype=np.float64)
            if side == "left":
                angles = np.asarray([-angles[0], angles[1]], dtype=np.float64)
            entries.append((base, angles, str(sample["group"]), side, lighting, condition))
            base_images.append(base)
    corner_targets = canonical_corner_targets_from_base(base_images)
    gray = []
    alpha = []
    targets = []
    groups = []
    sides = []
    lighting_names = []
    conditions = []
    for base, angles, group, side, lighting, condition in entries:
        canonical = canonicalize_base_eye(base, corner_targets)
        image, mask = resize_model_arrays(canonical.gray_base, canonical.alpha_base, width, height)
        gray.append(image)
        alpha.append(mask)
        targets.append(angles)
        groups.append(group)
        sides.append(side)
        lighting_names.append(lighting)
        conditions.append(condition)
    return PreparedSharedData(
        base_images=tuple(base_images),
        gray=np.asarray(gray, dtype=np.float32),
        alpha=np.asarray(alpha, dtype=np.float32),
        targets=np.asarray(targets, dtype=np.float64),
        groups=tuple(groups), sides=tuple(sides), corner_targets=corner_targets,
        lighting=tuple(lighting_names), conditions=tuple(conditions),
    )


def _recanonicalize(data: PreparedSharedData, corner_targets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    gray = []
    alpha = []
    for base in data.base_images:
        canonical = canonicalize_base_eye(base, corner_targets)
        image, mask = resize_model_arrays(
            canonical.gray_base, canonical.alpha_base, data.gray.shape[2], data.gray.shape[1],
        )
        gray.append(image)
        alpha.append(mask)
    return np.asarray(gray, dtype=np.float32), np.asarray(alpha, dtype=np.float32)


def _with_geometry(data: PreparedSharedData, corner_targets: np.ndarray) -> PreparedSharedData:
    gray, alpha = _recanonicalize(data, corner_targets)
    return PreparedSharedData(
        base_images=data.base_images, gray=gray, alpha=alpha,
        targets=data.targets, groups=data.groups, sides=data.sides,
        corner_targets=np.asarray(corner_targets, dtype=np.float64),
        lighting=data.lighting, conditions=data.conditions,
    )


def _error_summary(targets: np.ndarray, predictions: np.ndarray) -> dict:
    errors = np.degrees(np.linalg.norm(predictions - targets, axis=1))
    return {
        "samples": int(len(errors)),
        "median_deg": float(np.median(errors)),
        "mean_deg": float(np.mean(errors)),
        "p95_deg": float(np.percentile(errors, 95.0)),
        "max_deg": float(np.max(errors)),
    }


def _head_pose_summary(dataset: dict) -> dict:
    samples = list(dataset.get("samples") or [])
    if not samples:
        return {}
    values = np.degrees(np.asarray([
        [sample.get("head_yaw", 0.0), sample.get("head_pitch", 0.0), sample.get("head_roll", 0.0)]
        for sample in samples
    ], dtype=np.float64))
    names = ("yaw", "pitch", "roll")
    return {
        name: {
            "median_deg": float(np.median(values[:, index])),
            "p05_deg": float(np.percentile(values[:, index], 5.0)),
            "p95_deg": float(np.percentile(values[:, index], 95.0)),
            "span_deg": float(np.max(values[:, index]) - np.min(values[:, index])),
        }
        for index, name in enumerate(names)
    }


def require_torch():
    try:
        import torch
        from torch import nn
    except ImportError as error:
        raise RuntimeError("PyTorch is required for the shared-eye CNN route") from error
    return torch, nn


def _tiny_cnn_class(torch, nn):
    class TinyEyeCnn(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(CNN_INPUT_CHANNELS, 8, 3, padding=1),
                nn.LeakyReLU(CNN_ACTIVATION_NEGATIVE_SLOPE),
                nn.Conv2d(8, 8, 3, padding=1, groups=8),
                nn.Conv2d(8, 12, 1), nn.LeakyReLU(CNN_ACTIVATION_NEGATIVE_SLOPE),
                nn.AvgPool2d(2),
                nn.Conv2d(12, 12, 3, padding=1, groups=12),
                nn.Conv2d(12, 16, 1), nn.LeakyReLU(CNN_ACTIVATION_NEGATIVE_SLOPE),
                nn.AdaptiveAvgPool2d((12, 20)),
            )
            self.head = nn.Sequential(
                nn.Flatten(),
                nn.Linear(16 * 12 * 20, 16),
                nn.LeakyReLU(CNN_ACTIVATION_NEGATIVE_SLOPE), nn.Linear(16, 2),
            )

        def forward(
            self, values: torch.Tensor,
            lighting_gain: Optional[torch.Tensor] = None,
            lighting_bias: Optional[torch.Tensor] = None,
        ):
            features = self.features[0](values)
            features = self.features[1](features)
            if lighting_gain is not None and lighting_bias is not None:
                features = features * lighting_gain + lighting_bias
            features = self.features[2](features)
            features = self.features[3](features)
            features = self.features[4](features)
            features = self.features[5](features)
            features = self.features[6](features)
            features = self.features[7](features)
            features = self.features[8](features)
            features = self.features[9](features)
            return self.head(features)

    return TinyEyeCnn


def _trace_tiny_cnn(torch, model):
    """Export the fixed CNN graph without requiring access to Python source."""
    model = model.cpu().eval()
    values = torch.zeros(
        (1, CNN_INPUT_CHANNELS, CNN_MODEL_HEIGHT, CNN_MODEL_WIDTH),
        dtype=torch.float32,
    )
    lighting_gain = torch.ones((1, 8, 1, 1), dtype=torch.float32)
    lighting_bias = torch.zeros((1, 8, 1, 1), dtype=torch.float32)
    return torch.jit.trace(
        model,
        (values, lighting_gain, lighting_bias),
        check_trace=True,
        strict=True,
    )


CNN_AUGMENTATION_MODES = ("geometry", "geometry")
CNN_VALIDATION_SEEDS = (4022, 5022, 6022)
CNN_ACTIVATION_NEGATIVE_SLOPE = 0.05
CNN_VALIDATION_GAIN_PENALTY_DEG = 5.0


def _cnn_arrays(
    data: PreparedSharedData, indices: np.ndarray, seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    images = []
    targets = []
    for index in indices.tolist():
        images.append(cnn_eye_input(data.gray[index], data.alpha[index]))
        targets.append(data.targets[index])
        for mode in CNN_AUGMENTATION_MODES:
            images.append(augment_cnn_eye_input(
                data.gray[index], data.alpha[index], rng, mode=mode,
            ))
            targets.append(data.targets[index])
    return np.asarray(images, dtype=np.float32), np.asarray(targets, dtype=np.float32)


def _balanced_source_weights(
    data: PreparedSharedData,
    indices: np.ndarray,
    pose_mass: float = 0.20,
) -> np.ndarray:
    """Give static and head-pose conditions fixed mass regardless of frame count."""
    conditions = np.asarray([
        "pose" if str(data.groups[index]).startswith("pose-") else "static"
        for index in indices.tolist()
    ], dtype=object)
    available = list(dict.fromkeys(conditions.tolist()))
    pose_mass = float(np.clip(pose_mass, 0.0, 1.0))
    condition_mass = {"static": 1.0 - pose_mass, "pose": pose_mass}
    if len(available) == 1:
        condition_mass[available[0]] = 1.0
    weights = np.zeros(len(indices), dtype=np.float64)
    for condition in available:
        condition_indices = np.flatnonzero(conditions == condition)
        groups = np.asarray([data.groups[indices[index]] for index in condition_indices], dtype=object)
        unique_groups = list(dict.fromkeys(groups.tolist()))
        mass = condition_mass[condition]
        for group in unique_groups:
            members = condition_indices[groups == group]
            weights[members] = mass / max(len(unique_groups) * len(members), 1)
    return weights / np.mean(weights)


def _cnn_gain_summary(targets: np.ndarray, predictions: np.ndarray) -> dict:
    targets = np.asarray(targets, dtype=np.float64)
    predictions = np.asarray(predictions, dtype=np.float64)
    if targets.shape != predictions.shape or targets.ndim != 2 or targets.shape[1] != 2:
        raise ValueError("CNN gain diagnostics require matching [N,2] arrays")
    axes = {}
    collapsed = False
    for axis, name in enumerate(("yaw", "pitch")):
        expected = targets[:, axis]
        actual = predictions[:, axis]
        design = np.column_stack([expected, np.ones(len(expected), dtype=np.float64)])
        slope, intercept = np.linalg.lstsq(design, actual, rcond=None)[0]
        expected_std = float(np.std(expected))
        actual_std = float(np.std(actual))
        std_ratio = actual_std / max(expected_std, 1e-12)
        correlation = (
            float(np.corrcoef(expected, actual)[0, 1])
            if expected_std > 1e-9 and actual_std > 1e-9 else 0.0
        )
        axis_collapsed = bool(std_ratio < 0.25 or slope < 0.25 or correlation < 0.25)
        collapsed = collapsed or axis_collapsed
        axes[name] = {
            "slope": float(slope),
            "intercept_deg": float(np.degrees(intercept)),
            "std_ratio": float(std_ratio),
            "correlation": correlation,
            "collapsed": axis_collapsed,
        }
    return {"collapsed": collapsed, "axes": axes}


def _predict_cnn_arrays(model, images: np.ndarray, fit: dict) -> np.ndarray:
    torch, _ = require_torch()
    image_mean = np.asarray(fit["image_mean"], dtype=np.float32).reshape(
        1, CNN_INPUT_CHANNELS, 1, 1,
    )
    image_scale = np.asarray(fit["image_scale"], dtype=np.float32).reshape(
        1, CNN_INPUT_CHANNELS, 1, 1,
    )
    with torch.inference_mode():
        normalized = model(torch.from_numpy((images - image_mean) / image_scale)).numpy()
    return (
        normalized * np.asarray(fit["target_scale"], dtype=np.float32)
        + np.asarray(fit["target_mean"], dtype=np.float32)
    )


def _train_cnn_arrays(
    images: np.ndarray, targets: np.ndarray,
    validation_images: np.ndarray | None, validation_targets: np.ndarray | None,
    epochs: int, patience: int, seed: int,
    sample_weights: np.ndarray | None = None,
) -> tuple[object, dict]:
    torch, nn = require_torch()
    torch.manual_seed(seed)
    # Runtime inference stays single-threaded, but calibration training should
    # not inherit the launcher's one-thread game-friendly limit.
    torch.set_num_threads(max(1, min(8, int(os.cpu_count() or 1))))
    model = _tiny_cnn_class(torch, nn)()
    if images.ndim != 4 or images.shape[1:] != (
        CNN_INPUT_CHANNELS, CNN_MODEL_HEIGHT, CNN_MODEL_WIDTH,
    ):
        raise ValueError(
            f"CNN inputs must be [N,{CNN_INPUT_CHANNELS},{CNN_MODEL_HEIGHT},{CNN_MODEL_WIDTH}]"
        )
    image_mean = np.mean(images, axis=(0, 2, 3)).astype(np.float32)
    image_scale = np.maximum(np.std(images, axis=(0, 2, 3)), 1e-3).astype(np.float32)
    image_mean_view = image_mean.reshape(1, CNN_INPUT_CHANNELS, 1, 1)
    image_scale_view = image_scale.reshape(1, CNN_INPUT_CHANNELS, 1, 1)
    target_mean = np.mean(targets, axis=0).astype(np.float32)
    target_scale = np.maximum(np.std(targets, axis=0), np.radians(1.0)).astype(np.float32)
    train_x = torch.from_numpy((images - image_mean_view) / image_scale_view)
    train_y = torch.from_numpy((targets - target_mean) / target_scale)
    views_per_source = 1 + len(CNN_AUGMENTATION_MODES)
    if len(train_x) % views_per_source != 0:
        raise ValueError(
            f"CNN training inputs must contain one original and {len(CNN_AUGMENTATION_MODES)} augmented views"
        )
    source_count = len(train_x) // views_per_source
    train_x = train_x.reshape(
        source_count, views_per_source, CNN_INPUT_CHANNELS, CNN_MODEL_HEIGHT, CNN_MODEL_WIDTH,
    )
    train_y = train_y.reshape(source_count, views_per_source, 2)
    if sample_weights is None:
        source_weights = np.ones(source_count, dtype=np.float64)
    else:
        source_weights = np.asarray(sample_weights, dtype=np.float64).reshape(-1)
        if len(source_weights) != source_count:
            raise ValueError("CNN sample weights must match source samples")
        if np.any(source_weights <= 0.0) or not np.isfinite(source_weights).all():
            raise ValueError("CNN sample weights must be finite and positive")
    sampling_probability = source_weights / np.sum(source_weights)
    validation_x = None if validation_images is None else torch.from_numpy(
        (validation_images - image_mean_view) / image_scale_view
    )
    validation_y = None if validation_targets is None else torch.from_numpy((validation_targets - target_mean) / target_scale)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_function = nn.SmoothL1Loss(beta=0.35)
    rng = np.random.default_rng(seed)
    batch_size = 32
    consistency_weight = 0.05
    delta_weight = 0.15
    best_state = None
    best_epoch = epochs
    best_score = float("inf")
    stale = 0
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        order = rng.choice(
            source_count, size=source_count, replace=True, p=sampling_probability,
        )
        losses = []
        supervised_losses = []
        consistency_losses = []
        delta_losses = []
        for start in range(0, len(order), batch_size):
            batch = torch.from_numpy(order[start:start + batch_size].astype(np.int64))
            optimizer.zero_grad(set_to_none=True)
            batch_x = train_x[batch]
            batch_y = train_y[batch]
            predicted = model(batch_x.flatten(0, 1)).reshape(len(batch), views_per_source, 2)
            supervised = loss_function(predicted, batch_y)
            reference = predicted[:, :1, :].expand_as(predicted[:, 1:, :])
            consistency = loss_function(predicted[:, 1:, :], reference)
            original_prediction = predicted[:, 0, :]
            original_target = batch_y[:, 0, :]
            delta = loss_function(
                original_prediction - original_prediction.mean(dim=0, keepdim=True),
                original_target - original_target.mean(dim=0, keepdim=True),
            )
            loss = supervised + consistency_weight * consistency + delta_weight * delta
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
            supervised_losses.append(float(supervised.detach()))
            consistency_losses.append(float(consistency.detach()))
            delta_losses.append(float(delta.detach()))
        record = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "supervised_loss": float(np.mean(supervised_losses)),
            "consistency_loss": float(np.mean(consistency_losses)),
            "delta_loss": float(np.mean(delta_losses)),
        }
        if validation_x is not None and validation_y is not None:
            model.eval()
            with torch.inference_mode():
                validation_normalized = model(validation_x).numpy()
            validation_prediction = validation_normalized * target_scale + target_mean
            errors = np.degrees(np.linalg.norm(validation_prediction - validation_targets, axis=1))
            score = float(np.median(errors) + 0.25 * np.percentile(errors, 95.0))
            record.update({
                "validation_median_deg": float(np.median(errors)),
                "validation_p95_deg": float(np.percentile(errors, 95.0)),
                "validation_max_deg": float(np.max(errors)),
            })
            if score < best_score - 1e-5:
                best_score = score
                best_epoch = epoch
                best_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
                stale = 0
            else:
                stale += 1
                if stale >= patience:
                    history.append(record)
                    break
        history.append(record)
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {
        "image_mean": image_mean.tolist(), "image_scale": image_scale.tolist(),
        "target_mean": target_mean.tolist(), "target_scale": target_scale.tolist(),
        "consistency_weight": consistency_weight,
        "delta_weight": delta_weight,
        "best_epoch": int(best_epoch),
        "best_validation": next(
            (record for record in history if record["epoch"] == best_epoch), None,
        ),
        "history": history,
    }


def _fit_cnn_lighting_adapter(
    model, images: np.ndarray, targets: np.ndarray, fit: dict,
    seed: int,
) -> tuple[dict, dict]:
    torch, nn = require_torch()
    torch.manual_seed(seed)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    image_mean = np.asarray(fit["image_mean"], dtype=np.float32).reshape(
        1, CNN_INPUT_CHANNELS, 1, 1,
    )
    image_scale = np.asarray(fit["image_scale"], dtype=np.float32).reshape(
        1, CNN_INPUT_CHANNELS, 1, 1,
    )
    target_mean = np.asarray(fit["target_mean"], dtype=np.float32)
    target_scale = np.asarray(fit["target_scale"], dtype=np.float32)
    values = torch.from_numpy((images - image_mean) / image_scale)
    expected = torch.from_numpy((targets - target_mean) / target_scale)
    gain = nn.Parameter(torch.ones(1, 8, 1, 1))
    bias = nn.Parameter(torch.zeros(1, 8, 1, 1))
    optimizer = torch.optim.Adam([gain, bias], lr=0.01)
    loss_function = nn.SmoothL1Loss(beta=0.35)
    best_state = None
    best_loss = float("inf")
    stale = 0
    for epoch in range(400):
        prediction = model(values, gain, bias)
        supervised = loss_function(prediction, expected)
        regularization = 0.03 * (
            (gain - 1.0).square().mean() + bias.square().mean()
        )
        loss = supervised + regularization
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        current = float(supervised.detach())
        if current < best_loss - 1e-6:
            best_loss = current
            best_state = (gain.detach().clone(), bias.detach().clone())
            stale = 0
        else:
            stale += 1
            if stale >= 60:
                break
    if best_state is not None:
        gain_value, bias_value = best_state
    else:
        gain_value, bias_value = gain.detach(), bias.detach()
    with torch.inference_mode():
        prediction = model(values, gain_value, bias_value).numpy()
    angles = prediction * target_scale + target_mean
    errors = np.degrees(np.linalg.norm(angles - targets, axis=1))
    profile = {
        "gain": gain_value.reshape(-1).numpy().astype(float).tolist(),
        "bias": bias_value.reshape(-1).numpy().astype(float).tolist(),
        "samples": int(len(images)),
    }
    diagnostics = {
        "epochs": epoch + 1,
        "normalized_loss": best_loss,
        "error_deg": {
            "median": float(np.median(errors)),
            "p95": float(np.percentile(errors, 95.0)),
            "max": float(np.max(errors)),
        },
    }
    return profile, diagnostics


class SharedTinyCnnModel:
    def __init__(self, metadata: dict, module) -> None:
        if metadata.get("schema") != SHARED_CNN_SCHEMA:
            raise ValueError("unsupported shared-eye CNN model")
        self.metadata = metadata
        self.module = module
        self.landmarker_backend = str(metadata["preprocessing"].get("landmarker_backend", "tasks"))
        self.corner_targets = np.asarray(metadata["preprocessing"]["corner_targets"], dtype=np.float64)
        self.image_mean = np.asarray(
            metadata["normalization"]["image_mean"], dtype=np.float32,
        ).reshape(1, CNN_INPUT_CHANNELS, 1, 1)
        self.image_scale = np.asarray(
            metadata["normalization"]["image_scale"], dtype=np.float32,
        ).reshape(1, CNN_INPUT_CHANNELS, 1, 1)
        self.target_mean = np.asarray(metadata["normalization"]["target_mean"], dtype=np.float32)
        self.target_scale = np.asarray(metadata["normalization"]["target_scale"], dtype=np.float32)
        self.lighting_profiles = dict(metadata.get("lighting_profiles") or {})
        self.active_lighting_profile = "reference"
        self._lighting_gain = None
        self._lighting_bias = None
        self.set_lighting_profile(str(metadata.get("default_lighting_profile", "reference")))

    def set_lighting_profile(self, name: str) -> None:
        torch, _ = require_torch()
        profile = self.lighting_profiles.get(name)
        if profile is None:
            if name != "reference" and self.lighting_profiles:
                raise ValueError(f"unknown lighting profile: {name}")
            self.active_lighting_profile = "reference"
            self._lighting_gain = torch.ones((1, 8, 1, 1), dtype=torch.float32)
            self._lighting_bias = torch.zeros((1, 8, 1, 1), dtype=torch.float32)
            return
        gain = np.asarray(profile["gain"], dtype=np.float32).reshape(1, 8, 1, 1)
        bias = np.asarray(profile["bias"], dtype=np.float32).reshape(1, 8, 1, 1)
        self.active_lighting_profile = name
        self._lighting_gain = torch.from_numpy(gain)
        self._lighting_bias = torch.from_numpy(bias)

    def model_image(self, patch, side: str) -> np.ndarray:
        return runtime_cnn_input(patch, side == "right", self.corner_targets)

    def predict_images(self, images: Sequence[np.ndarray], sides: Sequence[str]) -> list[SharedEyePrediction]:
        torch, _ = require_torch()
        values = np.asarray(images, dtype=np.float32)
        if values.ndim != 4 or values.shape[1:] != (
            CNN_INPUT_CHANNELS, CNN_MODEL_HEIGHT, CNN_MODEL_WIDTH,
        ):
            raise ValueError(
                f"CNN inputs must be [N,{CNN_INPUT_CHANNELS},{CNN_MODEL_HEIGHT},{CNN_MODEL_WIDTH}]"
            )
        tensor = torch.from_numpy((values - self.image_mean) / self.image_scale)
        with torch.inference_mode():
            normalized = self.module(
                tensor, self._lighting_gain, self._lighting_bias,
            ).numpy()
        angles = normalized * self.target_scale + self.target_mean
        return [
            SharedEyePrediction(
                float(angle[0] if side == "right" else -angle[0]),
                float(angle[1]), float("nan"),
            )
            for angle, side in zip(angles, sides)
        ]

    @classmethod
    def load(cls, metadata_path: Path, module_path: Path | None = None) -> "SharedTinyCnnModel":
        torch, _ = require_torch()
        torch.set_num_threads(1)
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        path = module_path or metadata_path.with_name(metadata["module_file"])
        module = torch.jit.load(str(path), map_location="cpu").eval()
        return cls(metadata, module)

    @classmethod
    def fit_lighting_profile(
        cls, dataset: dict, metadata_path: Path, module_path: Path,
        profile_name: str,
    ) -> tuple["SharedTinyCnnModel", dict]:
        if not profile_name or profile_name == "reference":
            raise ValueError("lighting profile name must be non-reference")
        model = cls.load(metadata_path, module_path)
        architecture = str((model.metadata.get("architecture") or {}).get("kind", ""))
        if "lighting_film" not in architecture:
            raise RuntimeError("base CNN does not support lighting adapters; run a full calibration first")
        profile_samples = []
        for sample in dataset.get("samples", []):
            lighting = sample.get("lighting") or {}
            lighting_name = (
                lighting.get("name") if isinstance(lighting, dict) else str(lighting)
            )
            if sample.get("condition") == "lighting_anchor" and lighting_name == profile_name:
                profile_samples.append(sample)
        profile_dataset = dict(dataset)
        profile_dataset["samples"] = profile_samples
        data = prepare_shared_dataset(profile_dataset, CNN_MODEL_WIDTH, CNN_MODEL_HEIGHT)
        data = _with_geometry(data, model.corner_targets)
        conditions = np.asarray(data.conditions, dtype=object)
        lighting = np.asarray(data.lighting, dtype=object)
        indices = np.flatnonzero(
            (conditions == "lighting_anchor") & (lighting == profile_name)
        )
        if len(indices) < 20:
            raise RuntimeError(
                f"lighting profile {profile_name} has only {len(indices)} eye samples; need at least 20"
            )
        images = np.asarray([
            cnn_eye_input(data.gray[index], data.alpha[index])
            for index in indices.tolist()
        ], dtype=np.float32)
        fit = {
            "image_mean": model.image_mean.reshape(-1).tolist(),
            "image_scale": model.image_scale.reshape(-1).tolist(),
            "target_mean": model.target_mean.tolist(),
            "target_scale": model.target_scale.tolist(),
        }
        profile, diagnostics = _fit_cnn_lighting_adapter(
            model.module, images, data.targets[indices].astype(np.float32), fit, 7121,
        )
        metadata = dict(model.metadata)
        profiles = dict(metadata.get("lighting_profiles") or {})
        profiles[profile_name] = profile
        metadata["lighting_profiles"] = profiles
        metadata.setdefault("diagnostics", {}).setdefault("lighting_adapters", {})[
            profile_name
        ] = diagnostics
        temporary = metadata_path.with_suffix(metadata_path.suffix + ".tmp")
        temporary.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(metadata_path)
        return cls(metadata, model.module), diagnostics

    @classmethod
    def fit_dataset(
        cls, dataset: dict, metadata_path: Path, module_path: Path,
    ) -> tuple["SharedTinyCnnModel", dict]:
        torch, _ = require_torch()
        data = prepare_shared_dataset(dataset, CNN_MODEL_WIDTH, CNN_MODEL_HEIGHT)
        groups = np.asarray(data.groups, dtype=object)
        conditions = np.asarray(
            data.conditions if len(data.conditions) == len(data.targets)
            else ("unknown",) * len(data.targets),
            dtype=object,
        )
        base_indices = np.flatnonzero(conditions != "lighting_anchor")
        if len(base_indices) == 0:
            base_indices = np.arange(len(data.targets), dtype=np.int64)
        validation_labels = {
            f"grid-{value:02d}" for value in cnn_validation_grid_indices(data.groups)
        }
        base_mask = np.zeros(len(data.targets), dtype=bool)
        base_mask[base_indices] = True
        validation_indices = np.flatnonzero(base_mask & np.asarray([
            group in validation_labels for group in data.groups
        ], dtype=bool))
        train_indices = np.flatnonzero(base_mask & np.asarray([
            group not in validation_labels for group in data.groups
        ], dtype=bool))
        if len(validation_indices) == 0 or len(train_indices) == 0:
            unique = list(dict.fromkeys(data.groups))
            validation_labels = set(unique[::5])
            validation_indices = np.flatnonzero(base_mask & np.asarray([
                group in validation_labels for group in data.groups
            ], dtype=bool))
            train_indices = np.flatnonzero(base_mask & np.asarray([
                group not in validation_labels for group in data.groups
            ], dtype=bool))
        validation_corner_targets = canonical_corner_targets_from_base(
            [data.base_images[index] for index in train_indices]
        )
        validation_data = _with_geometry(data, validation_corner_targets)
        train_images, train_targets = _cnn_arrays(validation_data, train_indices, 4021)
        validation_images = np.asarray([
            cnn_eye_input(validation_data.gray[index], validation_data.alpha[index])
            for index in validation_indices
        ], dtype=np.float32)
        validation_targets = validation_data.targets[validation_indices].astype(np.float32)
        validation_candidates = []
        validation_weights = _balanced_source_weights(validation_data, train_indices)
        for seed in CNN_VALIDATION_SEEDS:
            validation_model, validation_fit = _train_cnn_arrays(
                train_images, train_targets, validation_images, validation_targets,
                epochs=120, patience=18, seed=seed,
                sample_weights=validation_weights,
            )
            validation_prediction = _predict_cnn_arrays(
                validation_model, validation_images, validation_fit,
            )
            holdout = _error_summary(validation_targets, validation_prediction)
            gain = _cnn_gain_summary(validation_targets, validation_prediction)
            error_score = float(
                holdout["median_deg"] + 0.25 * holdout["p95_deg"]
            )
            gain_deviation = max(
                abs(float(axis["slope"]) - 1.0)
                for axis in gain["axes"].values()
            )
            gain_penalty = CNN_VALIDATION_GAIN_PENALTY_DEG * gain_deviation
            validation_candidates.append({
                "seed": int(seed), "fit": validation_fit,
                "holdout": holdout, "gain": gain,
                "error_score": error_score,
                "gain_penalty": float(gain_penalty),
                "score": float(error_score + gain_penalty),
            })
        usable_candidates = [
            candidate for candidate in validation_candidates
            if not candidate["gain"]["collapsed"]
        ]
        if not usable_candidates:
            details = ", ".join(
                f"seed {candidate['seed']}: {candidate['holdout']['median_deg']:.2f} deg"
                for candidate in validation_candidates
            )
            raise RuntimeError(f"all CNN validation restarts collapsed ({details})")
        selected_validation = min(
            usable_candidates, key=lambda candidate: (candidate["score"], candidate["seed"]),
        )
        validation_fit = selected_validation["fit"]
        selected_seed = int(selected_validation["seed"])
        selected_epochs = max(8, int(validation_fit["best_epoch"]))
        final_corner_targets = canonical_corner_targets_from_base([
            data.base_images[index] for index in base_indices
        ])
        data = _with_geometry(data, final_corner_targets)
        all_indices = base_indices
        final_images, final_targets = _cnn_arrays(data, all_indices, 5021)
        final_model, final_fit = _train_cnn_arrays(
            final_images, final_targets, None, None,
            epochs=selected_epochs, patience=selected_epochs + 1, seed=selected_seed,
            sample_weights=_balanced_source_weights(data, all_indices),
        )
        original_images = np.asarray([
            cnn_eye_input(data.gray[index], data.alpha[index])
            for index in all_indices
        ], dtype=np.float32)
        final_model.eval()
        original_prediction = _predict_cnn_arrays(final_model, original_images, final_fit)
        final_gain = _cnn_gain_summary(data.targets[all_indices], original_prediction)
        if final_gain["collapsed"]:
            raise RuntimeError(
                f"final CNN collapsed after selecting validation seed {selected_seed}"
            )
        training_errors = np.degrees(np.linalg.norm(
            original_prediction - data.targets[all_indices], axis=1,
        ))
        lighting_profiles = {
            "reference": {
                "gain": np.ones(8, dtype=float).tolist(),
                "bias": np.zeros(8, dtype=float).tolist(),
                "samples": int(len(all_indices)),
            },
        }
        lighting_diagnostics = {}
        lighting_values = np.asarray(
            data.lighting if len(data.lighting) == len(data.targets)
            else ("unknown",) * len(data.targets),
            dtype=object,
        )
        for profile_name in sorted(set(lighting_values[conditions == "lighting_anchor"].tolist())):
            profile_indices = np.flatnonzero(
                (conditions == "lighting_anchor") & (lighting_values == profile_name)
            )
            if len(profile_indices) < 16:
                continue
            profile_images = np.asarray([
                cnn_eye_input(data.gray[index], data.alpha[index])
                for index in profile_indices.tolist()
            ], dtype=np.float32)
            profile, profile_diagnostics = _fit_cnn_lighting_adapter(
                final_model, profile_images,
                data.targets[profile_indices].astype(np.float32),
                final_fit, 6021 + len(lighting_profiles),
            )
            lighting_profiles[profile_name] = profile
            lighting_diagnostics[profile_name] = profile_diagnostics
        metadata = {
            "schema": SHARED_CNN_SCHEMA,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "module_file": module_path.name,
            "screen": dict(dataset["screen"]),
            "screen_diagonal_inches": float(dataset["screen_diagonal_inches"]),
            "screen_camera_mount": dataset.get("screen_camera_mount", "legacy_camera_center_v0"),
            "camera_position_screen_cm": dataset.get("camera_position_screen_cm"),
            "screen_camera_origin_cm": dataset.get(
                "screen_camera_origin_cm", [0.0, 0.0, 0.0],
            ),
            "input_source": dataset.get("input_source", "phone_udp"),
            "windows_camera": dataset.get("windows_camera"),
            "preprocessing": {
                "model": PREPROCESSING_MODEL,
                "canonical_eye": "right; left images mirrored and left yaw negated",
                "corner_targets": data.corner_targets.astype(float).tolist(),
                "feature_size": [CNN_MODEL_WIDTH, CNN_MODEL_HEIGHT],
                "input_channels": ["robust_normalized_intensity", "soft_aperture_alpha"],
                "photometric_normalization": "valid-pixel median and p10-p90 scale",
                "augmentation_views": ["original", *CNN_AUGMENTATION_MODES],
                "mask_augmentation": "conservative aperture inset, morphology, and feather jitter",
                "photometric_augmentation": "disabled; real light anchors train separate lighting domains",
                "landmarker_backend": dataset_landmarker_backend(dataset),
            },
            "normalization": {
                "image_mean": final_fit["image_mean"], "image_scale": final_fit["image_scale"],
                "target_mean": final_fit["target_mean"], "target_scale": final_fit["target_scale"],
            },
            "default_lighting_profile": "reference",
            "lighting_profiles": lighting_profiles,
            "architecture": {
                "kind": "depthwise_tiny_cnn_v3_64x36_adaptive_spatial_with_lighting_film",
                "parameters": int(sum(parameter.numel() for parameter in final_model.parameters())),
                "input": [CNN_INPUT_CHANNELS, CNN_MODEL_HEIGHT, CNN_MODEL_WIDTH], "output": [2],
                "activation": {
                    "kind": "leaky_relu",
                    "negative_slope": CNN_ACTIVATION_NEGATIVE_SLOPE,
                },
                "lighting_adapter": "first-convolution FiLM, 8 gain + 8 bias parameters per profile",
            },
            "diagnostics": {
                "raw_frame_samples": len(dataset.get("samples") or []),
                "shared_eye_samples": len(data.targets),
                "training_samples_after_augmentation": len(final_targets),
                "validation_groups": sorted(validation_labels),
                "validation_samples": len(validation_indices),
                "selected_seed": selected_seed,
                "selected_epochs": selected_epochs,
                "validation_training": validation_fit,
                "validation_restarts": [
                    {
                        "seed": candidate["seed"],
                        "selected": candidate is selected_validation,
                        "best_epoch": int(candidate["fit"]["best_epoch"]),
                        "error_score": candidate["error_score"],
                        "gain_penalty": candidate["gain_penalty"],
                        "score": candidate["score"],
                        "holdout": candidate["holdout"],
                        "gain": candidate["gain"],
                    }
                    for candidate in validation_candidates
                ],
                "final_training": final_fit,
                "final_training_gain": final_gain,
                "training_loss": "smooth_l1_supervised_plus_0.05_augmented_view_consistency_and_0.15_centered_delta",
                "sampling": "80% static / 20% head-pose when both conditions exist; groups equal within condition",
                "training_error_deg": {
                    "median": float(np.median(training_errors)),
                    "mean": float(np.mean(training_errors)),
                    "p95": float(np.percentile(training_errors, 95.0)),
                    "max": float(np.max(training_errors)),
                },
                "comparison_holdout": selected_validation["holdout"],
                "captured_head_pose": _head_pose_summary(dataset),
                "lighting_adapters": lighting_diagnostics,
            },
        }
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        scripted = _trace_tiny_cnn(torch, final_model)
        scripted.save(str(module_path))
        metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        return cls(metadata, scripted), metadata["diagnostics"]
