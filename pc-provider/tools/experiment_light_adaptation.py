from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from opengazelink_pc.shared_eye_appearance import (
    CNN_MODEL_HEIGHT,
    CNN_MODEL_WIDTH,
    canonical_corner_targets_from_base,
    cnn_eye_input,
)
from opengazelink_pc.shared_eye_models import (
    PreparedSharedData,
    _with_geometry,
    prepare_shared_dataset,
    require_torch,
)
from experiment_cnn_v4 import (
    dataset_geometry,
    evaluate,
    physical_screen_size,
    train_candidate,
    torch_screen_points,
)


def synthetic_light_shift(gray: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Create one deterministic local shadow/highlight domain for an offline smoke test."""
    height, width = gray.shape
    yy, xx = np.mgrid[-1.0:1.0:complex(height), -1.0:1.0:complex(width)]
    values = np.clip(gray.astype(np.float32) / 255.0, 0.0, 1.0)
    values = np.power(values, 0.82)
    directional = 1.0 + 0.24 * xx - 0.16 * yy
    shadow = 1.0 - 0.32 * np.exp(
        -(((xx + 0.42) ** 2) / 0.22 + ((yy - 0.05) ** 2) / 0.80)
    )
    highlight = 1.0 + 0.18 * np.exp(
        -(((xx - 0.38) ** 2) / 0.16 + ((yy + 0.20) ** 2) / 0.25)
    )
    shifted = np.clip(values * directional * shadow * highlight, 0.0, 1.0)
    # Keep the outside of the aperture harmless; cnn_eye_input masks it later.
    return (shifted * 255.0 * np.clip(alpha + (alpha <= 0.0), 0.0, 1.0)).astype(np.float32)


def model_angles(
    runtime: dict, data: PreparedSharedData, indices: np.ndarray, geometry,
    model_override=None, track_grad: bool = False,
):
    torch, _ = require_torch()
    images = np.asarray([
        cnn_eye_input(data.gray[index], data.alpha[index])
        for index in indices.tolist()
    ], dtype=np.float32)
    sides = (geometry.yaw_signs[indices] < 0.0).astype(np.int64)
    centered = images - runtime["image_mean"]
    inputs = torch.from_numpy(centered / runtime["image_scale"].reshape(1, 2, 1, 1))
    model = model_override or runtime["model"]
    if track_grad:
        normalized = model(
            inputs, torch.from_numpy(geometry.yaw_signs[indices]).float(),
        )
    else:
        with torch.inference_mode():
            normalized = model(
                inputs, torch.from_numpy(geometry.yaw_signs[indices]).float(),
            )
    return normalized, sides


def fit_output_adapter(runtime: dict, data: PreparedSharedData, geometry, indices: np.ndarray):
    torch, nn = require_torch()
    normalized, side_indices = model_angles(runtime, data, indices, geometry)
    target_means = torch.from_numpy(runtime["target_means"][side_indices]).to(normalized)
    target_scale = torch.from_numpy(runtime["target_scale"]).to(normalized)
    base_angles = normalized * target_scale + target_means
    targets = torch.from_numpy(data.targets[indices].astype(np.float32)).to(base_angles)
    source_indices = np.asarray(indices, dtype=np.int64)
    screen_size = physical_screen_size(_DATASET)
    screen = _DATASET["screen"]
    screen_resolution = (int(screen["width"]), int(screen["height"]))
    target_fraction = geometry.target_pixels[indices] / np.asarray([
        max(screen_resolution[0] - 1, 1), max(screen_resolution[1] - 1, 1),
    ], dtype=np.float32)
    target_points = torch.from_numpy(target_fraction).to(base_angles)
    screen_scale = torch.tensor([0.1, 0.1], dtype=base_angles.dtype)

    matrix_delta = nn.Parameter(torch.zeros((2, 2), dtype=base_angles.dtype))
    bias = nn.Parameter(torch.zeros(2, dtype=base_angles.dtype))
    optimizer = torch.optim.Adam([matrix_delta, bias], lr=0.02)
    identity = torch.eye(2, dtype=base_angles.dtype)
    for _ in range(500):
        transformed = base_angles @ (identity + matrix_delta).T + bias
        predicted_points = torch_screen_points(
            torch, transformed, geometry, source_indices, screen_size,
        )
        error = ((predicted_points - target_points) / screen_scale).square().mean()
        regularization = 0.02 * matrix_delta.square().mean() + 0.02 * bias.square().mean()
        loss = error + regularization
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    return matrix_delta.detach(), bias.detach()


def apply_adapter(
    runtime: dict, data: PreparedSharedData, geometry, indices: np.ndarray,
    matrix_delta, bias,
):
    torch, _ = require_torch()
    normalized, side_indices = model_angles(runtime, data, indices, geometry)
    target_means = torch.from_numpy(runtime["target_means"][side_indices]).to(normalized)
    target_scale = torch.from_numpy(runtime["target_scale"]).to(normalized)
    base_angles = normalized * target_scale + target_means
    identity = torch.eye(2, dtype=base_angles.dtype)
    return base_angles @ (identity + matrix_delta).T + bias


def evaluate_domain(
    runtime, data, geometry, indices, matrix_delta=None, bias=None,
    model_override=None,
):
    torch, _ = require_torch()
    angles = model_angles(
        runtime, data, indices, geometry, model_override=model_override,
    )[0]
    side_indices = (geometry.yaw_signs[indices] < 0.0).astype(np.int64)
    means = torch.from_numpy(runtime["target_means"][side_indices]).to(angles)
    scale = torch.from_numpy(runtime["target_scale"]).to(angles)
    predictions = angles * scale + means
    if matrix_delta is not None:
        identity = torch.eye(2, dtype=predictions.dtype)
        predictions = predictions @ (identity + matrix_delta).T + bias
    resolution = _DATASET["screen"]
    return evaluate(
        _DATA.targets[indices].astype(np.float32), predictions.detach().numpy(),
        tuple(_DATA.groups[index] for index in indices.tolist()),
        _GEOMETRY.select(indices), physical_screen_size(_DATASET),
        (int(resolution["width"]), int(resolution["height"])),
    )


def make_feature_adapter(base_model, channel_count: int):
    torch, nn = require_torch()

    class FeatureAdapter(nn.Module):
        def __init__(self):
            super().__init__()
            self.base = base_model
            for parameter in self.base.parameters():
                parameter.requires_grad_(False)
            self.gain = nn.Parameter(torch.ones(1, channel_count, 1, 1))
            self.bias = nn.Parameter(torch.zeros(1, channel_count, 1, 1))

        def forward(self, values, sides):
            del sides
            first = self.base.features[0](values)
            first = self.base.features[1](first)
            first = first * self.gain + self.bias
            features = self.base.features[2:](first)
            return self.base.head(features)

    return FeatureAdapter()


def make_multilayer_feature_adapter(base_model):
    torch, nn = require_torch()
    channel_counts = (12, 16, 24, 12)

    class MultiLayerFeatureAdapter(nn.Module):
        def __init__(self):
            super().__init__()
            self.base = base_model
            for parameter in self.base.parameters():
                parameter.requires_grad_(False)
            self.gains = nn.ParameterList([
                nn.Parameter(torch.ones(1, channels, 1, 1))
                for channels in channel_counts
            ])
            self.biases = nn.ParameterList([
                nn.Parameter(torch.zeros(1, channels, 1, 1))
                for channels in channel_counts
            ])

        def forward(self, values, sides):
            del sides
            features = values
            for block in range(4):
                features = self.base.features[block * 2](features)
                features = self.base.features[block * 2 + 1](features)
                features = features * self.gains[block] + self.biases[block]
            return self.base.head(features)

    return MultiLayerFeatureAdapter()


def fit_feature_adapter(
    runtime, data: PreparedSharedData, geometry, indices: np.ndarray,
    multilayer: bool = False,
):
    torch, _ = require_torch()
    adapter = (
        make_multilayer_feature_adapter(runtime["model"])
        if multilayer else make_feature_adapter(runtime["model"], 12)
    )
    images = np.asarray([
        cnn_eye_input(data.gray[index], data.alpha[index])
        for index in indices.tolist()
    ], dtype=np.float32)
    centered = images - runtime["image_mean"]
    inputs = torch.from_numpy(
        centered / runtime["image_scale"].reshape(1, 2, 1, 1)
    )
    side_values = torch.from_numpy(geometry.yaw_signs[indices]).float()
    side_indices = (geometry.yaw_signs[indices] < 0.0).astype(np.int64)
    normalized = adapter(inputs, side_values)
    target_means = torch.from_numpy(runtime["target_means"][side_indices]).to(normalized)
    target_scale = torch.from_numpy(runtime["target_scale"]).to(normalized)
    targets = torch.from_numpy(data.targets[indices].astype(np.float32)).to(normalized)
    screen_size = physical_screen_size(_DATASET)
    screen = _DATASET["screen"]
    resolution = (int(screen["width"]), int(screen["height"]))
    target_fraction = geometry.target_pixels[indices] / np.asarray([
        max(resolution[0] - 1, 1), max(resolution[1] - 1, 1),
    ], dtype=np.float32)
    target_points = torch.from_numpy(target_fraction).to(normalized)
    screen_scale = torch.tensor([0.1, 0.1], dtype=normalized.dtype)
    trainable = [parameter for parameter in adapter.parameters() if parameter.requires_grad]
    optimizer = torch.optim.Adam(trainable, lr=0.01)
    for _ in range(700):
        normalized = adapter(inputs, side_values)
        angles = normalized * target_scale + target_means
        points = torch_screen_points(
            torch, angles, geometry, indices, screen_size,
        )
        error = ((points - target_points) / screen_scale).square().mean()
        if multilayer:
            regularization = 0.0
            for gain, adapter_bias in zip(adapter.gains, adapter.biases):
                regularization = regularization + 0.03 * (
                    (gain - 1.0).square().mean() + adapter_bias.square().mean()
                )
        else:
            regularization = 0.03 * (
                (adapter.gain - 1.0).square().mean() + adapter.bias.square().mean()
            )
        loss = error + regularization
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    return adapter


def main() -> None:
    global _DATASET, _DATA, _GEOMETRY
    dataset_path = Path("data/shared-eye-angle-calibration.json")
    _DATASET = json.loads(dataset_path.read_text(encoding="utf-8"))
    raw_data = prepare_shared_dataset(_DATASET, CNN_MODEL_WIDTH, CNN_MODEL_HEIGHT)
    _GEOMETRY = dataset_geometry(_DATASET)
    validation_labels = {f"grid-{value:02d}" for value in (0, 5, 7, 9, 14)}
    validation_indices = np.flatnonzero(np.asarray([
        group in validation_labels for group in raw_data.groups
    ], dtype=bool))
    train_indices = np.flatnonzero(np.asarray([
        group not in validation_labels for group in raw_data.groups
    ], dtype=bool))
    adaptation_groups = {"grid-01", "grid-03", "grid-04", "grid-10", "grid-13"}
    adaptation_indices = []
    for group in sorted(adaptation_groups):
        group_indices = [
            index for index in train_indices.tolist()
            if raw_data.groups[index] == group
        ]
        # Prepared data stores right/left consecutively: 6 frames = 12 eye samples.
        adaptation_indices.extend(group_indices[:12])
    adaptation_indices = np.asarray(adaptation_indices, dtype=np.int64)

    corner_targets = canonical_corner_targets_from_base([
        raw_data.base_images[index] for index in train_indices
    ])
    base_data = _with_geometry(raw_data, corner_targets)
    _DATA = base_data
    shifted_gray = np.asarray([
        synthetic_light_shift(base_data.gray[index], base_data.alpha[index])
        for index in range(len(base_data.gray))
    ], dtype=np.float32)
    shifted_data = replace(base_data, gray=shifted_gray)

    result = {"schema": "cnn-light-adaptation-feasibility-v1", "results": []}
    for seed in (6101, 6102, 6103):
        trained = train_candidate(
            _DATASET, base_data, _GEOMETRY, train_indices, validation_indices,
            "dense-full-deep", "screen-mse", "geometry", seed, 160,
            return_runtime=True,
        )
        runtime = trained.pop("runtime")
        matrix_delta, bias = fit_output_adapter(
            runtime, shifted_data, _GEOMETRY, adaptation_indices,
        )
        feature_adapter = fit_feature_adapter(
            runtime, shifted_data, _GEOMETRY, adaptation_indices,
        )
        multilayer_adapter = fit_feature_adapter(
            runtime, shifted_data, _GEOMETRY, adaptation_indices, multilayer=True,
        )
        record = {
            "seed": seed,
            "base_normal": evaluate_domain(runtime, base_data, _GEOMETRY, validation_indices),
            "base_shifted": evaluate_domain(runtime, shifted_data, _GEOMETRY, validation_indices),
            "adapted_shifted": evaluate_domain(
                runtime, shifted_data, _GEOMETRY, validation_indices, matrix_delta, bias,
            ),
            "adapted_normal": evaluate_domain(
                runtime, base_data, _GEOMETRY, validation_indices, matrix_delta, bias,
            ),
            "feature_adapted_shifted": evaluate_domain(
                runtime, shifted_data, _GEOMETRY, validation_indices,
                model_override=feature_adapter,
            ),
            "feature_adapted_normal": evaluate_domain(
                runtime, base_data, _GEOMETRY, validation_indices,
                model_override=feature_adapter,
            ),
            "multilayer_adapted_shifted": evaluate_domain(
                runtime, shifted_data, _GEOMETRY, validation_indices,
                model_override=multilayer_adapter,
            ),
            "multilayer_adapted_normal": evaluate_domain(
                runtime, base_data, _GEOMETRY, validation_indices,
                model_override=multilayer_adapter,
            ),
            "adapter_matrix_delta": matrix_delta.numpy().tolist(),
            "adapter_bias": bias.numpy().tolist(),
            "adaptation_sources": len(adaptation_indices),
        }
        result["results"].append(record)
        print(json.dumps({
            "seed": seed,
            "base_shifted_median": record["base_shifted"]["screen_median_px"],
            "adapted_shifted_median": record["adapted_shifted"]["screen_median_px"],
            "adapted_normal_median": record["adapted_normal"]["screen_median_px"],
            "feature_adapted_shifted_median": record[
                "feature_adapted_shifted"
            ]["screen_median_px"],
            "feature_adapted_normal_median": record[
                "feature_adapted_normal"
            ]["screen_median_px"],
            "multilayer_adapted_shifted_median": record[
                "multilayer_adapted_shifted"
            ]["screen_median_px"],
            "multilayer_adapted_normal_median": record[
                "multilayer_adapted_normal"
            ]["screen_median_px"],
        }, ensure_ascii=False), flush=True)
    Path("data/cnn-light-adaptation-feasibility.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8",
    )


if __name__ == "__main__":
    main()
