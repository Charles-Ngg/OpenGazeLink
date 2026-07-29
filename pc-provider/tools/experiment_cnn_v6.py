from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import types

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
# The isolated CUDA environment supplies Torch, while preprocessing still uses
# the production environment's MediaPipe package and its Python dependencies.
production_site_packages = PROJECT_ROOT / ".venv" / "Lib" / "site-packages"
if production_site_packages.exists():
    sys.path.append(str(production_site_packages))
try:
    import mediapipe  # noqa: F401
except Exception:
    # Dataset-only experiments never instantiate a landmarker. A stub avoids
    # loading the production cp38 MediaPipe binary in the CUDA cp39 process.
    sys.modules["mediapipe"] = types.ModuleType("mediapipe")

from opengazelink_pc.shared_eye_appearance import (
    CNN_INPUT_CHANNELS,
    CNN_MODEL_HEIGHT,
    CNN_MODEL_WIDTH,
    canonical_corner_targets_from_base,
    cnn_eye_input,
)
from opengazelink_pc.shared_eye_models import (
    CNN_ACTIVATION_NEGATIVE_SLOPE,
    _balanced_source_weights,
    _cnn_arrays,
    _cnn_gain_summary,
    _tiny_cnn_class,
    _with_geometry,
    prepare_shared_dataset,
    require_torch,
)


OUTER_STATIC_GROUPS = ("grid-00", "grid-06", "grid-12", "grid-18", "grid-24")
INNER_VALIDATION_GROUPS = ("grid-04", "grid-10", "grid-14", "grid-20", "grid-22")
OUTER_POSE_GROUPS = ("pose-04",)
DEFAULT_SEEDS = (8201, 8202, 8203)
TRAINER_CURRENT = "fixed-lr"
TRAINER_IMPROVED = "warmup-cosine-ema"


def _activation(nn):
    return nn.LeakyReLU(CNN_ACTIVATION_NEGATIVE_SLOPE)


def _residual_block_class(nn):
    class ResidualBlock(nn.Module):
        def __init__(self, inputs: int, outputs: int) -> None:
            super().__init__()
            groups = max(1, math.gcd(outputs, 8))
            self.body = nn.Sequential(
                nn.Conv2d(inputs, outputs, 3, padding=1, bias=False),
                nn.GroupNorm(groups, outputs),
                _activation(nn),
                nn.Conv2d(outputs, outputs, 3, padding=1, bias=False),
                nn.GroupNorm(groups, outputs),
            )
            self.skip = (
                nn.Conv2d(inputs, outputs, 1, bias=False)
                if inputs != outputs else nn.Identity()
            )
            self.output_activation = _activation(nn)

        def forward(self, values):
            return self.output_activation(self.body(values) + self.skip(values))

    return ResidualBlock


def _residual_medium_class(nn):
    ResidualBlock = _residual_block_class(nn)

    class ResidualMedium(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.stem = nn.Sequential(
                nn.Conv2d(CNN_INPUT_CHANNELS, 16, 3, padding=1, bias=False),
                nn.GroupNorm(8, 16),
                _activation(nn),
            )
            self.features = nn.Sequential(
                ResidualBlock(16, 16),
                ResidualBlock(16, 16),
                nn.AvgPool2d(2),
                ResidualBlock(16, 24),
                ResidualBlock(24, 24),
            )
            feature_count = 24 * (CNN_MODEL_HEIGHT // 2) * (CNN_MODEL_WIDTH // 2)
            self.head = nn.Sequential(
                nn.Flatten(),
                nn.Linear(feature_count, 16),
                _activation(nn),
                nn.Linear(16, 2),
            )

        def forward(self, values):
            return self.head(self.features(self.stem(values)))

    return ResidualMedium


def _residual_large_class(nn):
    ResidualBlock = _residual_block_class(nn)

    class ResidualLarge(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.stem = nn.Sequential(
                nn.Conv2d(CNN_INPUT_CHANNELS, 24, 3, padding=1, bias=False),
                nn.GroupNorm(8, 24),
                _activation(nn),
            )
            self.features = nn.Sequential(
                ResidualBlock(24, 24),
                ResidualBlock(24, 24),
                ResidualBlock(24, 24),
                nn.AvgPool2d(2),
                ResidualBlock(24, 40),
                ResidualBlock(40, 40),
                ResidualBlock(40, 40),
            )
            feature_count = 40 * (CNN_MODEL_HEIGHT // 2) * (CNN_MODEL_WIDTH // 2)
            self.head = nn.Sequential(
                nn.Flatten(),
                nn.Linear(feature_count, 32),
                _activation(nn),
                nn.Linear(32, 2),
            )

        def forward(self, values):
            return self.head(self.features(self.stem(values)))

    return ResidualLarge


def _dsnt_class(torch, nn):
    ResidualBlock = _residual_block_class(nn)

    class SpatialDsnt(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.stem = nn.Sequential(
                nn.Conv2d(CNN_INPUT_CHANNELS, 24, 3, padding=1, bias=False),
                nn.GroupNorm(8, 24),
                _activation(nn),
                ResidualBlock(24, 24),
                ResidualBlock(24, 24),
                nn.AvgPool2d(2),
                ResidualBlock(24, 32),
                ResidualBlock(32, 32),
            )
            self.heatmaps = nn.Conv2d(32, 8, 1)
            self.head = nn.Sequential(
                nn.Linear(8 * 2 + 32, 32),
                _activation(nn),
                nn.Linear(32, 2),
            )
            height = CNN_MODEL_HEIGHT // 2
            width = CNN_MODEL_WIDTH // 2
            self.register_buffer(
                "x_coordinates", torch.linspace(-1.0, 1.0, width).reshape(1, 1, 1, width),
            )
            self.register_buffer(
                "y_coordinates", torch.linspace(-1.0, 1.0, height).reshape(1, 1, height, 1),
            )

        def forward(self, values):
            features = self.stem(values)
            heatmaps = self.heatmaps(features)
            logits = heatmaps.flatten(2)
            probability = logits.softmax(dim=2).reshape_as(heatmaps)
            x_position = (probability * self.x_coordinates).sum(dim=(2, 3))
            y_position = (probability * self.y_coordinates).sum(dim=(2, 3))
            spatial = torch.stack((x_position, y_position), dim=2).flatten(1)
            appearance = features.mean(dim=(2, 3))
            return self.head(torch.cat((spatial, appearance), dim=1))

    return SpatialDsnt


def _network(architecture: str):
    torch, nn = require_torch()
    if architecture == "tiny":
        return _tiny_cnn_class(torch, nn)()
    if architecture == "residual-12x20-m":
        return _residual_medium_class(nn)()
    if architecture == "residual-12x20-l":
        return _residual_large_class(nn)()
    if architecture == "spatial-dsnt":
        return _dsnt_class(torch, nn)()
    raise ValueError(f"unknown architecture: {architecture}")


def _initialize_model(model, nn) -> None:
    linear_layers = []
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(
                module.weight, a=CNN_ACTIVATION_NEGATIVE_SLOPE,
                mode="fan_in", nonlinearity="leaky_relu",
            )
            if module.bias is not None:
                nn.init.zeros_(module.bias)
            if isinstance(module, nn.Linear):
                linear_layers.append(module)
        elif isinstance(module, nn.GroupNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
    if linear_layers:
        nn.init.normal_(linear_layers[-1].weight, mean=0.0, std=0.01)


def _images(data, indices: np.ndarray) -> np.ndarray:
    return np.asarray([
        cnn_eye_input(data.gray[index], data.alpha[index])
        for index in indices.tolist()
    ], dtype=np.float32)


def _metrics(targets: np.ndarray, predictions: np.ndarray) -> dict:
    errors = np.degrees(np.linalg.norm(predictions - targets, axis=1))
    design = np.column_stack([targets, np.ones(len(targets), dtype=np.float64)])
    mapping = np.linalg.lstsq(design, predictions, rcond=None)[0]
    return {
        "samples": int(len(errors)),
        "median_deg": float(np.median(errors)),
        "p95_deg": float(np.percentile(errors, 95.0)),
        "max_deg": float(np.max(errors)),
        "gain": _cnn_gain_summary(targets, predictions),
        "cross_axis": {
            "yaw_from_yaw": float(mapping[0, 0]),
            "yaw_from_pitch": float(mapping[1, 0]),
            "pitch_from_yaw": float(mapping[0, 1]),
            "pitch_from_pitch": float(mapping[1, 1]),
            "yaw_offset_deg": float(np.degrees(mapping[2, 0])),
            "pitch_offset_deg": float(np.degrees(mapping[2, 1])),
        },
    }


def _predict(model, images: np.ndarray, normalization: dict, device) -> np.ndarray:
    torch, _ = require_torch()
    mean = np.asarray(normalization["image_mean"], dtype=np.float32).reshape(1, 2, 1, 1)
    scale = np.asarray(normalization["image_scale"], dtype=np.float32).reshape(1, 2, 1, 1)
    tensor = torch.from_numpy((images - mean) / scale).to(device)
    with torch.inference_mode():
        normalized = model(tensor).detach().cpu().numpy()
    return (
        normalized * np.asarray(normalization["target_scale"], dtype=np.float32)
        + np.asarray(normalization["target_mean"], dtype=np.float32)
    )


def _ema_update(ema_state: dict, model, decay: float) -> None:
    for name, value in model.state_dict().items():
        if value.is_floating_point():
            ema_state[name].mul_(decay).add_(value.detach(), alpha=1.0 - decay)
        else:
            ema_state[name].copy_(value)


def _axis_pair_indices(data, train_indices: np.ndarray) -> np.ndarray:
    candidates = []
    groups = np.asarray(data.groups, dtype=object)
    sides = np.asarray(data.sides, dtype=object)
    targets = data.targets.astype(np.float64)
    for side in ("left", "right"):
        side_indices = train_indices[
            (sides[train_indices] == side)
            & np.asarray([str(groups[index]).startswith("grid-") for index in train_indices])
        ]
        for first_offset, first in enumerate(side_indices.tolist()):
            for second in side_indices[first_offset + 1:].tolist():
                difference = np.abs(targets[first] - targets[second])
                if (
                    difference[0] >= math.radians(3.0)
                    and difference[1] <= math.radians(0.8)
                ):
                    candidates.append((first, second, 0))
                elif (
                    difference[1] >= math.radians(2.0)
                    and difference[0] <= math.radians(0.8)
                ):
                    candidates.append((first, second, 1))
    if not candidates:
        return np.empty((0, 3), dtype=np.int64)
    source_lookup = {int(index): offset for offset, index in enumerate(train_indices.tolist())}
    return np.asarray([
        (source_lookup[first], source_lookup[second], axis)
        for first, second, axis in candidates
    ], dtype=np.int64)


def _train(
    data, train_indices: np.ndarray, validation_indices: np.ndarray,
    architecture: str, trainer: str, seed: int, epochs: int,
    device, axis_pair_weight: float = 0.0,
) -> tuple[object, dict]:
    torch, nn = require_torch()
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    model = _network(architecture)
    if trainer == TRAINER_IMPROVED:
        _initialize_model(model, nn)
    model = model.to(device)
    train_images, train_targets = _cnn_arrays(data, train_indices, seed + 101)
    validation_images = _images(data, validation_indices)
    validation_targets = data.targets[validation_indices].astype(np.float32)

    image_mean = np.mean(train_images, axis=(0, 2, 3)).astype(np.float32)
    image_scale = np.maximum(np.std(train_images, axis=(0, 2, 3)), 1e-3).astype(np.float32)
    target_mean = np.mean(train_targets, axis=0).astype(np.float32)
    target_scale = np.maximum(np.std(train_targets, axis=0), np.radians(1.0)).astype(np.float32)
    normalization = {
        "image_mean": image_mean.tolist(), "image_scale": image_scale.tolist(),
        "target_mean": target_mean.tolist(), "target_scale": target_scale.tolist(),
    }
    views = len(train_images) // len(train_indices)
    train_x = torch.from_numpy(
        (train_images - image_mean.reshape(1, 2, 1, 1))
        / image_scale.reshape(1, 2, 1, 1)
    ).reshape(len(train_indices), views, 2, CNN_MODEL_HEIGHT, CNN_MODEL_WIDTH).to(device)
    train_y = torch.from_numpy(
        (train_targets - target_mean) / target_scale
    ).reshape(len(train_indices), views, 2).to(device)
    weights = _balanced_source_weights(data, train_indices)
    probability = weights / np.sum(weights)
    pair_indices = _axis_pair_indices(data, train_indices)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_function = nn.SmoothL1Loss(beta=0.35)
    rng = np.random.default_rng(seed)
    best_state = None
    best_score = float("inf")
    best_epoch = epochs
    best_validation = None
    stale = 0
    history = []
    batch_size = 32
    ema_state = None
    if trainer == TRAINER_IMPROVED:
        ema_state = {
            name: value.detach().clone() for name, value in model.state_dict().items()
        }
    ema_updates = 0
    for epoch in range(1, epochs + 1):
        if trainer == TRAINER_IMPROVED:
            warmup = min(1.0, epoch / 5.0)
            progress = max(0.0, (epoch - 5) / max(epochs - 5, 1))
            cosine = 0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * progress))
            learning_rate = 1e-3 * warmup * cosine
            for group in optimizer.param_groups:
                group["lr"] = learning_rate
        else:
            learning_rate = 1e-3
        model.train()
        order = rng.choice(
            len(train_indices), size=len(train_indices), replace=True, p=probability,
        )
        epoch_losses = []
        for start in range(0, len(order), batch_size):
            source = order[start:start + batch_size]
            batch = torch.from_numpy(source.astype(np.int64)).to(device)
            batch_x = train_x[batch]
            batch_y = train_y[batch]
            optimizer.zero_grad(set_to_none=True)
            prediction = model(batch_x.flatten(0, 1)).reshape(len(batch), views, 2)
            supervised = loss_function(prediction, batch_y)
            reference = prediction[:, :1, :].expand_as(prediction[:, 1:, :])
            consistency = loss_function(prediction[:, 1:, :], reference)
            original_prediction = prediction[:, 0, :]
            original_target = batch_y[:, 0, :]
            delta = loss_function(
                original_prediction - original_prediction.mean(dim=0, keepdim=True),
                original_target - original_target.mean(dim=0, keepdim=True),
            )
            pair_loss = original_prediction.sum() * 0.0
            if axis_pair_weight > 0.0 and len(pair_indices):
                selected = pair_indices[rng.integers(0, len(pair_indices), size=len(batch))]
                first = torch.from_numpy(selected[:, 0]).to(device)
                second = torch.from_numpy(selected[:, 1]).to(device)
                axes = torch.from_numpy(selected[:, 2]).to(device)
                paired_prediction = model(torch.cat((train_x[first, 0], train_x[second, 0]), dim=0))
                first_prediction, second_prediction = paired_prediction.chunk(2, dim=0)
                prediction_difference = first_prediction - second_prediction
                target_difference = train_y[first, 0] - train_y[second, 0]
                rows = torch.arange(len(first), device=device)
                pair_loss = loss_function(
                    prediction_difference[rows, axes], target_difference[rows, axes],
                )
            loss = (
                supervised + 0.05 * consistency + 0.15 * delta
                + axis_pair_weight * pair_loss
            )
            loss.backward()
            optimizer.step()
            if ema_state is not None:
                ema_updates += 1
                ema_decay = min(0.995, (1.0 + ema_updates) / (10.0 + ema_updates))
                _ema_update(ema_state, model, ema_decay)
            epoch_losses.append(float(loss.detach().cpu()))

        evaluation_model = model
        raw_state = None
        if ema_state is not None:
            raw_state = copy.deepcopy(model.state_dict())
            model.load_state_dict(ema_state)
        model.eval()
        validation_prediction = _predict(
            evaluation_model, validation_images, normalization, device,
        )
        validation = _metrics(validation_targets, validation_prediction)
        worst_gain_deviation = max(
            abs(axis["slope"] - 1.0) for axis in validation["gain"]["axes"].values()
        )
        score = validation["median_deg"] + 0.25 * validation["p95_deg"] + 5.0 * worst_gain_deviation
        history.append({
            "epoch": int(epoch), "learning_rate": float(learning_rate),
            "loss": float(np.mean(epoch_losses)), "validation_score": float(score),
            "validation_median_deg": validation["median_deg"],
            "validation_p95_deg": validation["p95_deg"],
        })
        if not validation["gain"]["collapsed"] and score < best_score - 1e-5:
            best_score = float(score)
            best_epoch = epoch
            best_validation = validation
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in evaluation_model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
        if raw_state is not None:
            model.load_state_dict(raw_state)
        if stale >= 30 and epoch >= 35:
            break
    if best_state is None:
        raise RuntimeError(f"{architecture} seed {seed} produced no healthy validation model")
    model.load_state_dict(best_state)
    model.eval()
    return model, {
        "normalization": normalization,
        "best_epoch": int(best_epoch),
        "validation": best_validation,
        "validation_score": float(best_score),
        "epochs_run": int(epoch),
        "history": history,
        "axis_pair_candidates": int(len(pair_indices)),
    }


def _split_data(dataset: dict):
    data = prepare_shared_dataset(dataset, CNN_MODEL_WIDTH, CNN_MODEL_HEIGHT)
    groups = np.asarray(data.groups, dtype=object)
    conditions = np.asarray(data.conditions, dtype=object)
    base = conditions != "lighting_anchor"
    inner_mask = np.asarray([group in INNER_VALIDATION_GROUPS for group in groups], dtype=bool)
    static_mask = np.asarray([group in OUTER_STATIC_GROUPS for group in groups], dtype=bool)
    pose_mask = np.asarray([group in OUTER_POSE_GROUPS for group in groups], dtype=bool)
    train_indices = np.flatnonzero(base & ~inner_mask & ~static_mask & ~pose_mask)
    inner_indices = np.flatnonzero(base & inner_mask)
    static_indices = np.flatnonzero(base & static_mask)
    pose_indices = np.flatnonzero(base & pose_mask)
    corner_targets = canonical_corner_targets_from_base([
        data.base_images[index] for index in train_indices.tolist()
    ])
    return (
        _with_geometry(data, corner_targets), train_indices, inner_indices,
        static_indices, pose_indices,
    )


def _training_gap(training: dict, holdout: dict) -> dict:
    return {
        "median_deg": float(holdout["median_deg"] - training["median_deg"]),
        "p95_deg": float(holdout["p95_deg"] - training["p95_deg"]),
    }


def _benchmark_subprocess(architecture: str, ensemble_size: int = 1) -> dict:
    root = Path(__file__).resolve().parents[1]
    production_python = root / ".venv" / "Scripts" / "python.exe"
    command = [
        str(production_python), str(Path(__file__).resolve()),
        "--benchmark-only", "--architectures", architecture,
        "--ensemble-size", str(ensemble_size),
    ]
    completed = subprocess.run(
        command, cwd=str(root), check=True, capture_output=True, text=True,
    )
    return json.loads(completed.stdout.strip().splitlines()[-1])


def _windows_affinity(core_ids: list[int]):
    if os.name != "nt":
        return None
    import ctypes
    kernel32 = ctypes.windll.kernel32
    process = kernel32.GetCurrentProcess()
    process_mask = ctypes.c_size_t()
    system_mask = ctypes.c_size_t()
    if not kernel32.GetProcessAffinityMask(
        process, ctypes.byref(process_mask), ctypes.byref(system_mask),
    ):
        return None
    requested = sum(1 << core_id for core_id in core_ids)
    allowed = requested & int(system_mask.value)
    if allowed == 0:
        allowed = int(process_mask.value)
    if not kernel32.SetProcessAffinityMask(process, ctypes.c_size_t(allowed)):
        return None
    return int(process_mask.value)


def _restore_windows_affinity(mask) -> None:
    if os.name == "nt" and mask is not None:
        import ctypes
        ctypes.windll.kernel32.SetProcessAffinityMask(
            ctypes.windll.kernel32.GetCurrentProcess(), ctypes.c_size_t(mask),
        )


def _local_benchmark(architecture: str, ensemble_size: int) -> dict:
    torch, _ = require_torch()
    models = [_network(architecture).eval() for _ in range(ensemble_size)]
    tensor = torch.randn(2, CNN_INPUT_CHANNELS, CNN_MODEL_HEIGHT, CNN_MODEL_WIDTH)
    results = {}
    for threads, core_ids in ((1, [0]), (2, [0, 2])):
        original_mask = _windows_affinity(core_ids)
        torch.set_num_threads(threads)
        durations = []
        with torch.inference_mode():
            for _ in range(100):
                for model in models:
                    model(tensor)
            for _ in range(1000):
                started = time.perf_counter_ns()
                for model in models:
                    model(tensor)
                durations.append((time.perf_counter_ns() - started) / 1e6)
        _restore_windows_affinity(original_mask)
        results[str(threads)] = {
            "p50_ms": float(np.percentile(durations, 50.0)),
            "p95_ms": float(np.percentile(durations, 95.0)),
            "mean_ms": float(np.mean(durations)),
            "logical_cores": core_ids,
        }
    return {
        "architecture": architecture,
        "ensemble_size": int(ensemble_size),
        "torch": str(torch.__version__),
        "parameters_each": int(sum(parameter.numel() for parameter in models[0].parameters())),
        "two_eye_batch": True,
        "threads": results,
    }


def run_candidate(
    data, train_indices, inner_indices, static_indices, pose_indices,
    architecture: str, trainer: str, seed: int, epochs: int, device,
    axis_pair_weight: float = 0.0,
) -> tuple[dict, dict]:
    started = time.perf_counter()
    model, fit = _train(
        data, train_indices, inner_indices, architecture, trainer,
        seed, epochs, device, axis_pair_weight,
    )
    training_seconds = time.perf_counter() - started
    predictions = {
        "train": _predict(model, _images(data, train_indices), fit["normalization"], device),
        "inner": _predict(model, _images(data, inner_indices), fit["normalization"], device),
        "outer_static": _predict(model, _images(data, static_indices), fit["normalization"], device),
        "outer_pose": _predict(model, _images(data, pose_indices), fit["normalization"], device),
    }
    metrics = {
        "train": _metrics(data.targets[train_indices], predictions["train"]),
        "inner_validation": _metrics(data.targets[inner_indices], predictions["inner"]),
        "outer_static": _metrics(data.targets[static_indices], predictions["outer_static"]),
        "outer_pose": _metrics(data.targets[pose_indices], predictions["outer_pose"]),
    }
    result = {
        "architecture": architecture,
        "trainer": trainer,
        "axis_pair_weight": float(axis_pair_weight),
        "seed": int(seed),
        "parameters": int(sum(parameter.numel() for parameter in model.parameters())),
        "training_device": str(device),
        "training_seconds": float(training_seconds),
        "best_epoch": fit["best_epoch"],
        "epochs_run": fit["epochs_run"],
        "inner_selection_score": fit["validation_score"],
        "axis_pair_candidates": fit["axis_pair_candidates"],
        **metrics,
        "static_training_gap": _training_gap(metrics["train"], metrics["outer_static"]),
        "pose_training_gap": _training_gap(metrics["train"], metrics["outer_pose"]),
    }
    return result, predictions


def _aggregate(results: list[dict]) -> dict:
    fields = (
        ("inner_validation", "median_deg"), ("inner_validation", "p95_deg"),
        ("outer_static", "median_deg"), ("outer_static", "p95_deg"),
        ("outer_pose", "median_deg"), ("outer_pose", "p95_deg"),
    )
    summary = {"seeds": [result["seed"] for result in results]}
    for section, field in fields:
        values = np.asarray([result[section][field] for result in results], dtype=np.float64)
        summary[f"{section}_{field}"] = {
            "median": float(np.median(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "std": float(np.std(values)),
        }
    return summary


def _ensemble_result(
    candidates: list[tuple[dict, dict]], data, static_indices, pose_indices,
) -> dict:
    selected = sorted(candidates, key=lambda item: item[0]["inner_selection_score"])[:2]
    static_prediction = np.mean([
        predictions["outer_static"] for _, predictions in selected
    ], axis=0)
    pose_prediction = np.mean([
        predictions["outer_pose"] for _, predictions in selected
    ], axis=0)
    return {
        "selected_seeds": [result["seed"] for result, _ in selected],
        "outer_static": _metrics(data.targets[static_indices], static_prediction),
        "outer_pose": _metrics(data.targets[pose_indices], pose_prediction),
    }


def _write_payload(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="CNN v6 trainer and capacity comparison")
    parser.add_argument(
        "--dataset", type=Path,
        default=Path("data/shared-eye-legacy-angle-calibration.json"),
    )
    parser.add_argument("--output", type=Path, default=Path("data/cnn-v6-capacity.json"))
    parser.add_argument("--epochs", type=int, default=160)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument(
        "--architectures", nargs="+",
        default=["tiny", "residual-12x20-m", "residual-12x20-l", "spatial-dsnt"],
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--benchmark-only", action="store_true")
    parser.add_argument("--ensemble-size", type=int, default=1)
    args = parser.parse_args()
    if args.benchmark_only:
        print(json.dumps(_local_benchmark(args.architectures[0], args.ensemble_size)))
        return

    torch, _ = require_torch()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device_name = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device_name == "auto":
        device_name = "cpu"
    device = torch.device(device_name)
    dataset_bytes = args.dataset.read_bytes()
    dataset = json.loads(dataset_bytes.decode("utf-8"))
    data, train_indices, inner_indices, static_indices, pose_indices = _split_data(dataset)
    epochs = min(args.epochs, 20) if args.smoke else args.epochs
    seeds = args.seeds[:1] if args.smoke else args.seeds
    payload = {
        "schema": "eyetracing-cnn-v6-capacity-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": str(args.dataset.resolve()),
        "dataset_sha256": hashlib.sha256(dataset_bytes).hexdigest(),
        "split": {
            "train_samples": int(len(train_indices)),
            "inner_validation_samples": int(len(inner_indices)),
            "outer_static_samples": int(len(static_indices)),
            "outer_pose_samples": int(len(pose_indices)),
            "inner_validation_groups": list(INNER_VALIDATION_GROUPS),
            "outer_static_groups": list(OUTER_STATIC_GROUPS),
            "outer_pose_groups": list(OUTER_POSE_GROUPS),
        },
        "training": {
            "device": str(device), "torch": str(torch.__version__),
            "epochs": int(epochs), "seeds": [int(seed) for seed in seeds],
        },
        "results": [], "aggregates": {}, "ensembles": {}, "latency": {},
    }

    experiment_sets = []
    if "tiny" in args.architectures:
        experiment_sets.extend([
            ("tiny", TRAINER_CURRENT, 0.0),
            ("tiny", TRAINER_IMPROVED, 0.0),
        ])
    experiment_sets.extend([
        (architecture, TRAINER_IMPROVED, 0.0)
        for architecture in args.architectures if architecture != "tiny"
    ])
    candidates_by_key = {}
    for architecture, trainer, pair_weight in experiment_sets:
        key = f"{architecture}|{trainer}|pair={pair_weight:.2f}"
        candidates_by_key[key] = []
        print(f"\n[{key}]", flush=True)
        for seed in seeds:
            print(f"training seed={seed} on {device}", flush=True)
            try:
                result, predictions = run_candidate(
                    data, train_indices, inner_indices, static_indices, pose_indices,
                    architecture, trainer, seed, epochs, device, pair_weight,
                )
            except Exception as error:
                failure = {
                    "architecture": architecture, "trainer": trainer,
                    "axis_pair_weight": float(pair_weight), "seed": int(seed),
                    "error": f"{type(error).__name__}: {error}",
                }
                payload.setdefault("failures", []).append(failure)
                _write_payload(args.output, payload)
                print(json.dumps(failure), flush=True)
                continue
            payload["results"].append(result)
            candidates_by_key[key].append((result, predictions))
            _write_payload(args.output, payload)
            print(json.dumps({
                "seed": seed, "parameters": result["parameters"],
                "best_epoch": result["best_epoch"],
                "static": result["outer_static"], "pose": result["outer_pose"],
            }, ensure_ascii=False), flush=True)

    if not args.smoke:
        capacity_keys = [
            key for key in candidates_by_key
            if f"|{TRAINER_IMPROVED}|" in key and candidates_by_key[key]
        ]
        if not capacity_keys:
            raise RuntimeError("all improved-trainer capacity candidates failed")
        winning_key = min(
            capacity_keys,
            key=lambda key: np.median([
                item[0]["inner_selection_score"] for item in candidates_by_key[key]
            ]),
        )
        winning_architecture = winning_key.split("|", 1)[0]
        pair_key = f"{winning_architecture}|{TRAINER_IMPROVED}|pair=0.10"
        candidates_by_key[pair_key] = []
        print(f"\n[{pair_key}]", flush=True)
        for seed in seeds:
            try:
                result, predictions = run_candidate(
                    data, train_indices, inner_indices, static_indices, pose_indices,
                    winning_architecture, TRAINER_IMPROVED, seed, epochs, device, 0.10,
                )
            except Exception as error:
                failure = {
                    "architecture": winning_architecture,
                    "trainer": TRAINER_IMPROVED, "axis_pair_weight": 0.10,
                    "seed": int(seed), "error": f"{type(error).__name__}: {error}",
                }
                payload.setdefault("failures", []).append(failure)
                _write_payload(args.output, payload)
                print(json.dumps(failure), flush=True)
                continue
            payload["results"].append(result)
            candidates_by_key[pair_key].append((result, predictions))
            _write_payload(args.output, payload)

    for key, candidates in candidates_by_key.items():
        if not candidates:
            continue
        payload["aggregates"][key] = _aggregate([item[0] for item in candidates])
        if len(candidates) >= 2:
            payload["ensembles"][key] = _ensemble_result(
                candidates, data, static_indices, pose_indices,
            )
    for architecture in dict.fromkeys(item[0] for item in experiment_sets):
        payload["latency"][architecture] = _benchmark_subprocess(architecture)
        if len(seeds) >= 2:
            payload["latency"][f"{architecture}-ensemble-2"] = _benchmark_subprocess(
                architecture, ensemble_size=2,
            )
    _write_payload(args.output, payload)
    print(f"\nwrote {args.output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
