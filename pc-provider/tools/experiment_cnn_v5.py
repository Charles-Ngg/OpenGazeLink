from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

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
AFFINE_GROUPS = ("grid-04", "grid-10", "grid-14", "grid-20", "grid-22")
OUTER_POSE_GROUPS = ("pose-04",)
DEFAULT_SEEDS = (7101, 7102, 7103)


def _residual_network_class(torch, nn):
    activation = lambda: nn.LeakyReLU(CNN_ACTIVATION_NEGATIVE_SLOPE)

    class ResidualBlock(nn.Module):
        def __init__(self, inputs: int, outputs: int, groups: int) -> None:
            super().__init__()
            self.body = nn.Sequential(
                nn.Conv2d(inputs, outputs, 3, padding=1, bias=False),
                nn.GroupNorm(groups, outputs), activation(),
                nn.Conv2d(outputs, outputs, 3, padding=1, bias=False),
                nn.GroupNorm(groups, outputs),
            )
            self.skip = (
                nn.Conv2d(inputs, outputs, 1, bias=False)
                if inputs != outputs else nn.Identity()
            )
            self.activation = activation()

        def forward(self, values):
            return self.activation(self.body(values) + self.skip(values))

    class ResidualEyeCnn(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.stem = nn.Sequential(
                nn.Conv2d(CNN_INPUT_CHANNELS, 12, 3, padding=1, bias=False),
                nn.GroupNorm(3, 12), activation(),
            )
            self.features = nn.Sequential(
                ResidualBlock(12, 16, 4), nn.AvgPool2d(2),
                ResidualBlock(16, 24, 6), nn.AvgPool2d(2),
            )
            feature_count = 24 * (CNN_MODEL_HEIGHT // 4) * (CNN_MODEL_WIDTH // 4)
            self.head = nn.Sequential(
                nn.Flatten(), nn.Linear(feature_count, 24),
                activation(), nn.Linear(24, 2),
            )

        def forward(self, values):
            return self.head(self.features(self.stem(values)))

    return ResidualEyeCnn


def _network(architecture: str):
    torch, nn = require_torch()
    if architecture == "tiny":
        return _tiny_cnn_class(torch, nn)()
    if architecture == "residual-6x10":
        return _residual_network_class(torch, nn)()
    raise ValueError(f"unknown architecture: {architecture}")


def _predict(model, images: np.ndarray, normalization: dict) -> np.ndarray:
    torch, _ = require_torch()
    mean = np.asarray(normalization["image_mean"], dtype=np.float32).reshape(1, 2, 1, 1)
    scale = np.asarray(normalization["image_scale"], dtype=np.float32).reshape(1, 2, 1, 1)
    with torch.inference_mode():
        normalized = model(torch.from_numpy((images - mean) / scale)).numpy()
    return (
        normalized * np.asarray(normalization["target_scale"], dtype=np.float32)
        + np.asarray(normalization["target_mean"], dtype=np.float32)
    )


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


def _fit_diagonal_affine(predictions: np.ndarray, targets: np.ndarray) -> dict:
    gains = []
    offsets = []
    for axis in range(2):
        design = np.column_stack([
            predictions[:, axis], np.ones(len(predictions), dtype=np.float64),
        ])
        gain, offset = np.linalg.lstsq(design, targets[:, axis], rcond=None)[0]
        gains.append(float(gain))
        offsets.append(float(offset))
    return {"gains": gains, "offsets": offsets}


def _apply_affine(predictions: np.ndarray, affine: dict | None) -> np.ndarray:
    if affine is None:
        return predictions
    return (
        predictions * np.asarray(affine["gains"], dtype=np.float64)
        + np.asarray(affine["offsets"], dtype=np.float64)
    )


def _images(data, indices: np.ndarray) -> np.ndarray:
    return np.asarray([
        cnn_eye_input(data.gray[index], data.alpha[index])
        for index in indices.tolist()
    ], dtype=np.float32)


def _train(
    data, train_indices: np.ndarray, validation_indices: np.ndarray,
    architecture: str, structured_pairs: bool, seed: int, epochs: int,
) -> tuple[object, dict]:
    torch, nn = require_torch()
    torch.manual_seed(seed)
    torch.set_num_threads(max(1, min(8, int(torch.get_num_threads()))))
    model = _network(architecture)
    train_images, train_targets = _cnn_arrays(data, train_indices, seed + 101)
    validation_images = _images(data, validation_indices)
    validation_targets = data.targets[validation_indices].astype(np.float32)

    image_mean = np.mean(train_images, axis=(0, 2, 3)).astype(np.float32)
    image_scale = np.maximum(
        np.std(train_images, axis=(0, 2, 3)), 1e-3,
    ).astype(np.float32)
    target_mean = np.mean(train_targets, axis=0).astype(np.float32)
    target_scale = np.maximum(
        np.std(train_targets, axis=0), np.radians(1.0),
    ).astype(np.float32)
    normalization = {
        "image_mean": image_mean.tolist(), "image_scale": image_scale.tolist(),
        "target_mean": target_mean.tolist(), "target_scale": target_scale.tolist(),
    }
    views = len(train_images) // len(train_indices)
    train_x = torch.from_numpy(
        (train_images - image_mean.reshape(1, 2, 1, 1))
        / image_scale.reshape(1, 2, 1, 1)
    ).reshape(len(train_indices), views, 2, CNN_MODEL_HEIGHT, CNN_MODEL_WIDTH)
    train_y = torch.from_numpy(
        (train_targets - target_mean) / target_scale
    ).reshape(len(train_indices), views, 2)
    weights = _balanced_source_weights(data, train_indices)
    probability = weights / np.sum(weights)
    source_conditions = np.asarray([
        1 if data.groups[index].startswith("pose-") else 0
        for index in train_indices.tolist()
    ], dtype=np.int64)
    source_sides = np.asarray([
        1 if data.sides[index] == "right" else 0
        for index in train_indices.tolist()
    ], dtype=np.int64)
    group_ids = {
        group: group_id for group_id, group in enumerate(dict.fromkeys(data.groups))
    }
    source_groups = np.asarray([
        group_ids[data.groups[index]] for index in train_indices.tolist()
    ], dtype=np.int64)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_function = nn.SmoothL1Loss(beta=0.35)
    rng = np.random.default_rng(seed)
    best_state = None
    best_score = float("inf")
    best_epoch = epochs
    best_validation = None
    stale = 0
    batch_size = 32
    for epoch in range(1, epochs + 1):
        model.train()
        order = rng.choice(
            len(train_indices), size=len(train_indices), replace=True, p=probability,
        )
        for start in range(0, len(order), batch_size):
            source = order[start:start + batch_size]
            batch = torch.from_numpy(source.astype(np.int64))
            batch_x = train_x[batch]
            batch_y = train_y[batch]
            optimizer.zero_grad(set_to_none=True)
            prediction = model(batch_x.flatten(0, 1)).reshape(len(batch), views, 2)
            supervised = loss_function(prediction, batch_y)
            reference = prediction[:, :1, :].expand_as(prediction[:, 1:, :])
            consistency = loss_function(prediction[:, 1:, :], reference)
            original_prediction = prediction[:, 0, :]
            original_target = batch_y[:, 0, :]
            if structured_pairs:
                conditions = torch.from_numpy(source_conditions[source])
                sides = torch.from_numpy(source_sides[source])
                groups = torch.from_numpy(source_groups[source])
                pair_mask = (
                    (conditions[:, None] == conditions[None, :])
                    & (sides[:, None] == sides[None, :])
                    & (groups[:, None] != groups[None, :])
                    & torch.triu(torch.ones(len(source), len(source), dtype=torch.bool), diagonal=1)
                )
                target_difference = original_target[:, None, :] - original_target[None, :, :]
                pair_mask &= torch.linalg.vector_norm(target_difference, dim=2) >= 0.20
                if bool(torch.any(pair_mask)):
                    prediction_difference = (
                        original_prediction[:, None, :] - original_prediction[None, :, :]
                    )
                    delta = loss_function(
                        prediction_difference[pair_mask], target_difference[pair_mask],
                    )
                else:
                    delta = original_prediction.sum() * 0.0
            else:
                delta = loss_function(
                    original_prediction - original_prediction.mean(dim=0, keepdim=True),
                    original_target - original_target.mean(dim=0, keepdim=True),
                )
            loss = supervised + 0.05 * consistency + 0.15 * delta
            loss.backward()
            optimizer.step()

        model.eval()
        validation_prediction = _predict(model, validation_images, normalization)
        validation = _metrics(validation_targets, validation_prediction)
        worst_gain_deviation = max(
            abs(axis["slope"] - 1.0)
            for axis in validation["gain"]["axes"].values()
        )
        score = (
            validation["median_deg"] + 0.25 * validation["p95_deg"]
            + 5.0 * worst_gain_deviation
        )
        if not validation["gain"]["collapsed"] and score < best_score - 1e-5:
            best_score = float(score)
            best_epoch = epoch
            best_validation = validation
            best_state = {
                name: value.detach().clone() for name, value in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
            if stale >= 22:
                break
    if best_state is None:
        raise RuntimeError(
            f"{architecture} seed {seed} produced no non-collapsed validation model"
        )
    model.load_state_dict(best_state)
    return model.eval(), {
        "normalization": normalization,
        "best_epoch": int(best_epoch),
        "validation": best_validation,
        "validation_score": best_score,
        "epochs_run": epoch,
    }


def run_experiment(
    dataset: dict, architecture: str, structured_pairs: bool,
    seed: int, epochs: int,
) -> list[dict]:
    data = prepare_shared_dataset(dataset, CNN_MODEL_WIDTH, CNN_MODEL_HEIGHT)
    groups = np.asarray(data.groups, dtype=object)
    conditions = np.asarray(data.conditions, dtype=object)
    base = conditions != "lighting_anchor"
    validation_mask = np.asarray([group in AFFINE_GROUPS for group in groups], dtype=bool)
    outer_static_mask = np.asarray([group in OUTER_STATIC_GROUPS for group in groups], dtype=bool)
    outer_pose_mask = np.asarray([group in OUTER_POSE_GROUPS for group in groups], dtype=bool)
    train_indices = np.flatnonzero(
        base & ~validation_mask & ~outer_static_mask & ~outer_pose_mask
    )
    validation_indices = np.flatnonzero(base & validation_mask)
    outer_static_indices = np.flatnonzero(base & outer_static_mask)
    outer_pose_indices = np.flatnonzero(base & outer_pose_mask)
    corner_targets = canonical_corner_targets_from_base([
        data.base_images[index] for index in train_indices.tolist()
    ])
    data = _with_geometry(data, corner_targets)
    started = time.perf_counter()
    model, training = _train(
        data, train_indices, validation_indices,
        architecture, structured_pairs, seed, epochs,
    )
    training_seconds = time.perf_counter() - started
    validation_prediction = _predict(
        model, _images(data, validation_indices), training["normalization"],
    )
    affine = _fit_diagonal_affine(
        validation_prediction, data.targets[validation_indices],
    )
    static_prediction = _predict(
        model, _images(data, outer_static_indices), training["normalization"],
    )
    pose_prediction = _predict(
        model, _images(data, outer_pose_indices), training["normalization"],
    )

    torch, _ = require_torch()
    benchmark = _images(data, outer_static_indices[:2])
    mean = np.asarray(training["normalization"]["image_mean"], dtype=np.float32).reshape(1, 2, 1, 1)
    scale = np.asarray(training["normalization"]["image_scale"], dtype=np.float32).reshape(1, 2, 1, 1)
    tensor = torch.from_numpy((benchmark - mean) / scale)
    with torch.inference_mode():
        for _ in range(20):
            model(tensor)
        infer_started = time.perf_counter()
        for _ in range(300):
            model(tensor)
    inference_ms = (time.perf_counter() - infer_started) * 1000.0 / 300.0
    common = {
        "architecture": architecture,
        "structured_pairs": structured_pairs,
        "seed": int(seed),
        "parameters": int(sum(parameter.numel() for parameter in model.parameters())),
        "best_epoch": training["best_epoch"],
        "epochs_run": training["epochs_run"],
        "training_seconds": float(training_seconds),
        "inference_two_eyes_ms": float(inference_ms),
        "inner_validation": training["validation"],
    }
    return [
        {
            **common, "affine_output": False, "affine": None,
            "outer_static": _metrics(data.targets[outer_static_indices], static_prediction),
            "outer_pose": _metrics(data.targets[outer_pose_indices], pose_prediction),
        },
        {
            **common, "affine_output": True, "affine": affine,
            "outer_static": _metrics(
                data.targets[outer_static_indices], _apply_affine(static_prediction, affine),
            ),
            "outer_pose": _metrics(
                data.targets[outer_pose_indices], _apply_affine(pose_prediction, affine),
            ),
        },
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare v5 CNN gain, affine, and residual variants")
    parser.add_argument(
        "--dataset", type=Path,
        default=Path("data/shared-eye-legacy-angle-calibration.json"),
    )
    parser.add_argument("--output", type=Path, default=Path("data/cnn-v5-factorial.json"))
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    args = parser.parse_args()
    dataset = json.loads(args.dataset.read_text(encoding="utf-8"))
    payload = {
        "schema": "eyetracing-cnn-v5-factorial-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": str(args.dataset.resolve()),
        "outer_static_groups": list(OUTER_STATIC_GROUPS),
        "affine_groups": list(AFFINE_GROUPS),
        "outer_pose_groups": list(OUTER_POSE_GROUPS),
        "epochs": int(args.epochs),
        "seeds": [int(seed) for seed in args.seeds],
        "results": [],
    }
    for architecture in ("tiny", "residual-6x10"):
        for structured_pairs in (False, True):
            for seed in args.seeds:
                print(
                    f"training {architecture} pairs={structured_pairs} seed={seed}",
                    flush=True,
                )
                results = run_experiment(
                    dataset, architecture, structured_pairs, seed, args.epochs,
                )
                payload["results"].extend(results)
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(
                    json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8",
                )
                for result in results:
                    print(json.dumps({
                        "variant": [
                            architecture, structured_pairs, result["affine_output"],
                        ],
                        "seed": seed,
                        "static": result["outer_static"],
                        "pose": result["outer_pose"],
                        "inference_ms": result["inference_two_eyes_ms"],
                    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
