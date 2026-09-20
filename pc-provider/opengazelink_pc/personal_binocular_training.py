"""Personal binocular adaptation starting from an immutable public checkpoint."""
from __future__ import annotations

import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .conditioned_eye_model import CONDITIONED_EYE_SCHEMA
from .normalized_eye import screen_camera_origin
from .paths import RESOURCE_ROOT
from .training_runtime import configure_training_threads, training_batch_size, spatial_epoch_limit, spatial_update_budget
from .video_session import write_json


WEIGHT_FLOOR = .02


def spatial_supervision(rows):
    """Zero means unknown; a rail is an interval, never an exact gaze target."""
    constraints = [row.get("constraint") for row in rows]
    weight = np.asarray([row["weight"] for row in rows], np.float32)
    exact = np.asarray([w > .5 and c is None for w, c in zip(weight, constraints)])
    rail = np.asarray([w > 0 and c is not None for w, c in zip(weight, constraints)])
    return {"weight": torch.from_numpy(weight * (exact | rail)),
            "exact": torch.from_numpy(exact), "rail": torch.from_numpy(rail),
            **{key: torch.as_tensor([c[key] if c else default for c in constraints], dtype=torch.float32)
               for key, default in (("normal", [0., 0.]), ("tangent", [0., 0.]),
                                    ("normal_target", 0.), ("lower", 0.), ("upper", 0.))}}


def spatial_supervision_loss(prediction, target, direction, eye_target, supervision):
    exact = F.smooth_l1_loss(prediction, target, reduction="none", beta=.03).sum(1)
    eye = (1 - (direction * eye_target).sum(1)).reshape(-1, 2).mean(1)
    normal = (prediction * supervision["normal"]).sum(1)
    along = (prediction * supervision["tangent"]).sum(1)
    outside = F.relu(supervision["lower"] - along) + F.relu(along - supervision["upper"])
    rail = F.smooth_l1_loss(normal, supervision["normal_target"], reduction="none", beta=.03)
    rail = rail + .15 * F.smooth_l1_loss(outside, torch.zeros_like(outside), reduction="none", beta=.03)
    loss = torch.where(supervision["exact"], exact + .08 * eye, rail)
    return loss * supervision["weight"]


class PublicEyeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.image = nn.Sequential(
            nn.Conv2d(2, 16, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((3, 6)), nn.Flatten(),
            nn.Linear(64 * 3 * 6, 128), nn.ReLU(),
        )
        self.geometry = nn.Sequential(nn.Linear(70, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU())
        self.regressor = nn.Sequential(nn.Linear(160, 64), nn.ReLU(), nn.Linear(64, 3))
        self.register_buffer("base", torch.tensor([0., 0., 1.]))

    def features(self, image, geometry):
        return torch.cat((self.image(image), self.geometry(geometry)), 1)

    def forward(self, image, geometry):
        return F.normalize(self.base + .2 * self.regressor(self.features(image, geometry)), dim=1)


class PersonalBinocularNet(PublicEyeNet):
    def __init__(self):
        super().__init__()
        # Match the selected public pipeline checkpoint's stride-4 appearance path.
        self.image[4].stride = (1, 1)
        self.uncertainty = nn.Sequential(nn.Linear(160, 32), nn.ReLU(), nn.Linear(32, 1))
        nn.init.zeros_(self.uncertainty[-1].weight)
        nn.init.zeros_(self.uncertainty[-1].bias)

    def forward(self, image, geometry):
        features = self.features(image, geometry)
        direction = F.normalize(self.base + .2 * self.regressor(features), dim=1)
        log_variance = 4 * torch.tanh(self.uncertainty(features).squeeze(1) / 4)
        return direction, log_variance


def fusion_weights(log_variance):
    return WEIGHT_FLOOR + (1 - 2 * WEIGHT_FLOOR) * torch.softmax(
        -log_variance.reshape(-1, 2), dim=1,
    )


class PersonalBinocularInference(nn.Module):
    def __init__(self, model, mean, scale, mask):
        super().__init__()
        self.model = model
        self.register_buffer("mean", torch.as_tensor(mean, dtype=torch.float32))
        self.register_buffer("scale", torch.as_tensor(scale, dtype=torch.float32))
        self.register_buffer("mask", torch.as_tensor(mask, dtype=torch.float32))

    def forward(self, patches, raw_geometry):
        image = patches.to(torch.float32) / 255.
        gray = image[:, 0]
        mean = gray.mean((1, 2), keepdim=True)
        scale = gray.std((1, 2), unbiased=False, keepdim=True).clamp_min(10 / 255.)
        image = torch.stack(((gray - mean) / scale, image[:, 1]), 1)
        geometry = ((raw_geometry - self.mean) / self.scale).clamp(-10., 10.) * self.mask
        direction, log_variance = self.model(image, geometry)
        return direction, fusion_weights(log_variance).reshape(-1)


def _public_files():
    override = os.environ.get("OPENGAZELINK_PUBLIC_MODEL_DIR")
    if override:
        source = Path(override).expanduser().resolve()
        return source / "best.pt", source / "result.json"
    packaged = RESOURCE_ROOT / "models" / "public-conditioned"
    if (packaged / "best.pt").is_file() and (packaged / "result.json").is_file():
        return packaged / "best.pt", packaged / "result.json"
    development = Path(__file__).resolve().parents[1] / "data" / "conditioned-full-pipeline-20260908" / "public" / "pipeline_corners4"
    return development / "best.pt", development / "result.json"


def _geometry_mask():
    keep = np.zeros(70, dtype=np.float32)
    keep[:10] = 1
    keep[62:69] = 1
    keep[[10, 11, 26, 27]] = 1
    return keep


from .spatial_metrics import spatial_metrics as _metrics, eccentricity_degrees, acceptance


def train_personal_binocular(session_path, rows, usable, run_path, cfg, camera_profile,
                             epochs=120, progress=print, cancelled=lambda: False,
                             max_optimizer_steps="auto"):
    """Retrain personal eye direction and binocular fusion from public weights."""
    public_checkpoint, public_result = _public_files()
    if not public_checkpoint.is_file() or not public_result.is_file():
        raise ValueError("公开预训练眼部模型未随程序安装；原始采集已保留")
    public_metadata = json.loads(public_result.read_text(encoding="utf-8"))
    normalization = public_metadata["normalization"]
    public_digest = hashlib.sha256(public_checkpoint.read_bytes()).hexdigest()
    threads = configure_training_threads()
    batch_size = training_batch_size()
    torch.manual_seed(9300)

    images, raw_geometry, rotations, centers = [], [], [], []
    for index, row in enumerate(usable):
        if index % 240 == 0 and cancelled():
            raise RuntimeError("个人双眼训练已取消，原始采集保留")
        with np.load(session_path / row["input"], allow_pickle=False) as stored:
            images.append(stored["images"][:, 1].copy())
            raw_geometry.append(np.concatenate((stored["head"], stored["points"][:, 1], stored["crop"]), 1).astype(np.float32))
            rotations.append(stored["rotation"].copy())
            centers.append(stored["center"].copy())
    images = torch.from_numpy(np.concatenate(images))
    raw_geometry_np = np.concatenate(raw_geometry)
    normalized_geometry = torch.from_numpy(
        np.clip((raw_geometry_np - np.asarray(normalization["mean"], np.float32)) /
                np.asarray(normalization["scale"], np.float32), -10., 10.) * _geometry_mask()
    )
    rotations_np = np.concatenate(rotations).astype(np.float32)
    centers_np = np.concatenate(centers).astype(np.float32)
    wh = np.asarray([cfg["screen_width"], cfg["screen_height"]], np.float32)
    camera_position = [cfg["camera_offset_x_cm"], cfg["camera_offset_y_cm"], cfg["camera_offset_z_cm"]]
    origin = screen_camera_origin(int(wh[0]), int(wh[1]), cfg["screen_diagonal_inches"], camera_position)
    screen_size = cfg["screen_diagonal_inches"] * 2.54 * wh / np.linalg.norm(wh)
    targets = np.asarray([row["target"] for row in usable], np.float32)
    target_cm = np.c_[origin[0] - (targets[:, 0] - .5) * screen_size[0],
                      origin[1] + (targets[:, 1] - .5) * screen_size[1],
                      np.full(len(targets), origin[2])]
    camera_rays = target_cm[:, None] - np.asarray(centers)
    camera_rays /= np.linalg.norm(camera_rays, axis=2, keepdims=True)
    local_targets = np.einsum("neji,nej->nei", np.asarray(rotations), camera_rays)
    local_targets[:, 1, 0] *= -1
    local_targets = torch.from_numpy(local_targets.astype(np.float32).reshape(-1, 3))
    frame_weight = np.asarray([row["weight"] for row in usable], np.float32)
    supervision = spatial_supervision(usable)
    from .head_coverage import pose_balance
    capture = json.loads((Path(session_path) / "session.json").read_text(encoding="utf-8"))
    from .calibration_split import split_masks, spatial_stage, selection_score
    user_spatial = spatial_stage(capture.get("plan"))
    train_mask, selection_mask, test_mask = split_masks(usable, capture.get("plan"))
    train = np.flatnonzero(train_mask & (supervision["weight"].numpy() > 0))
    requested_epochs = epochs
    if user_spatial:
        if max_optimizer_steps == "auto":
            replay_path = Path(session_path) / "replay-report.json"
            replay_report = json.loads(replay_path.read_text(encoding="utf-8")) if replay_path.exists() else capture.get("replay", {})
            max_optimizer_steps = spatial_update_budget(replay_report)
        epochs = spatial_epoch_limit(epochs, len(train), batch_size, max_optimizer_steps)
    balance, head_audit = pose_balance(np.asarray(rotations), usable, train, capture.get("plan"))
    validation, test = np.flatnonzero(selection_mask), np.flatnonzero(test_mask)

    model = PersonalBinocularNet()
    state = torch.load(public_checkpoint, map_location="cpu", weights_only=True)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if set(missing) != {"uncertainty.0.weight", "uncertainty.0.bias", "uncertainty.2.weight", "uncertainty.2.bias"} or unexpected:
        raise ValueError(f"公开预训练 checkpoint 结构不匹配: missing={missing}, unexpected={unexpected}")

    def model_input(frame_ids):
        eye_ids = np.column_stack((2 * frame_ids, 2 * frame_ids + 1)).ravel()
        patch = images[eye_ids].to(torch.float32) / 255.
        gray = patch[:, 0]
        patch[:, 0] = (gray - gray.mean((1, 2), keepdim=True)) / gray.std((1, 2), unbiased=False, keepdim=True).clamp_min(10 / 255.)
        return patch, normalized_geometry[eye_ids], eye_ids

    rotation = torch.from_numpy(rotations_np)
    center = torch.from_numpy(centers_np)
    origin_t = torch.as_tensor(origin, dtype=torch.float32)
    size_t = torch.as_tensor(screen_size, dtype=torch.float32)

    def project(direction, weights, frame_ids):
        eyes = np.column_stack((2 * frame_ids, 2 * frame_ids + 1)).ravel()
        sign = direction.new_tensor([[1., 1., 1.], [-1., 1., 1.]]).repeat(len(frame_ids), 1)
        rays = torch.bmm(rotation[eyes], (direction * sign).unsqueeze(2)).squeeze(2)
        distance = (origin_t[2] - center[eyes, 2]) / rays[:, 2].clamp(max=-1e-5)
        point = center[eyes] + distance[:, None] * rays
        xy = (point[:, :2] - origin_t[:2]) / size_t * direction.new_tensor([-1., 1.]) + .5
        return (xy.reshape(-1, 2, 2) * weights[..., None]).sum(1)

    def evaluate(frame_ids):
        predictions = []
        model.eval()
        with torch.no_grad():
            for chunk in np.array_split(frame_ids, max(1, int(np.ceil(len(frame_ids) / batch_size)))):
                patch, geometry, _ = model_input(chunk)
                direction, log_variance = model(patch, geometry)
                predictions.append(project(direction, fusion_weights(log_variance), chunk).numpy())
        return np.concatenate(predictions)

    eccentricity = eccentricity_degrees(targets, np.asarray(centers), origin, screen_size)
    def metrics_for(prediction, ids):
        return _metrics(prediction, targets[ids], frame_weight[ids], wh,
                        eccentricity[ids], [usable[i] for i in ids])

    baseline_prediction = evaluate(validation)
    baseline = metrics_for(baseline_prediction, validation)
    public_geometry = {name: value.detach().clone() for name, value in model.geometry.named_parameters()}
    best_state = copy.deepcopy(model.state_dict())
    best_metrics = baseline
    best_epoch = 0
    best_score = float("inf")
    history = []
    optimizer_steps = 0
    best_optimizer_step = 0
    optimizer = torch.optim.AdamW([
        {"params": list(model.image.parameters()), "lr": 1e-4},
        {"params": list(model.geometry.parameters()), "lr": 1e-4},
        {"params": list(model.regressor.parameters()), "lr": 4e-4},
        {"params": model.uncertainty.parameters(), "lr": 8e-4},
    ], weight_decay=1e-4)
    rng = np.random.default_rng(9300)
    progress("personal_eye_and_binocular_fusion")
    for epoch in range(1, epochs + 1):
        if cancelled():
            raise RuntimeError("个人双眼训练已取消，原始采集保留")
        model.train()
        order = rng.permutation(train)
        for start in range(0, len(train), batch_size):
            frame_ids = order[start:start + batch_size]
            patch, geometry, eye_ids = model_input(frame_ids)
            direction, log_variance = model(patch, geometry)
            weights = fusion_weights(log_variance)
            fused = project(direction, weights, frame_ids)
            supervised = {key: value[frame_ids] for key, value in supervision.items()}
            loss = (spatial_supervision_loss(fused, torch.from_numpy(targets[frame_ids]),
                    direction, local_targets[eye_ids], supervised) * torch.from_numpy(balance[frame_ids])).mean()
            # Keep the public pose mapping near its starting solution while
            # crossed eye/head observations teach personal corrections.
            loss = loss + .001 * sum((value-public_geometry[name]).square().mean() / public_geometry[name].square().mean().clamp_min(1e-4)
                                    for name, value in model.geometry.named_parameters())
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            optimizer_steps += 1
        candidate = evaluate(validation)
        metrics = metrics_for(candidate, validation)
        history.append({"epoch": epoch, "optimizer_steps": optimizer_steps, **metrics})
        score = selection_score(candidate, [usable[i] for i in validation], wh) if user_spatial else None
        if user_spatial:
            history[-1]["test_selection_loss"] = score if np.isfinite(score) else None
        progress(f"personal_epoch:{epoch}/{epochs}")
        if (score < best_score if user_spatial else acceptance(metrics, best_metrics) and acceptance(metrics, baseline)):
            best_state = copy.deepcopy(model.state_dict())
            best_metrics = metrics
            best_epoch = epoch
            best_optimizer_step = optimizer_steps
            if user_spatial:
                best_score = score
        if epoch >= 12 and epoch - best_epoch >= 8:
            break
    if user_spatial and not best_epoch:
        raise ValueError("训练未产生有限的模型输出；原始数据已保留")
    model.load_state_dict(best_state)
    model.eval()
    independent_test = metrics_for(evaluate(test), test) if len(test) else None
    reference_indices = [66, 67]
    reference_values = np.median(raw_geometry_np[:, reference_indices], axis=0).tolist()
    inference = PersonalBinocularInference(model, normalization["mean"], normalization["scale"], _geometry_mask()).eval()
    example = (images[:2], torch.from_numpy(raw_geometry_np[:2]))
    scripted = torch.jit.trace(inference, example)
    destination = Path(run_path) / "personal-spatial"
    destination.mkdir()
    module = destination / "conditioned-eye-binocular.pt"
    scripted.save(str(module))
    item = {"module_file": module.name, "module_sha256": hashlib.sha256(module.read_bytes()).hexdigest(),
            "normalization": normalization, "raw_inputs": True, "uses_iris_points": False,
            "fusion": "reinitialized_learned_relative_precision", "weight_floor": WEIGHT_FLOOR,
            "runtime_geometry_reference": {"indices": reference_indices, "values": reference_values, "relative_tolerance": .25}}
    metadata = {"schema": CONDITIONED_EYE_SCHEMA, "created_at": datetime.now(timezone.utc).isoformat(),
                "preprocessing": {"landmarker_backend": "tasks", "version": "tasks_conditioned_full_texture_v1", "raw_inputs": True},
                "screen": {"width": int(wh[0]), "height": int(wh[1])}, "screen_diagonal_inches": cfg["screen_diagonal_inches"],
                "camera_position_screen_cm": camera_position, "screen_camera_origin_cm": origin.tolist(),
                "input_source": cfg["input_source"], "camera_profile": camera_profile,
                "training": {"base_source": "public_pretrained", "public_checkpoint_sha256": public_digest,
                             "trainable": ["image", "geometry", "gaze_direction", "binocular_fusion"],
                             "fusion_reinitialized": True, "training_cpu_threads": threads,
                             "best_epoch": best_epoch, "batch_size": batch_size, "validation_baseline": baseline,
                             "optimizer_steps": optimizer_steps, "selected_optimizer_step": best_optimizer_step,
                             "requested_epochs": requested_epochs, "epoch_limit": epochs,
                             "max_optimizer_steps": max_optimizer_steps,
                             "split_policy": "train/test epoch selection" if user_spatial else "research train/validation/test",
                             "test_selection_loss": best_score if user_spatial else None,
                             "supervision": "settled exact anchors; interval rails; zero-weight frames excluded; equal trial mass",
                             "supervised_train_frames": len(train), "head_pose_coverage": head_audit,
                             "validation": best_metrics, "independent_test": independent_test},
                "variants": {"conditioned_binocular": item}}
    metadata_path = destination / "conditioned-binocular-model.json"
    write_json(metadata_path, metadata)
    write_json(destination / "training-report.json", {"base_source": "public_pretrained", "public_checkpoint_sha256": public_digest,
               "baseline": baseline, "candidate": best_metrics, "best_epoch": best_epoch,
               "independent_test": independent_test, "history": history, "head_pose_coverage": head_audit, "training_cpu_threads": threads,
               "training_batch_size": batch_size, "optimizer_steps": optimizer_steps,
               "selected_optimizer_step": best_optimizer_step,
               "requested_epochs": requested_epochs, "epoch_limit": epochs,
               "max_optimizer_steps": max_optimizer_steps})
    return metadata_path
