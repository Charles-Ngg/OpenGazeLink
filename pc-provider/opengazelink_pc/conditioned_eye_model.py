from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Sequence

import numpy as np

from .shared_eye_models import SharedEyePrediction, require_torch


CONDITIONED_EYE_SCHEMA = "opengazelink-conditioned-eye-v1"
CONDITIONED_VARIANTS = ("conditioned_with_iris", "conditioned_without_iris", "conditioned_binocular", "conditioned_video")
IRIS_GEOMETRY_SLICE = slice(42, 52)


class EmbeddedJointForecast:
    """Runtime for motion parameters emitted by the VIDEO GRU itself."""

    def __init__(self, metadata: dict) -> None:
        self.metadata = metadata
        self.motion = None

    def reset(self) -> None:
        self.motion = None

    def set_motion(self, motion) -> None:
        value = np.asarray(motion, dtype=np.float64).reshape(-1)
        self.motion = value.copy() if value.shape in ((4,), (11,)) and np.isfinite(value).all() else None

    def update(self, raw, stable, timestamp, feature, hidden, screen_size, horizon_ms,
               max_lead_fraction=.12, reset=False):
        pixels = np.maximum(1., np.asarray(screen_size, dtype=np.float64) - 1.)
        requested_horizon = float(horizon_ms)
        horizon = float(np.clip(requested_horizon, 0., self.metadata.get("max_horizon_ms", 100.)))
        if reset or self.motion is None or horizon <= 0 or max_lead_fraction <= 0:
            return tuple(stable), {"mode": "joint_prediction_warmup" if self.motion is None else "disabled",
                "horizon_ms": horizon, "requested_horizon_ms": requested_horizon,
                "lead_px": [0., 0.], "lead_distance_px": 0., "shared_temporal_state": True}
        seconds = horizon / 1000.
        pursuit = self.motion[:2] * seconds + .5 * self.motion[2:4] * seconds * seconds
        phase = None
        if len(self.motion) == 11:
            logits = self.motion[4:7] - np.max(self.motion[4:7])
            phase = np.exp(logits); phase /= phase.sum()
            remaining = max(.008, self.motion[9])
            landing = self.motion[7:9] * (1. - np.exp(-3. * seconds / remaining))
            delta = phase[1] * pursuit + phase[2] * landing
            # Match the displacement used by joint training and evaluation.
            reliability = float(np.clip(.06 / max(.003, self.motion[10]), .25, 1.))
            delta *= reliability
            # Fixation uses the causal stable point; moving candidates start
            # at the unfiltered current estimate.
            point = phase[0] * np.asarray(stable, dtype=np.float64) + (1.-phase[0]) * np.asarray(raw, dtype=np.float64) + delta * pixels
            lead = point - np.asarray(stable, dtype=np.float64)
        else:
            delta = pursuit
            lead = delta * pixels
        limit = min(.3, max(0., float(max_lead_fraction))) * float(np.linalg.norm(pixels))
        lead *= min(1., limit / max(1e-9, float(np.linalg.norm(lead))))
        point = (np.asarray(stable, dtype=np.float64) + lead) if phase is not None else (np.asarray(raw, dtype=np.float64) + lead)
        return tuple(point), {"mode": "joint_temporal_prediction", "horizon_ms": horizon,
            "requested_horizon_ms": requested_horizon, "target_source_ms": float(timestamp) + horizon,
            "lead_px": lead.tolist(), "lead_distance_px": float(np.linalg.norm(lead)),
            "motion_normalized": self.motion.tolist(), "phase_probability":phase.tolist() if phase is not None else None,
            "prediction_uncertainty_normalized":float(self.motion[10]) if len(self.motion)==11 else None,
            "shared_temporal_state": True,
            "label_source": "joint_offline_future_gaze_proxy"}


def runtime_geometry_reference(item: dict) -> tuple[np.ndarray, np.ndarray, float]:
    """Validate optional raw-geometry values fixed by personal calibration."""
    reference = item.get("runtime_geometry_reference") or {}
    indices = np.asarray(reference.get("indices", ()), dtype=np.int64)
    values = np.asarray(reference.get("values", ()), dtype=np.float32)
    tolerance = float(reference.get("relative_tolerance", 0.25))
    if indices.shape != values.shape or indices.ndim != 1:
        raise ValueError("conditioned-eye runtime geometry reference is invalid")
    if len(indices) and (
        np.any(indices < 0) or np.any(indices >= 70)
        or len(np.unique(indices)) != len(indices)
        or not np.isfinite(values).all() or not np.isfinite(tolerance)
        or not 0.0 <= tolerance <= 1.0
    ):
        raise ValueError("conditioned-eye runtime geometry reference is invalid")
    return indices, values, tolerance


def stabilize_runtime_geometry(geometry: np.ndarray, item: dict) -> np.ndarray:
    """Retain personal-fit route constants while live K drives PnP and warping."""
    indices, values, tolerance = runtime_geometry_reference(item)
    if not len(indices):
        return geometry
    result = np.asarray(geometry, dtype=np.float32).copy()
    current = result[..., indices]
    limit = tolerance * np.maximum(np.abs(values), 1e-6)
    result[..., indices] = np.where(np.abs(current - values) > limit, values, current)
    return result


class ConditionedEyeModel:
    """Runtime wrapper for the full-texture, head-conditioned gaze network."""

    is_conditioned = True

    def __init__(self, metadata: dict, module, variant: str) -> None:
        if metadata.get("schema") != CONDITIONED_EYE_SCHEMA:
            raise ValueError("unsupported conditioned-eye model")
        if variant not in CONDITIONED_VARIANTS:
            raise ValueError(f"unknown conditioned-eye variant: {variant}")
        variants = metadata.get("variants") or {}
        if variant not in variants:
            raise ValueError(f"conditioned-eye variant is unavailable: {variant}")
        self.metadata = metadata
        self.module = module
        self.variant = variant
        self.binocular = variant in ("conditioned_binocular", "conditioned_video")
        self.temporal = variant == "conditioned_video"
        self.forecast = None
        self.feature_dim = int(variants[variant].get("feature_dim", 148))
        self.variant_metadata = variants[variant]
        self.geometry_reference_indices, self.geometry_reference_values, self.geometry_reference_tolerance = runtime_geometry_reference(
            self.variant_metadata,
        )
        if self.temporal and self.feature_dim not in (148, 404):
            raise ValueError("unsupported VIDEO feature state size")
        self.reset_temporal()
        if self.binocular and not variants[variant].get("raw_inputs"):
            raise ValueError("binocular model must embed input normalization")
        self.landmarker_backend = str(metadata["preprocessing"]["landmarker_backend"])
        normalization = variants[variant]["normalization"]
        self.geometry_mean = np.asarray(normalization["mean"], dtype=np.float32)
        self.geometry_scale = np.asarray(normalization["scale"], dtype=np.float32)
        if self.geometry_mean.shape != (70,) or self.geometry_scale.shape != (70,):
            raise ValueError("conditioned-eye geometry normalization must contain 70 values")

    def set_lighting_profile(self, name: str) -> None:
        if name != "reference":
            raise ValueError("conditioned-eye A/B models currently support reference lighting only")

    @staticmethod
    def _numeric_input(value: dict) -> tuple[np.ndarray, np.ndarray]:
        image = np.asarray(value["images"][1], dtype=np.float32) / 255.0
        gray = image[0]
        mean = float(gray.mean())
        scale = max(float(gray.std()), 10.0 / 255.0)
        image[0] = (gray - mean) / scale
        geometry = np.concatenate((value["head"], value["points"][1], value["crop"]))
        return image, geometry.astype(np.float32)

    def reset_temporal(self) -> None:
        self._hidden = None
        self._previous = None
        self._timestamp_ms = None
        self.reset_probability = 1.0
        self._reset_latched = True
        if getattr(self, "forecast", None) is not None:
            self.forecast.reset()

    def temporal_reset_active(self) -> bool:
        """Return the hysteresis-filtered reset state for the live pipeline."""
        return bool(self._reset_latched)

    def forecast_filter_config(self) -> dict:
        """Return the stable-point preprocessing used when the forecast was trained."""
        if self.forecast is None:
            return {"one_euro_enabled": False}
        raw = self.forecast.metadata.get("filter_config") or {}
        enabled = bool(raw.get("one_euro_enabled", False))
        if not enabled:
            return {"one_euro_enabled": False}
        values = {
            "one_euro_enabled": True,
            "one_euro_min_cutoff": float(raw["one_euro_min_cutoff"]),
            "one_euro_beta": float(raw["one_euro_beta"]),
            "one_euro_derivative_cutoff": float(raw["one_euro_derivative_cutoff"]),
        }
        if not all(math.isfinite(value) for key, value in values.items() if key != "one_euro_enabled"):
            raise ValueError("forecast metadata contains invalid filter parameters")
        return values

    def forecast_point(self, raw, stable, timestamp_ms, config, timing=None):
        if self.forecast is None or self._previous is None or self._hidden is None:
            return None
        horizon=config.extrapolation_horizon_ms
        schema=self.forecast.metadata.get("schema")
        unified=schema=="opengazelink-unified-prediction-v3"
        dynamic=unified or schema in ("opengazelink-motion-prediction-v2", "opengazelink-joint-video-v1")
        timing=timing or {}
        if dynamic and getattr(config,"prediction_auto_horizon_enabled",True) and horizon>0 and timing.get("horizon_ms_proxy") is not None:
            horizon=timing["horizon_ms_proxy"]
        if dynamic and horizon>(500 if unified else 150):
            self.forecast.reset()
            return tuple(stable),{"mode":"prediction_stale_frame","horizon_ms":0.,"requested_horizon_ms":horizon,
                "lead_px":[0.,0.],"lead_distance_px":0.,"timing":timing}
        # Gate resets with hysteresis.  The GRU gate is a confidence signal and
        # can hover around a threshold during fixation; clearing temporal state
        # on every such crossing produces visible flicker and destroys the
        # velocity history needed by the landing predictor.
        if self.reset_probability >= .90:
            self._reset_latched = True
        elif self.reset_probability <= .30:
            self._reset_latched = False
        point,diagnostics=self.forecast.update(raw, stable, timestamp_ms, self._previous.numpy(), self._hidden.numpy(),
                                    (config.screen_width, config.screen_height), horizon,
                                    config.extrapolation_max_lead_fraction, reset=self._reset_latched)
        diagnostics["timing"]=timing
        diagnostics["filter_config"]=self.forecast_filter_config()
        if unified:
            diagnostics["horizon_clamped"]=horizon>250
        return point,diagnostics

    def predict_inputs(self, inputs: Sequence[dict], sides: Sequence[str], timestamp_ms=None,
                       *, spatial_only=False) -> list[SharedEyePrediction]:
        torch, _ = require_torch()
        if len(inputs) != len(sides):
            raise ValueError("eye inputs and sides must have matching lengths")
        if self.binocular:
            if len(inputs) != 2 or list(sides) != ["right", "left"]:
                raise ValueError("binocular inference requires one ordered right/left pair")
            images = np.stack([value["images"][1] for value in inputs]).astype(np.uint8)
            geometry = np.stack([np.concatenate((value["head"], value["points"][1], value["crop"])) for value in inputs])
            geometry = stabilize_runtime_geometry(geometry, self.variant_metadata)
        else:
            unpacked = [self._numeric_input(value) for value in inputs]
            images = np.stack([value[0] for value in unpacked])
            geometry = np.stack([value[1] for value in unpacked])
            geometry = (geometry - self.geometry_mean) / self.geometry_scale
            if self.variant == "conditioned_without_iris":
                geometry[:, IRIS_GEOMETRY_SLICE] = 0.0
        with torch.inference_mode():
            image_tensor = torch.from_numpy(images)
            geometry_tensor = torch.from_numpy(geometry.astype(np.float32))
            if self.temporal and spatial_only and self.spatial_fast_path_available():
                if timestamp_ms is None or not math.isfinite(float(timestamp_ms)):
                    raise ValueError("VIDEO inference requires a finite source timestamp")
                # An untrained zero correction adapter needlessly evaluates the
                # appearance trunk again and runs a GRU. The event path only
                # needs the identical calibrated spatial output.
                output = self.module.base(image_tensor, geometry_tensor)
                self._hidden = self._previous = None
                self._timestamp_ms = float(timestamp_ms)
                self.reset_probability = 0.
            elif self.temporal:
                if timestamp_ms is None or not math.isfinite(float(timestamp_ms)):
                    raise ValueError("VIDEO inference requires a finite source timestamp")
                dt = 0. if self._timestamp_ms is None else float(timestamp_ms) - self._timestamp_ms
                reset = self._hidden is None or not 0 < dt <= 250.
                if reset:
                    self._hidden, self._previous = torch.zeros(1, 64), torch.zeros(1, self.feature_dim)
                result = self.module(
                    image_tensor, geometry_tensor, torch.tensor([[max(0., min(dt, 250.))]]),
                    torch.tensor([[float(reset)]]), self._hidden, self._previous,
                )
                directions, weights, self._hidden, self._previous, gate = result[:5]
                if len(result) >= 6 and isinstance(self.forecast, EmbeddedJointForecast):
                    self.forecast.set_motion(result[5].numpy()[0])
                self._timestamp_ms = float(timestamp_ms)
                self.reset_probability = float(gate.item())
                output = (directions, weights)
            else:
                output = self.module(image_tensor, geometry_tensor)
        if self.binocular:
            directions, weights = (part.numpy() for part in output)
            if weights.shape != (2,) or not np.isfinite(weights).all() or np.any(weights < 0) or not np.isclose(weights.sum(), 1., atol=1e-5):
                raise ValueError("binocular model produced invalid fusion weights")
        else:
            directions = output.numpy()
            weights = np.full(len(inputs), .5)
        predictions = []
        for direction, side, weight in zip(directions, sides, weights):
            local = np.asarray(direction, dtype=np.float64).copy()
            if side == "left":
                local[0] *= -1.0
            norm = float(np.linalg.norm(local))
            if not np.isfinite(norm) or norm <= 1e-9 or local[2] <= 1e-9:
                raise ValueError("conditioned-eye model produced an invalid gaze direction")
            local /= norm
            predictions.append(SharedEyePrediction(
                yaw=math.atan2(float(local[0]), float(local[2])),
                pitch=math.atan2(-float(local[1]), math.hypot(float(local[0]), float(local[2]))),
                nearest_group_distance=float("nan"),
                fusion_weight=float(weight),
            ))
        return predictions

    def spatial_fast_path_available(self):
        if not hasattr(self, "_spatial_fast_path"):
            torch, _ = require_torch()
            available = self.variant_metadata.get("selected_stage") == "personal_spatial" and hasattr(self.module, "base")
            if available:
                adapter = getattr(self.module, "adapter", None)
                available = adapter is not None and all(
                    hasattr(adapter, name) and all(torch.count_nonzero(p).item() == 0
                                                  for p in getattr(adapter, name).parameters())
                    for name in ("current", "residual"))
            self._spatial_fast_path = bool(available)
        return self._spatial_fast_path

    @classmethod
    def load(cls, metadata_path: Path, variant: str) -> "ConditionedEyeModel":
        torch, _ = require_torch()
        torch.set_num_threads(1)
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        item = (metadata.get("variants") or {}).get(variant) or {}
        module_file = item.get("module_file")
        if not module_file:
            raise ValueError(f"conditioned-eye metadata has no module for {variant}")
        module = torch.jit.load(str(metadata_path.with_name(module_file)), map_location="cpu").eval()
        result = cls(metadata, module, variant)
        if result.temporal:
            if item.get("joint_prediction"):
                result.forecast = EmbeddedJointForecast({
                    "schema": "opengazelink-joint-video-v1", "filter_config": {},
                    "shared_temporal_state": True, "max_horizon_ms": 100.,
                })
            else:
                from .video_forecast import LearnedForecast
                result.forecast = LearnedForecast.load_for(metadata_path, item)
        return result
