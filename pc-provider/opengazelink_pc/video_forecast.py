"""Causal short-horizon gaze displacement prediction, with explicit history.

Training targets are later observations of the frozen gaze model, not measured
human gaze or future stimulus coordinates. The current gaze model stays frozen.
"""
from collections import deque
import hashlib
import json
from pathlib import Path

import numpy as np

SCHEMA = "opengazelink-video-forecast-v1"
HORIZONS = (33., 67., 85., 100.)
FEATURE_DIM = 483


class ForecastHistory:
    def __init__(self):
        self.samples = deque(maxlen=4)

    def reset(self):
        self.samples.clear()

    def update(self, timestamp, raw, stable, feature, hidden, reset=False):
        raw, stable = np.asarray(raw), np.asarray(stable)
        feature, hidden = np.asarray(feature).reshape(-1), np.asarray(hidden).reshape(-1)
        if feature.shape != (404,) or hidden.shape != (64,) or raw.shape != (2,) or stable.shape != (2,):
            raise ValueError("invalid forecast input shapes")
        if not np.isfinite(np.concatenate(([timestamp], raw, stable, feature, hidden))).all():
            self.reset()
            raise ValueError("non-finite forecast inputs")
        jump = False
        if self.samples:
            gap = timestamp - self.samples[-1][0]
            jump = np.linalg.norm(raw - self.samples[-1][1]) > .08
            reset = reset or not 5. <= gap <= 100. or jump
        if reset:
            self.reset()
        self.samples.append((float(timestamp), raw.copy(), stable.copy(), feature.copy()))
        if len(self.samples) < 4:
            return None, 0., "forecast_reset" if jump or reset else "forecast_warmup"
        times, raws, points, features = map(np.array, zip(*self.samples))
        delta = points[1:] - points[:-1]
        # Small motion fades continuously to zero: no learned positional offset
        # can move a truly stationary history. A reversal suppresses old lead.
        span = float(np.linalg.norm(points[-1] - points[0]))
        activity = float(np.clip((span - .0015) / .0045, 0., 1.))
        if np.dot(delta[-1], delta[-2]) < 0:
            activity = 0.
        vector = np.concatenate((delta.flatten() * 100., np.diff(raws, axis=0).flatten() * 100.,
                                 np.diff(times) / 33.333333, features[-1] - features[0], hidden))
        return vector.astype(np.float32), activity, "learned_forecast" if activity else "fixation"


class LearnedForecast:
    def __init__(self, module, metadata):
        self.module, self.metadata = module, metadata
        self.history = ForecastHistory()

    def reset(self):
        self.history.reset()

    def update(self, raw, stable, timestamp, feature, hidden, screen_size, horizon_ms,
               max_lead_fraction=.12, reset=False):
        import torch
        wh = np.maximum(1., np.asarray(screen_size, dtype=float) - 1.)
        horizon = float(np.clip(horizon_ms, 0., 100.))
        vector, activity, mode = self.history.update(timestamp, np.asarray(raw)/wh, np.asarray(stable)/wh,
                                                     feature, hidden, reset)
        lead = np.zeros(2)
        if vector is not None and horizon > 0 and activity > 0:
            with torch.inference_mode():
                displacement = self.module(torch.from_numpy(vector[None]), torch.tensor([[horizon/100.]]))[0].numpy()
            lead = displacement * activity * wh
            if not np.isfinite(lead).all():
                self.reset()
                raise ValueError("forecast produced non-finite displacement")
            limit = min(.06, max(0., max_lead_fraction)) * np.linalg.norm(wh)
            lead *= min(1., limit / max(1e-9, np.linalg.norm(lead)))
        if horizon == 0:
            mode = "disabled"
        output = np.asarray(stable) + lead
        return tuple(output), {"mode":mode, "horizon_ms":horizon, "requested_horizon_ms":float(horizon_ms),
                               "target_source_ms":float(timestamp)+horizon, "lead_px":lead.tolist(),
                               "lead_distance_px":float(np.linalg.norm(lead)), "activity":activity,
                               "sample_count":len(self.history.samples), "label_source":"future_frozen_model_observation"}

    @classmethod
    def load_for(cls, model_path, item):
        import torch
        path = Path(model_path).with_name("conditioned-video-forecast.json")
        if not path.is_file():
            return None
        metadata = json.loads(path.read_text(encoding="utf-8"))
        from .prediction import SCHEMA as MOTION_SCHEMA, MotionPrediction
        from .unified_prediction import SCHEMA as UNIFIED_SCHEMA, UnifiedPrediction
        if (metadata.get("schema") not in (SCHEMA, MOTION_SCHEMA, UNIFIED_SCHEMA) or not metadata.get("accepted")
                or metadata.get("base_sha256") != item.get("module_sha256")):
            return None
        module_path = path.with_name(metadata["module_file"])
        if hashlib.sha256(module_path.read_bytes()).hexdigest() != metadata["module_sha256"]:
            raise ValueError("forecast module checksum mismatch")
        runtime = UnifiedPrediction if metadata["schema"] == UNIFIED_SCHEMA else MotionPrediction if metadata["schema"] == MOTION_SCHEMA else cls
        return runtime(torch.jit.load(str(module_path), map_location="cpu").eval(), metadata)
