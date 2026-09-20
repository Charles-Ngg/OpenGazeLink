"""Causal joint current-gaze and future-motion adaptation."""
import torch
from torch import nn
from torch.nn import functional as F


class VideoAdapter(nn.Module):
    def __init__(self, feature_dim=148, detach_motion_state=False, event_motion=False):
        super().__init__()
        self.feature_dim = feature_dim
        self.detach_motion_state = bool(detach_motion_state)
        self.event_motion = bool(event_motion)
        # Preserve the signed frame-to-frame change explicitly.  At 120 FPS a
        # recurrent state alone has to recover a small ~8 ms difference from
        # appearance features; the normalized causal delta gives the shared
        # GRU and future head a direct motion signal without looking ahead.
        self.encoder = nn.Sequential(nn.Linear(feature_dim * 2 + 1, 64), nn.LayerNorm(64), nn.SiLU())
        self.current = nn.Linear(64, 6)
        self.reset_gate = nn.Sequential(nn.Linear(feature_dim + 1, 32), nn.SiLU(), nn.Linear(32, 1))
        self.memory = nn.GRUCell(64, 64)
        self.residual = nn.Linear(64, 6)
        # The same recurrent state that corrects current gaze also predicts a
        # screen-normalized velocity and acceleration.  Keeping this head here
        # lets future-coordinate loss update the eye features, binocular
        # fusion and temporal state in one graph.
        # Legacy motion: velocity(2), acceleration(2). Event motion adds
        # fixation/pursuit/saccade logits(3), landing offset(2), remaining
        # saccade time(1), and predictive uncertainty(1).
        self.forecast = nn.Sequential(nn.Linear(64, 32), nn.SiLU(), nn.Linear(32, 11 if event_motion else 4))
        nn.init.zeros_(self.current.weight)
        nn.init.zeros_(self.current.bias)
        nn.init.zeros_(self.residual.weight)
        nn.init.zeros_(self.residual.bias)
        nn.init.zeros_(self.forecast[-1].weight)
        nn.init.zeros_(self.forecast[-1].bias)
        nn.init.constant_(self.reset_gate[-1].bias, -3.)

    def forward(self, features, dt, reset, hidden, previous):
        scaled_dt = dt / 100.
        delta = (features - previous) / scaled_dt.clamp_min(.05)
        delta = torch.where(reset > 0, torch.zeros_like(delta), delta).clamp(-10., 10.)
        encoded = self.encoder(torch.cat((features, delta, scaled_dt), dim=-1))
        gate = torch.sigmoid(self.reset_gate(torch.cat((torch.abs(features - previous), scaled_dt), dim=-1)))
        keep = (1. - reset) * (1. - gate)
        keep = torch.where(gate >= .8, torch.zeros_like(keep), keep)
        hidden = self.memory(encoded, hidden * keep)
        correction = self.current(encoded) + keep * self.residual(hidden)
        direction = F.normalize(features[:, :6].reshape(-1, 2, 3) + .1 * torch.tanh(correction.reshape(-1, 2, 3)), dim=-1)
        # The future task can learn from the same visual evidence without
        # sending its noisy/ambiguous labels through the current-gaze memory.
        # Encoder gradients remain shared; only the recurrent localization
        # state is protected in this ablation.
        motion_state = encoded + hidden.detach() if self.detach_motion_state else hidden
        raw_motion = self.forecast(motion_state)
        # Units are normalized-screen / second and / second squared.  Bounds
        # prevent an uncertain transition from producing an unbounded lead.
        motion = torch.cat((3. * torch.tanh(raw_motion[:, :2]),
                            20. * torch.tanh(raw_motion[:, 2:4])), dim=1)
        if self.event_motion:
            motion = torch.cat((motion, raw_motion[:, 4:7],
                                .5 * torch.tanh(raw_motion[:, 7:9]),
                                .008 + .112 * torch.sigmoid(raw_motion[:, 9:10]),
                                .003 + .097 * torch.sigmoid(raw_motion[:, 10:11])), dim=1)
        return direction, hidden, features, gate, motion


class VideoInference(nn.Module):
    def __init__(self, base, adapter, mean, scale, appearance=None,
                 geometry_reference_indices=(), geometry_reference_values=(),
                 geometry_reference_tolerance=0.25):
        super().__init__()
        self.base, self.adapter = base, adapter
        self.appearance = appearance
        self.register_buffer("mean", torch.as_tensor(mean, dtype=torch.float32))
        self.register_buffer("scale", torch.as_tensor(scale, dtype=torch.float32))
        self.register_buffer("geometry_reference_indices", torch.as_tensor(
            geometry_reference_indices, dtype=torch.long,
        ))
        self.register_buffer("geometry_reference_values", torch.as_tensor(
            geometry_reference_values, dtype=torch.float32,
        ))
        self.has_geometry_reference = bool(len(geometry_reference_indices))
        self.geometry_reference_tolerance = float(geometry_reference_tolerance)

    def forward(self, images, geometry, dt, reset, hidden, previous):
        if self.has_geometry_reference:
            geometry = geometry.clone()
            current = geometry[:, self.geometry_reference_indices]
            limit = self.geometry_reference_tolerance * self.geometry_reference_values.abs().clamp_min(1e-6)
            geometry[:, self.geometry_reference_indices] = torch.where(
                (current - self.geometry_reference_values).abs() > limit,
                self.geometry_reference_values, current,
            )
        directions, weights = self.base(images, geometry)
        features = torch.cat((directions.reshape(1, 6), weights.reshape(1, 2),
                              ((geometry - self.mean) / self.scale).clamp(-10., 10.).reshape(1, 140)), -1)
        if self.appearance is not None:
            features = torch.cat((features, self.appearance(images).reshape(1, 256)), -1)
        corrected, hidden, previous, gate, motion = self.adapter(features, dt, reset, hidden, previous)
        return corrected.reshape(2, 3), weights, hidden, previous, gate, motion


class VideoAppearance(nn.Module):
    """Reuse the frozen CNN's 128 per-eye features, without target telemetry."""
    def __init__(self, image_network):
        super().__init__()
        self.image = image_network

    def forward(self, patches):
        image = patches.to(torch.float32) / 255.
        gray = image[:, 0]
        mean = gray.mean(dim=(1, 2), keepdim=True)
        scale = gray.std(dim=(1, 2), unbiased=False, keepdim=True).clamp_min(10 / 255.)
        normalized = torch.stack(((gray - mean) / scale, image[:, 1]), dim=1)
        return self.image(normalized)
