import torch
from torch import nn


class ForecastNetwork(nn.Module):
    def __init__(self, mean, scale):
        super().__init__()
        self.register_buffer("mean", torch.as_tensor(mean, dtype=torch.float32))
        self.register_buffer("scale", torch.as_tensor(scale, dtype=torch.float32))
        self.net = nn.Sequential(nn.Linear(484, 48), nn.SiLU(), nn.Linear(48, 2))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.motion_gain = nn.Parameter(torch.zeros(1))

    def forward(self, features, horizon):
        x = ((features - self.mean) / self.scale).clamp(-6., 6.)
        # Mean of the three observed stable displacements / elapsed source time.
        displacement = features[:, :6].reshape(-1, 3, 2).sum(1) / 100.
        elapsed = features[:, 12:15].sum(1, keepdim=True) * 33.333333
        velocity_lead = displacement / elapsed.clamp_min(1.) * horizon * 100.
        residual = self.net(torch.cat((x, horizon), 1)) * .01 * horizon
        return .06 * torch.tanh((self.motion_gain * velocity_lead + residual) / .06)
