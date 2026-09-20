"""Research-only dual-head models; public gaze coordinates never mix with local labels."""
from __future__ import annotations

from .shared_eye_models import require_torch, _tiny_cnn_class


def make_transfer_model(architecture):
    torch, nn = require_torch()

    class ResidualBlock(nn.Module):
        def __init__(self, incoming, outgoing, stride=1):
            super().__init__()
            self.layers = nn.Sequential(
                nn.Conv2d(incoming, outgoing, 3, stride=stride, padding=1, bias=False),
                nn.GroupNorm(8, outgoing), nn.LeakyReLU(.05),
                nn.Conv2d(outgoing, outgoing, 3, padding=1, bias=False),
                nn.GroupNorm(8, outgoing),
            )
            self.skip = nn.Identity() if incoming == outgoing and stride == 1 else nn.Conv2d(incoming, outgoing, 1, stride=stride, bias=False)
            self.activation = nn.LeakyReLU(.05)

        def forward(self, value):
            return self.activation(self.layers(value) + self.skip(value))

    class DualHeadEye(nn.Module):
        def __init__(self):
            super().__init__()
            if architecture == "tiny":
                base = _tiny_cnn_class(torch, nn)()
                self.features = base.features
                self.embedding = nn.Sequential(*list(base.head.children())[:3])
                self.local_head = base.head[3]
                dimension = 16
            elif architecture == "resnet":
                self.features = nn.Sequential(
                    nn.Conv2d(2, 32, 3, padding=1, bias=False), nn.GroupNorm(8, 32), nn.LeakyReLU(.05),
                    ResidualBlock(32, 32), ResidualBlock(32, 32),
                    ResidualBlock(32, 64, 2), ResidualBlock(64, 64),
                    ResidualBlock(64, 128, 2), ResidualBlock(128, 128),
                    ResidualBlock(128, 256, 2), ResidualBlock(256, 256),
                    nn.AdaptiveAvgPool2d((3, 5)),
                )
                self.embedding = nn.Sequential(nn.Flatten(), nn.Linear(256 * 3 * 5, 128), nn.LeakyReLU(.05))
                dimension = 128
                self.local_head = nn.Linear(dimension, 2)
            else:
                raise ValueError(f"unknown architecture: {architecture}")
            self.public_head = nn.Linear(dimension, 2)

        def encode(self, value):
            return self.embedding(self.features(value))

        def forward(self, value):
            return self.local_head(self.encode(value))

        def public(self, value):
            return self.public_head(self.encode(value))

    return DualHeadEye()
