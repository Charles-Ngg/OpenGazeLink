from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Any

from .paths import CONFIG_PATH

DEFAULT_CONFIG_PATH = CONFIG_PATH


@dataclass
class ProviderConfig:
    udp_bind: str = "0.0.0.0"
    udp_port: int = 5007
    discovery_port: int = 5006
    paired_phone_id: str = ""
    paired_phone_name: str = ""
    rotate: str = "auto"
    mirror: bool = False
    landmarker: str = "legacy"
    lighting_profile: str = "reference"
    screen_width: int = 3840
    screen_height: int = 2160
    screen_diagonal_inches: float = 27.0
    camera_offset_x_cm: float = 0.0
    camera_offset_y_cm: float = 16.81
    camera_offset_z_cm: float = 0.0
    geometry_configured: bool = False
    shared_memory_name: str = "Local\\EyeTracingGazeV1"
    control_bind: str = "127.0.0.1"
    control_port: int = 8765
    one_euro_enabled: bool = True
    one_euro_min_cutoff: float = 1.0
    one_euro_beta: float = 2.0
    one_euro_derivative_cutoff: float = 1.0
    extrapolation_enabled: bool = True
    extrapolation_horizon_ms: float = 85.0
    extrapolation_max_lead_fraction: float = 0.12
    motion_diagnostics_enabled: bool = False

    def update(self, values: dict[str, Any]) -> None:
        if "udp_port" in values:
            self._validate_port(values["udp_port"], "UDP port")
        if "control_port" in values:
            self._validate_port(values["control_port"], "control port")
        if "discovery_port" in values:
            self._validate_port(values["discovery_port"], "discovery port")
        for key, value in values.items():
            if not hasattr(self, key):
                continue
            current = getattr(self, key)
            if isinstance(current, bool):
                setattr(self, key, bool(value))
            elif isinstance(current, int):
                setattr(self, key, int(value))
            elif isinstance(current, float):
                setattr(self, key, float(value))
            else:
                setattr(self, key, str(value))
        rotate = str(self.rotate).lower()
        self.rotate = rotate if rotate in ("auto", "0", "90", "180", "270") else "auto"
        self.landmarker = self.landmarker if self.landmarker in ("tasks", "legacy") else "legacy"
        self.screen_width = max(320, min(16384, self.screen_width))
        self.screen_height = max(240, min(16384, self.screen_height))
        self.screen_diagonal_inches = self._finite_clamped(
            self.screen_diagonal_inches, 10.0, 100.0,
        )
        self.camera_offset_x_cm = self._finite_clamped(self.camera_offset_x_cm, -200.0, 200.0)
        self.camera_offset_y_cm = self._finite_clamped(self.camera_offset_y_cm, -200.0, 200.0)
        self.camera_offset_z_cm = self._finite_clamped(self.camera_offset_z_cm, -200.0, 200.0)
        self.one_euro_min_cutoff = max(0.01, min(30.0, self.one_euro_min_cutoff))
        self.one_euro_beta = max(0.0, min(10.0, self.one_euro_beta))
        self.one_euro_derivative_cutoff = max(0.01, min(30.0, self.one_euro_derivative_cutoff))
        self.extrapolation_horizon_ms = self._finite_clamped(
            self.extrapolation_horizon_ms, 0.0, 300.0,
        )
        self.extrapolation_max_lead_fraction = self._finite_clamped(
            self.extrapolation_max_lead_fraction, 0.0, 0.5,
        )

    @staticmethod
    def _finite_clamped(value: float, minimum: float, maximum: float) -> float:
        value = float(value)
        if not math.isfinite(value):
            raise ValueError("screen/camera geometry must contain finite numbers")
        return max(minimum, min(maximum, value))

    @staticmethod
    def _validate_port(value: Any, label: str) -> None:
        try:
            port = int(value)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{label} must be an integer from 1 to 65535") from error
        if isinstance(value, float) and not value.is_integer():
            raise ValueError(f"{label} must be an integer from 1 to 65535")
        if not 1 <= port <= 65535:
            raise ValueError(f"{label} must be an integer from 1 to 65535")

    def require_geometry(self) -> None:
        if not self.geometry_configured:
            raise RuntimeError(
                "configure screen resolution, physical size, and camera position before calibration"
            )

    @property
    def camera_position_screen_cm(self) -> tuple[float, float, float]:
        """Camera position from screen centre: right, down, toward user are positive."""
        return (
            self.camera_offset_x_cm,
            self.camera_offset_y_cm,
            self.camera_offset_z_cm,
        )


def load_config(path: Path = DEFAULT_CONFIG_PATH) -> ProviderConfig:
    config = ProviderConfig()
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            config.update(payload)
    return config


def save_config(config: ProviderConfig, path: Path = DEFAULT_CONFIG_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(asdict(config), ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)
