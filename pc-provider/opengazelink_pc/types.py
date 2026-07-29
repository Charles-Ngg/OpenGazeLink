from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass(frozen=True)
class HeadPose:
    yaw: float
    pitch: float
    x: float
    y: float
    z: float
    rotation: Optional[Tuple[Tuple[float, float, float], Tuple[float, float, float], Tuple[float, float, float]]] = None


@dataclass(frozen=True)
class FeatureSample:
    t_ms: float
    features: Tuple[float, ...]
    confidence: float
    head: HeadPose
    virtual_point: Optional[Tuple[float, float]] = None
    proxy: Optional[dict] = None


@dataclass(frozen=True)
class GazeSample:
    t_ms: float
    x: float
    y: float
    raw_x: float
    raw_y: float
    confidence: float
    valid: bool
    status: str


@dataclass(frozen=True)
class CalibrationData:
    version: int
    created_at: str
    screen_width: int
    screen_height: int
    feature_names: List[str]
    x_calib: List[List[float]]
    y_calib: List[List[float]]
    points: List[dict]
    model: str = "thin_plate_spline"
    smoothing: Optional[dict] = None
    preprocess: Optional[dict] = None
