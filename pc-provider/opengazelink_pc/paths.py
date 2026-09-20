from __future__ import annotations

import os
from pathlib import Path
import shutil
import sys


APP_NAME = "OpenGazeLink"
LEGACY_APP_NAME = "EyeTracing"
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
FROZEN = bool(getattr(sys, "frozen", False))


def resource_root() -> Path:
    override = os.environ.get("OPENGAZELINK_RESOURCE_DIR") or os.environ.get(
        "EYETRACING_RESOURCE_DIR"
    )
    if override:
        return Path(override).expanduser().resolve()
    if FROZEN:
        bundle = getattr(sys, "_MEIPASS", None)
        return Path(bundle if bundle else Path(sys.executable).parent).resolve()
    return PACKAGE_ROOT


def user_root() -> Path:
    override = os.environ.get("OPENGAZELINK_USER_DIR") or os.environ.get(
        "EYETRACING_USER_DIR"
    )
    if override:
        return Path(override).expanduser().resolve()
    local_app_data = os.environ.get("LOCALAPPDATA")
    base = Path(local_app_data) if local_app_data else Path.home() / "AppData" / "Local"
    root = base / APP_NAME
    return (root if FROZEN else root / "Development").resolve()


RESOURCE_ROOT = resource_root()
USER_ROOT = user_root()
DATA_DIR = USER_ROOT / "data"
CONFIG_PATH = USER_ROOT / "config.json"
WEB_DIR = RESOURCE_ROOT / "web"
LANDMARKER_MODEL_DIR = RESOURCE_ROOT / "models"
LOG_DIR = DATA_DIR / "logs"
MOTION_DIAGNOSTICS_DIR = DATA_DIR / "motion-diagnostics"


_ACTIVE_DATA_FILES = (
    "camera_intrinsics.json",
    "lighting-profile-library.json",
    "shared-eye-angle-calibration.json",
    "shared-eye-legacy-angle-calibration.json",
    "shared-eye-cnn-model.json",
    "shared-eye-cnn-module.pt",
    "shared-eye-legacy-cnn-model.json",
    "shared-eye-legacy-cnn-module.pt",
    "conditioned-eye-model.json",
    "conditioned-eye-with-iris.pt",
    "conditioned-eye-without-iris.pt",
    "conditioned-binocular-model.json",
    "conditioned-eye-binocular.pt",
    "conditioned-video-model.json",
)


def ensure_user_layout(migrate_from: Path | None = None) -> None:
    """Create writable directories and migrate active artifacts once."""
    if FROZEN:
        local_app_data = os.environ.get("LOCALAPPDATA")
        base = Path(local_app_data) if local_app_data else Path.home() / "AppData" / "Local"
        legacy_root = (base / LEGACY_APP_NAME).resolve()
        if legacy_root.is_dir():
            USER_ROOT.mkdir(parents=True, exist_ok=True)
            for source in legacy_root.iterdir():
                target = USER_ROOT / source.name
                if not target.exists():
                    shutil.move(str(source), str(target))
            try:
                legacy_root.rmdir()
            except OSError:
                pass
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    MOTION_DIAGNOSTICS_DIR.mkdir(parents=True, exist_ok=True)
    if not FROZEN or migrate_from is None:
        return
    source_root = migrate_from.expanduser().resolve()
    source_data = source_root / "data"
    source_config = source_root / "config.json"
    if source_config.is_file() and not CONFIG_PATH.exists():
        shutil.copy2(source_config, CONFIG_PATH)
    for name in _ACTIVE_DATA_FILES:
        source = source_data / name
        target = DATA_DIR / name
        if source.is_file() and not target.exists():
            shutil.copy2(source, target)
