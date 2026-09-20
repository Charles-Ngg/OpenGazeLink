# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path
import os
from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs


root = Path(SPECPATH)
public_model = Path(os.environ.get("OPENGAZELINK_PUBLIC_MODEL_DIR", str(root / "models" / "public-conditioned"))).expanduser().resolve()
for filename in ("best.pt", "result.json"):
    if not (public_model / filename).is_file():
        raise SystemExit("Missing public base model asset: " + str(public_model / filename)
                         + ". See README.md: Base model assets. Personal calibration files must not be bundled.")
mediapipe_datas = collect_data_files("mediapipe", include_py_files=False)
mediapipe_binaries = collect_dynamic_libs("mediapipe")

a = Analysis(
    [str(root / "opengazelink_launcher.py")],
    pathex=[str(root)],
    binaries=mediapipe_binaries,
    datas=mediapipe_datas + [
        (str(root / "web"), "web"),
        (str(root / "models" / "face_landmarker.task"), "models"),
        (str(public_model / "best.pt"), "models/public-conditioned"),
        (str(public_model / "result.json"), "models/public-conditioned"),
        # Runtime modules live in PyInstaller's embedded archive.  Keep the
        # exact conditioned-eye source as provenance data so calibration can
        # record a reproducible preprocessing checksum in frozen builds.
        (str(root / "opengazelink_pc" / "conditioned_eye.py"), "provenance/opengazelink_pc"),
        *[(str(root / "opengazelink_pc" / name), "provenance/opengazelink_pc") for name in (
            "video_training.py", "personal_binocular_training.py", "video_network.py", "video_dataset.py", "video_session.py",
            "video_archive.py", "video_replay.py", "training_runtime.py", "normalized_eye.py", "latest_preprocessor.py", "camera.py", "one_euro.py",
            "video_forecast.py", "video_forecast_network.py", "video_forecast_training.py", "h264_stream.py",
            "prediction.py", "prediction_network.py", "prediction_dataset.py", "prediction_training.py", "prediction_timing.py", "prediction_evaluation.py",
            "event_temporal.py", "saccade_prediction.py", "stability_profile.py", "event_evaluation.py", "spatial_metrics.py", "head_coverage.py",
            "unified_prediction.py", "unified_prediction_training.py", "unified_capture.py", "unified_calibration_training.py", "calibration_split.py", "latency.py", "runtime_clock.py", "runtime_diagnostics.py", "runtime_scheduling.py", "transport_clock.py")],
        (str(root.parent / "NOTICE"), "."),
        (str(root.parent / "MODEL_NOTICE"), "."),
        (str(root.parent / "LICENSE"), "."),
        (str(root / "launcher" / "OpenGazeLink-Control.cmd"), "."),
        (str(root / "launcher" / "OpenGazeLink-Runtime.cmd"), "."),
        (str(root / "launcher" / "OpenGazeLink-Stop.cmd"), "."),
    ],
    hiddenimports=[
        "cv2",
        "av",
        "numpy",
        "torch",
        "mediapipe.tasks.python.core.base_options",
        "mediapipe.tasks.python.vision.face_landmarker",
        "mediapipe.python.solutions.face_mesh",
    ],
    hookspath=[],
    runtime_hooks=[],
    excludes=[
        "IPython", "jax", "jupyter", "scipy", "sounddevice",
        "tensorflow", "tkinter", "torch._dynamo", "torch._inductor",
        "torch.distributed", "torch.onnx", "torch.testing",
    ],
    noarchive=False,
)

# The Windows CPU Torch wheel contains hundreds of megabytes of linker-only
# static libraries. TorchScript inference needs the DLLs, never the .lib files.
a.binaries = [entry for entry in a.binaries if not entry[0].lower().endswith(".lib")]
a.datas = [
    entry for entry in a.datas
    if not entry[0].lower().endswith(".lib")
    and not entry[0].lower().replace("/", "\\").startswith("torch\\include\\")
]

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="OpenGazeLink",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    name="OpenGazeLink",
)
