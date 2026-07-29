# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path
from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs


root = Path(SPECPATH)
mediapipe_datas = collect_data_files("mediapipe", include_py_files=False)
mediapipe_binaries = collect_dynamic_libs("mediapipe")

a = Analysis(
    [str(root / "opengazelink_launcher.py")],
    pathex=[str(root)],
    binaries=mediapipe_binaries,
    datas=mediapipe_datas + [
        (str(root / "web"), "web"),
        (str(root / "models" / "face_landmarker.task"), "models"),
        (str(root.parent / "NOTICE"), "."),
        (str(root.parent / "LICENSE"), "."),
        (str(root / "launcher" / "OpenGazeLink-Control.cmd"), "."),
        (str(root / "launcher" / "OpenGazeLink-Runtime.cmd"), "."),
        (str(root / "launcher" / "OpenGazeLink-Stop.cmd"), "."),
    ],
    hiddenimports=[
        "cv2",
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
