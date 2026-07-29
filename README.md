# OpenGazeLink

Low-latency RGB gaze provider for gaze-driven rendering experiments.

## Production Projects

- `phone-app/`: Camera2 YUV crop/JPEG/UDP Android sender.
- `pc-provider/`: normalized-eye Legacy/Tasks CNN calibration, control UI, gaze
  runtime, and shared-memory output.

## Current Direction

```text
Android Camera2 YUV_420_888
  -> crop during the YUV copy
  -> NV21/JPEG latest-frame UDP packet
  -> PC MediaPipe eye normalization
  -> shared two-eye appearance model
  -> raw gaze coordinates in Local\\EyeTracingGazeV1
```

The MediaPipe iris/3D geometry direction remains archived because occluded iris
landmarks lose required vertical information before calibration. The current
production route uses a normalized-eye tiny CNN on CPU so the GPU remains
available to the game. Legacy Face Mesh and Tasks Face Landmarker are alternative
landmark providers for the same CNN architecture; there is no alternate gaze
model or silent fallback.

## Setup

Normal users install the Windows release and use one of two shortcuts:

- `OpenGazeLink 控制中心`: pairing, configuration, calibration, preview, and diagnostics.
- `OpenGazeLink 后台运行`: load the saved configuration and write gaze directly to shared memory.

Both shortcuts launch the same executable. Opening the control shortcut while
the runtime is already active attaches the page to the existing receiver and
inference engine; it does not open a second video stream.

The Windows provider requires a 64-bit Windows installation. The Android sender
requires Android 8.0 (API 26) or newer. Both devices must be on the same trusted
local network.

For development:

```bat
cd /d OpenGazeLink\pc-provider
setup-venv.bat
start-control.bat
```

The control page is `http://127.0.0.1:8765/`. For the lowest-overhead game
runtime, save the desired model in the control page and run `start-runtime.bat`.
Build the Android sender with `phone-app/build-local.ps1`; its debug APK is
written to `phone-app/app/build/outputs/apk/debug/`.

The phone build script downloads its Gradle and Android SDK command-line tools
into `%USERPROFILE%\.eyetracing-build-tools` and requires JDK 17. It never writes
the SDK, generated APK, or local Android settings into the source tree.

## Documentation

- Production PC logic: [`pc-provider/ALGORITHM.md`](pc-provider/ALGORITHM.md)
- PC setup and commands: [`pc-provider/README.md`](pc-provider/README.md)
- Phone sender: [`phone-app/README.md`](phone-app/README.md)

## Calibration Prototype

The production collector records up to 9 paired appearance samples at each of
25 reference-light full-screen targets, repeats five anchors under steady dark
and bright backgrounds, then repeats five gaze targets at four fixed head-pose
conditions (left, right, up, and down). It saves accepted
frames losslessly, fits Legacy and Tasks CNN models, and only replaces the
canonical datasets and models after the complete pass succeeds. CNN lighting
profiles freeze the base model and fit only 16 first-layer FiLM parameters;
later five-target sessions can append profiles without a full recalibration.
There is no online mouse training, clipping, Kalman filter, moving average, or
silent fallback.

## Privacy And Security

OpenGazeLink processes camera frames, calibration frames, and trained personal
models locally. The application does not send data to a cloud service. Mutable
data is stored under `%LOCALAPPDATA%\OpenGazeLink` and is intentionally excluded
from this repository and the installer.

Phone discovery and frame transport use unencrypted UDP on the local network.
Pair only with a trusted PC and use a trusted LAN; this protocol is not designed
to protect camera frames against a malicious local-network participant.

## License

OpenGazeLink source code is licensed under [Apache-2.0](LICENSE). Runtime
dependencies and the bundled Face Landmarker model are listed in
[pc-provider/THIRD_PARTY_NOTICES.md](pc-provider/THIRD_PARTY_NOTICES.md).
