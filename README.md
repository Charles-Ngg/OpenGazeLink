# OpenGazeLink

OpenGazeLink is a low-latency gaze-tracking system for gaze-driven rendering and
interaction experiments. An Android phone captures and streams camera frames,
while a Windows PC performs eye normalization, personalized model inference,
post-processing, and publication of screen-space gaze coordinates through shared
memory.

> **Project status: Experimental.** The current release supports pairing,
> calibration, real-time inference, and background output. Accuracy and latency
> still depend on the phone camera, mounting position, lighting, the user's eye
> characteristics, and calibration quality.

## Features

- Captures YUV frames with Android Camera2 and crops them during the copy step
  before JPEG encoding.
- Uses latest-only UDP transport: newer frames replace stale frames instead of
  building an ever-growing processing queue.
- Automatically discovers and pairs devices, so a PC address change does not
  require another manual `ipconfig` lookup.
- Receives Camera2 intrinsics for the selected front or rear camera and transforms
  them according to cropping, rotation, and mirroring.
- Runs either MediaPipe Face Mesh or Tasks Face Landmarker on the PC CPU.
- Uses a per-user tiny CNN to estimate each eye's gaze angle relative to the head.
- Supports reference, dark, bright, and user-defined lighting adaptation profiles.
- Provides a 1 Euro stabilization filter and state-aware latency compensation
  designed for input at approximately 30 FPS.
- Includes a control center, full-screen calibration, gaze preview, runtime
  diagnostics, and a minimal background mode.
- Publishes the latest gaze point to rendering applications through Windows shared
  memory.

## Pipeline

```text
Android Camera2 YUV_420_888
  -> crop while copying YUV planes
  -> NV21 / JPEG encoding
  -> latest-only UDP packets
  -> Windows receive, reassembly, and JPEG decode
  -> MediaPipe facial landmarks and head-pose estimation
  -> 3D eye-plane reprojection and eye-corner mapping
  -> 64 x 36 grayscale eye image + soft eyelid-aperture mask
  -> personalized tiny CNN shared by both eyes (CPU)
  -> ray intersection with the screen plane for each eye
  -> binocular fusion, 1 Euro filtering, and state-aware compensation
  -> `Local\EyeTracingGazeV1` shared memory
```

OpenGazeLink uses MediaPipe for facial landmarks, eye-region localization, and
head-pose estimation. Final gaze direction is not calculated directly from
MediaPipe iris landmarks or a spherical iris model. It is predicted by a
personalized eye-appearance CNN. The CNN runs on the CPU to keep the GPU available
for the game or rendering workload.

## Quick Start

### 1. Install

Download these files from GitHub Releases:

- `OpenGazeLink-Setup-0.1.0.exe`: Windows installer.
- `OpenGazeLink-portable-0.1.0.zip`: portable Windows package.
- `opengazelink-phone-debug.apk`: Android sender.

The Windows application requires 64-bit Windows. The Android application requires
Android 8.0 (API 26) or later. Connect the phone and PC to the same trusted local
network.

The Windows packages are not commercially code-signed, so Windows SmartScreen may
display an unknown-publisher warning.

### 2. Pair the Phone

1. Start **OpenGazeLink Control Center**.
2. Keep automatic discovery enabled in the Android application.
3. When the phone appears in the control center, verify its device name and approve
   the pairing request.
4. After pairing, low-frequency discovery heartbeats automatically update the PC
   address when it changes.

If the network blocks UDP broadcast, enter the PC address manually in the Android
application as a fallback.

### 3. Configure the Camera and Screen

1. On the phone, select the front or rear camera, rotation, crop region, resolution,
   JPEG quality, and send rate.
2. Adjust the phone preview so the entire head remains visible throughout the
   expected movement range and the final image behaves like a mirror.
3. Tap **Send camera intrinsics** on the phone.
4. In the PC control center, enter the screen resolution, diagonal size, and camera
   position relative to the screen center.

Camera position is measured in centimeters. Positive axes point right, down, and
toward the user. Perform a new full calibration after changing the camera, crop,
rotation, screen, or camera mounting position.

### 4. Run a Full Calibration

A full calibration contains:

- A screen-filling `5 x 5` gaze grid with a neutral head pose.
- Five repeated anchors under dark and bright lighting.
- Five gaze targets at the center and corners for each of four head poses: left,
  right, up, and down.

Each accepted frame produces one input for each eye. OpenGazeLink separately trains
CNNs for the Legacy and Tasks MediaPipe backends. A new dataset and its models
replace the active calibration only after the complete process succeeds; an
interrupted or failed calibration does not damage the working calibration.

Calibration data and trained models are stored under
`%LOCALAPPDATA%\OpenGazeLink`, outside the installation directory.

### 5. Run OpenGazeLink

- **Control Center** handles pairing, configuration, calibration, preview, and
  diagnostics.
- **Background Runtime** loads the saved configuration and starts only UDP receive,
  the selected MediaPipe backend, CNN inference, and shared-memory output. It does
  not start the web control service or video preview.

Both modes use the same inference and post-processing path. The final fused point
shown in preview is the same coordinate published to shared memory.

## Calibration Model and Lighting Adaptation

The CNN receives a two-channel `64 x 36` eye image:

1. A robustly normalized grayscale eye crop.
2. A soft eyelid-aperture mask generated from eyelid landmarks.

Both eyes share one CNN. One eye is mirrored so both use the same coordinate
convention. The model predicts horizontal and vertical eye-in-head angles, then
OpenGazeLink intersects rays from the two eyes' individual 3D positions with the
screen plane.

Full calibration trains the base CNN under reference lighting. Dark, bright, and
later user-defined lighting profiles do not retrain the entire network. They freeze
the base network and fit only a small set of FiLM gain and bias parameters in the
first layer, allowing shorter incremental calibration under new lighting.

## Stabilization and Latency Compensation

After binocular fusion, OpenGazeLink can apply a 1 Euro filter. State-aware
compensation uses the phone's source timestamps to distinguish fixation, continuous
motion, saccade onset, and landing:

- During fixation, it emits a stable point to suppress small-scale noise.
- During continuous motion, it performs bounded extrapolation over the configured
  prediction horizon.
- During saccade onset, it uses a short measured-velocity prediction and bypasses
  the stabilization filter.
- On landing, it immediately follows the latest raw point and resets the filter
  state.

The current predictor can reduce visible artifacts caused by approximately 80-90 ms
of pipeline latency. It cannot reliably recover a saccade target that has not yet
appeared in the first 30 FPS observation.

## Shared Memory Interface

The Windows runtime publishes a fixed-size `Local\EyeTracingGazeV1` shared-memory
slot containing:

- A validity flag and sequence lock.
- Screen-space pixel coordinates `x / y`.
- Screen width and height.
- Confidence.
- PC publication and phone source-frame timestamps.

## Current Limitations

- Personalized calibration is required for the user, phone camera, screen, and
  mounting position.
- At extreme head poses, MediaPipe landmark errors and residual eye-reprojection
  differences can reduce accuracy.
- Lighting changes outside the existing adaptation range require a new lighting
  profile.
- The 30 FPS Camera2 input and phone ISP latency are major sources of end-to-end
  delay.
- UDP transport is unencrypted. Use it only on a trusted LAN and do not expose it
  to the internet.
- The PC runtime currently supports Windows only; no macOS or Linux shared-memory
  implementation is provided.
- The release packages are not code-signed, and the project has not yet undergone
  large-scale testing across many users and devices.

## Build from Source

### Windows PC

Python 3.8 is required. From `pc-provider`, run:

```bat
setup-venv.bat
start-control.bat
```

For the lowest-overhead runtime, use:

```bat
start-runtime.bat
```

To build the Windows packages:

```powershell
./build-release.ps1
```

The script produces a PyInstaller `onedir` directory and portable ZIP. If Inno
Setup 6 or 7 is installed, it also builds the installer. Release packages do not
contain personal calibration data.

### Android

JDK 17 is required. From `phone-app`, run:

```powershell
./build-local.ps1
```

The script downloads and reuses Gradle and the Android SDK command-line tools under
`%USERPROFILE%\.eyetracing-build-tools`. The generated APK is written to:

```text
phone-app/app/build/outputs/apk/debug/app-debug.apk
```

### Tests

```powershell
cd pc-provider
./.venv/Scripts/python.exe -m unittest discover -s tests -q
node --check web/app.js
```

GitHub Actions runs the same PC tests and frontend syntax check on Windows.

## Repository Layout

```text
OpenGazeLink/
  phone-app/       Android Camera2 capture and UDP sender
  pc-provider/     Windows receive, calibration, CNN inference, control UI,
                   and shared-memory output
  LICENSE          Apache License 2.0
```

Runtime configuration, camera intrinsics, calibration frames, trained models, logs,
APKs, virtual environments, and Windows release artifacts are excluded by
`.gitignore`. Before opening a public issue, verify that attachments do not contain
face images or personal calibration data.

## Privacy and Security

OpenGazeLink does not use cloud services. Camera frames, calibration data, and
personalized models remain on the phone, trusted local network, and local Windows
PC.

Camera2 frames, discovery messages, and camera intrinsics are sent over unencrypted
UDP. Use OpenGazeLink only on a trusted private network, and do not expose UDP ports
`5006` or `5007` to the internet.

## License and Third-Party Components

OpenGazeLink source code is licensed under the
[Apache License 2.0](LICENSE).

MediaPipe, the Face Landmarker model, OpenCV, NumPy, PyTorch, AndroidX, Kotlin, and
other third-party components remain subject to their respective open-source
licenses.
