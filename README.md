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

- Offers two Android modes: constrained high-speed Camera2 sessions (120 FPS or
  higher, hardware H.264 over TCP) and standard camera capture (JPEG over UDP).
- Lists supported camera resolution/frame-rate combinations, checking hardware
  encoder compatibility for high-speed capture. High-speed mode sends the full
  frame; Camera mode supports pre-encode cropping and software JPEG quality tiers
  (Q50/65/80/90/95/100, default Q80), without resizing.
- Keeps PC processing on fresh frames with bounded transport and processing queues.
- Automatically discovers and pairs devices, so a PC address change does not
  require another manual `ipconfig` lookup.
- Automatically receives Camera2 intrinsics for the selected camera and applies
  the requested orientation on the PC.
- Uses Tasks Face Landmarker and the current pretrained, personally adapted gaze
  model. Legacy model selection and lighting presets are no longer runtime options.
- Integrates automatic fixation stability with a single saccade-prediction switch.
- Provides bilingual Chinese/English interfaces, two-stage full-screen calibration,
  gaze preview, live performance monitoring, and a minimal background mode.
- Publishes the latest gaze point to rendering applications through Windows shared
  memory.

## Pipeline

```text
Android Camera2 (complete selected frame)
  -> high-speed encoder Surface -> hardware H.264 / TCP
     or standard YUV capture -> copy retained crop -> software JPEG / UDP
  -> Windows decode and latest-frame processing
  -> Tasks landmarks, head pose, and normalized eye inputs
  -> current pretrained model with personal position calibration (CPU)
  -> per-eye screen-plane projection and binocular fusion
  -> automatic fixation stability + optional saccade prediction
  -> `Local\EyeTracingGazeV1` shared memory
```

OpenGazeLink uses MediaPipe for facial landmarks, eye-region localization, and
head-pose estimation. Final gaze direction is not calculated directly from
MediaPipe iris landmarks or a spherical iris model. It is predicted by a
personally adapted eye-appearance model. The model runs on the CPU to keep the GPU available
for the game or rendering workload.

## Quick Start

### 1. Install

Download these files from GitHub Releases:

- `OpenGazeLink-Setup-<version>.exe`: Windows installer.
- `OpenGazeLink-portable-<version>.zip`: portable Windows package.
- `opengazelink-phone-debug.apk`: Android sender.

The interface described here is the current source version. Older released
packages may still contain the development controls; use matching newly built
PC and phone versions for the simplified interface.

The Windows application requires 64-bit Windows. The Android application requires
Android 8.0 (API 26) or later. Connect the phone and PC to the same trusted local
network.

The Windows packages are not commercially code-signed, so Windows SmartScreen may
display an unknown-publisher warning.

### 2. Pair the Phone

1. Start **OpenGazeLink Control Center**.
2. Tap **Find PC / 发现电脑** in the Android application; discovery also runs automatically.
3. When the phone appears in the control center, verify its device name and approve
   the pairing request.
4. After pairing, low-frequency discovery heartbeats automatically update the PC
   address when it changes.

If the network blocks UDP broadcast, enter the PC address manually in the Android
application as a fallback.

### 3. Configure the Camera and Screen

1. On the phone, choose **High-speed session / 高速会话** or **Camera / 摄像头**,
   then select a camera. Choose resolution and target frame rate separately in
   either order; the other field adjusts only when needed to keep a supported pair.
2. Check the image on the PC. Keep the head visible over the expected movement
   range. In Camera mode, set software JPEG quality and optional edge crops
   (0–45% per edge). Edges refer to the clockwise-rotated view; leave PC rotation
   on Auto and mirroring off to match these labels. Crop boundaries align to even
   YUV pixels. The phone shows the resulting view size. Stop streaming to change
   settings. Orientation settings are in a collapsed section.
3. Camera intrinsics are sent automatically when streaming starts and refreshed
   during capture; there is no separate send button.
4. In the PC control center, enter the screen resolution, diagonal size, and camera
   position relative to the screen center.

Camera position is measured in centimeters. Positive axes point right, down, and
toward the user. Perform a new position calibration after changing the camera, resolution,
rotation, crop, screen, or camera mounting position. Cropping shifts the optical
center by the removed left/top pixels; it does not re-center or rescale the camera
intrinsics. Rotation is applied once on the PC, to both pixels and intrinsics.

### 4. Run Two-stage Calibration

- **Stage 1 — Position:** 12 screen-spanning lines in three groups. Follow and drag
  the blue dot, including the stationary holds at both ends. No prescribed head
  movements are required. Perimeter lines reach all four corners with an 18 CSS
  pixel margin to keep the target visible; inner lines retain central coverage.
  Eight whole trials train the personal model and four select/check it. These
  selection trials are not an independent final accuracy test. After successful
  training and export checks, the selected candidate replaces the active position
  model; this stage has no separate accuracy gate against the previous model.
- **Stage 2 — Saccades & fixations:** 12 short jumping-dot sequences, 24 jumps and
  approximately 55–60 seconds of target presentation, excluding pauses. Covers
  short, medium and long normalized-screen distances across horizontal, vertical
  and diagonal directions. Checks landing behavior and estimates automatic stability
  parameters from training fixations, accepting them only after validation. The
  position model is not retrained. This stage may be done later.

Each stage starts separately. Space pauses capture; Esc leaving fullscreen also
pauses it. Ending a run retains captured data. Lighting-preset capture has been removed.

Spatial replay samples at up to 120 Hz without duplicating slower camera frames.
Capture duration does not increase at 120 FPS. Training uses 256-frame batches,
validation early stopping and a maximum of 120 epochs. Recordings with measured
source rate of at least 100 FPS receive a larger optimizer-update budget; slower
or unknown rates retain the conservative budget. Denser sampling alone did not
improve the controlled comparison at a fixed training budget. Allowing more
training improved average and tail error on that recording, with small regional
regressions; this is not a cross-user or cross-session accuracy guarantee.

Completion and recapture checks use observed timestamps and stable endpoint time,
rather than requiring 120 FPS sample counts from a 30 FPS camera. Only incomplete
or insufficiently observed trials should require recapture.

Calibration data and trained models are stored under
`%LOCALAPPDATA%\OpenGazeLink`, outside the installation directory.

### 5. Run OpenGazeLink

- **Control Center** handles pairing, configuration, calibration, preview, and
  diagnostics.
- **Background Runtime** loads the saved configuration and starts camera receive,
  Tasks landmarks, current model inference, and shared-memory output. It does
  not start the web control service or video preview.

Both modes use the same inference and post-processing path. The final fused point
shown in preview is the same coordinate published to shared memory. Preview shows
only the fused point. In fullscreen, its coordinate surface fills the viewport;
the status bar overlays it without shifting or shrinking gaze coordinates.

## Current Model and Automatic Stability

Runtime uses `tasks` + `conditioned_video`. Loading old preferences migrates
model/lighting selections to this path and disables the old learned-forecast and
generic-extrapolation branches. Historical model files and research tools are
retained; they are not exposed as runtime choices.

After binocular fusion, event processing uses source timestamps to distinguish
fixation, pursuit, saccade motion, and landing. Fixation smoothing is automatic;
pursuit, head movement and landing follow the current observation without that
stable-state filter. The single **Saccade prediction / 眼跳预测** switch controls
bounded continuation and braking-based landing prediction, not fixation stability.
Time-based confirmation, separate entry/exit thresholds and a landing quiet period
reduce repeated state switching. New saccades are suppressed during head-motion
protection. Target changes are aligned to submitted display events; transition
frames are retained but excluded from stable coordinate labels for at least
700 ms plus available clock uncertainty. This guard is not a measured personal
reaction time, and low-rate video can still miss small saccades.

Stage 2 can refine the noise prior and landing quiet period. Training data estimates
the parameters, validation can reject regressions, and test data never selects
parameters. Saved evidence is bound to the model, camera and screen configuration;
missing or stale evidence falls back to default event detection.

High frame rate does not itself prove lower end-to-end latency or accurate future
eye position. Prediction cannot reliably infer an unobserved destination. The
Performance page distinguishes measured PC processing from cross-clock estimates
and assumed display delay.

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
- Large lighting changes may require another position calibration; lighting
  presets are no longer selectable.
- Advertised camera/encoder combinations may still be rejected by a device at
  startup. Actual FPS depends on capture, ISP, encoder, network and PC throughput.
- TCP/UDP transport is unencrypted. Use it only on a trusted LAN and do not expose it
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

The source checkout includes the Face Landmarker asset. Personal calibration also
requires the separately supplied base eye model described below; installing Python
dependencies alone does not supply that model.

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

Before archiving, the build runs the frozen application's `self-check` command:
base-model loading, one optimizer step, TorchScript export/reload, MediaPipe,
H.264 decoder and a fresh local control server. It uses temporary user data and
loopback ports, without contacting the running application. To repeat it from
an extracted package:

```powershell
./OpenGazeLink.exe self-check --report "$env:TEMP/opengazelink-self-check.json"
```

The JSON report records success or the failing dependency. This automated check
does not measure personal gaze accuracy or replace a live calibration trial.

### Base model assets

The bundled `face_landmarker.task` is the Apache-2.0 MediaPipe
[Face Landmarker model](https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task).
Its pinned SHA-256 is
`64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff`;
the upstream `latest` URL is provenance, not an automatic update source.

The current personal trainer needs a compatible public-data base checkpoint
(`best.pt`) and its matching normalization metadata (`result.json`). They are
release assets, excluded from Git along with personal weights. Put the pair in
`pc-provider/models/public-conditioned/`, or set `OPENGAZELINK_PUBLIC_MODEL_DIR`
to a directory containing both files before training or building. Use a trusted
maintainer-supplied pair; an arbitrary PyTorch checkpoint is not compatible.
The v0.2.0 Release includes `OpenGazeLink-base-model-0.2.0.zip` for source users;
the Windows packages already contain this pair. There is no automatic download.

The bundled base model was trained on MPIIFaceGaze. Its weights and normalization
metadata are distributed under **CC BY-NC-SA 4.0 for non-commercial scientific
use**; they are not covered by the source code's Apache-2.0 license. See
[MODEL_NOTICE](MODEL_NOTICE) for attribution, training provenance and terms.

The Windows build fails with an explicit prerequisite error if the pair is
missing. It no longer takes model files implicitly from a dated experiment
directory. Existing source development retains its legacy local fallback for
compatibility; that directory is neither committed nor used by the release build.

Before distributing a new binary release, supply the base model with its source,
checksum and applicable redistribution terms, and verify calibration from a fresh
user-data directory. Source-code licensing does not establish redistribution
rights for separately trained weights or their source datasets. Personal captures
and adapted models must never be substituted for the base model in a release.

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
node tools/check_control_contract.cjs
node tools/check_spatial_plan.cjs
node tools/check_event_plan.cjs
```

GitHub Actions runs PC unit tests, frontend syntax and deterministic capture-plan
checks on Windows. Tests use synthetic inputs, mocks and temporary directories;
personal recordings and offline research trainers are not required.

Optional browser regressions mock the control API and do not connect to a live
camera or start real training. From `pc-provider`, install Playwright locally:

```powershell
npm install --no-save --package-lock=false playwright@1.55.1
npx playwright install chromium
node tools/check_preview_coordinates.cjs
node tools/check_unified_calibration_ui.cjs
```

They verify fullscreen coordinates, fusion-only display, complete capture,
pause/resume and recapture. `OPENGAZELINK_TEST_PYTHON` can override the Python
executable and `PLAYWRIGHT_CHANNEL=msedge` can select an installed Edge browser.
Screenshots and reports go to ignored `data/ui-checks/` directories.

For Android, run `gradle :app:testDebugUnitTest :app:lintDebug :app:assembleDebug`
with the same JDK/SDK used by the build script. Automated tests do not replace
device checks of sustained 30/120 FPS transport, first-run pairing, both
calibration stages, upgrade data preservation and background output.

## Contributing and Release Scope

Keep changes focused on one observable behavior and record the baseline before
tuning. First check capture timestamps, camera geometry and display-coordinate
mapping; then compare model or filter changes against the same retained input.
Split by whole trial, keep final evaluation separate from epoch selection, and
report regional errors, tail error, latency and state-switching alongside averages.
A regression may be acceptable when its magnitude and the overall benefit are
documented. Failed experiments do not need to become runtime options.

Commit application code and reusable regression tests. Keep personal recordings,
trained personal weights, machine-specific scripts, one-off parameter sweeps,
benchmark outputs and agent working notes local. The tools directory uses an
explicit publication allowlist. Build artifacts belong in GitHub Releases.

Prepare each update on a branch and review it through a pull request. Before
tagging, set matching PC installer/archive versions, include the required
base-model provenance and complete a clean-install/upgrade smoke test on both
capture routes. Prefer a
single squash merge for a consolidated release branch so `main` records the
delivered behavior rather than local experimental iterations.

## Repository Layout

```text
OpenGazeLink/
  phone-app/       Android Camera2 capture and TCP/UDP sender
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
TCP/UDP. Use OpenGazeLink only on a trusted private network, and do not expose UDP
ports `5006`/`5007` or TCP port `5007` to the internet.

## License and Third-Party Components

OpenGazeLink source code is licensed under the
[Apache License 2.0](LICENSE).

MediaPipe, the Face Landmarker model, OpenCV, NumPy, PyTorch, AndroidX, Kotlin, and
other third-party components remain subject to their respective open-source
licenses.
