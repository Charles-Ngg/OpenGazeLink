# OpenGazeLink for iOS

An iOS sender for the existing OpenGazeLink PC provider. It reimplements the
Android `phone-app` protocol on AVFoundation + VideoToolbox, so the Windows side
needs **no changes**: the same discovery port, the same UDP JPEG framing, the
same TCP H.264 framing, the same intrinsics JSON, and the same clock probe.

Target device: **iPhone 16 Pro Max** on iOS 17 or later. The app runs on any
iOS 17+ iPhone, but the capture options it offers come from what the device
actually advertises, never from a hard-coded list.

---

## 1. Why 1280 × 720 for High-speed mode

`pc-provider/opengazelink_pc/h264_stream.py:205` rejects any H.264 packet whose
header geometry is not exactly `(1280, 720)`:

```python
if magic != b'AVC1' or not (0 < size <= MAX_PACKET_BYTES) or (width, height) != (1280, 720):
    raise ValueError('Invalid H.264 packet header')
```

That check runs **before** decoding, so a 1080p H.264 stream is not degraded — it
is dropped entirely. High-speed mode therefore offers exactly the geometry the
PC accepts, and lists the rest as unavailable with the reason shown in the UI.

This is not only a compatibility compromise, it is the right target anyway:

| | value |
|---|---|
| Gaze model input | a 64 × 36 px eye patch covering a 4.2 × 2.4 cm plane (`normalized_eye.py:25`) |
| Density the model needs | 15.2 px/cm |
| 720p at 60 cm from a 76° front camera | 13.6 px/cm |
| 720p at 40 cm | 20.4 px/cm |
| 1080p at 60 cm | 20.4 px/cm |

So at typical desk distance 720p already meets or exceeds what the model
consumes, and 1080p only adds oversampling. Meanwhile the iPhone's H.264
hardware encoder is rated far below 1080p120 (≈250 Mpix/s), so choosing 1080p
would force the frame rate down to 60 FPS — trading temporal sampling of
saccades (20–80 ms events) for resolution the model cannot use. **720p120 is the
target; 1080p is not worth a PC-side change.**

1080p and 4K remain fully available in **Camera (JPEG/UDP) mode**, where the PC
path is dimension agnostic and higher resolution is genuinely useful for
calibration frames and detail inspection.

If you do want 1080p over H.264, the PC change is one line plus a test:

```python
# h264_stream.py:205
if magic != b'AVC1' or not (0 < size <= MAX_PACKET_BYTES) \
        or not (0 < width <= 4096 and 0 < height <= 4096 and width % 2 == 0 and height % 2 == 0):
    raise ValueError('Invalid H.264 packet header')
```

and then widen `WireFormat.acceptsH264` plus
`ios-app/Tools/check_wire_contract.py`'s geometry check to match.

---

## 2. Build

The Xcode project is generated, not committed: `*.xcodeproj` is in
`.gitignore`, and CI regenerates it from `project.yml`. This mirrors the
`iphoneclaw` setup and keeps the repository editable from Linux.

### Requirements

| | |
|---|---|
| Xcode | 16.2 or newer |
| iOS deployment target | **17.0** (needed for `AVCaptureDevice.RotationCoordinator`) |
| XcodeGen | `brew install xcodegen` |
| Device | any iOS 17+ iPhone; optimised for iPhone 16 Pro Max |

### Local build

```bash
cd ios-app
xcodegen generate
open OpenGazeLink.xcodeproj
```

Then set your signing team in the target's Signing & Capabilities tab, select
your iPhone, and run. Or from the command line:

```bash
xcodegen generate
xcodebuild test -project OpenGazeLink.xcodeproj -scheme OpenGazeLink \
  -destination 'platform=iOS Simulator,name=iPhone 16'
xcodebuild build -project OpenGazeLink.xcodeproj -scheme OpenGazeLink \
  -sdk iphoneos -configuration Release
```

### Unsigned IPA from GitHub Actions

`.github/workflows/ios-build.yml` runs on a macOS runner, generates the project,
runs the unit tests, builds unsigned Release and Debug `iphoneos` products, and
publishes both IPAs to a GitHub Release tagged `ios-<ref>-<run id>`. It only
triggers when `ios-app/**` changes.

The IPAs are intentionally unsigned (`CODE_SIGNING_ALLOWED=NO`,
`CODE_SIGNING_REQUIRED=NO`), so no Apple Developer certificate is needed for CI.
Sign or load them with your usual sideloading flow (LiveContainer / SideStore /
Sideloadly) as you do for `iphoneclaw`.

> CI success proves the source compiles and packages. It does **not** prove the
> camera pipeline works — that needs a physical device. See §6.

A Linux-only, dependency-free contract check also runs in the same workflow and
can be run locally at any time:

```bash
python3 ios-app/Tools/check_wire_contract.py .
```

It reads the PC's `struct.Struct` format strings out of `pc-provider`, derives
the golden byte sequences, and asserts the Swift constants and the unit-test
vectors still match. It catches Swift/PC protocol drift without a Mac.

---

## 3. Permissions

`Info.plist` declares both, and iOS prompts on first use:

| key | why |
|---|---|
| `NSCameraUsageDescription` | capture frames |
| `NSLocalNetworkUsageDescription` | discovery and streaming to the PC on the LAN |

There is **no** `NSBonjourServices` entry: the PC provider has no mDNS responder,
so Bonjour would need a PC change and is deliberately not used.

### Broadcast discovery and the multicast entitlement

iOS 14+ refuses UDP broadcast (`255.255.255.255` and subnet-directed broadcast)
unless the app carries Apple's **Multicast Networking** entitlement
(`com.apple.developer.networking.multicast`). `Network.framework` cannot
broadcast at all, so the discovery socket is a BSD one.

`OpenGazeLink.entitlements` therefore ships with the key **commented out**, and
`PairingService` sends the identical JSON payload three ways in increasing cost:

1. **Unicast to the last paired PC address.** This is the normal case after the
   first pairing and needs nothing special.
2. **Broadcast** to `255.255.255.255` and to each interface's directed broadcast.
   Free with the entitlement; refused with `EPERM` without it. The refusal is
   surfaced in the UI, not swallowed.
3. **A paced unicast sweep** of the phone's own /24 (32 addresses per 10 ms
   burst, phone's own addresses excluded). This needs no entitlement and is why
   discovery still works on a free-account sideload.

A free Apple account cannot sign the multicast entitlement, so on a free account
step 2 will always fail and **manual PC address entry is the primary path**. Enter
the PC's LAN address and port and press Start.

To enable broadcast with a paid account, uncomment the key in
`OpenGazeLink.entitlements`, make sure your provisioning profile carries the
capability, and re-sign. The CI job fails on purpose if the key is left enabled,
so this stays a deliberate action.

---

## 4. Protocol mapping

Everything below is verified against the PC source by
`ios-app/Tools/check_wire_contract.py` and by
`OpenGazeLinkTests/WireFormatTests.swift`.

| direction | port | framing | Swift |
|---|---|---|---|
| discovery | UDP 5006 | `EYETRACING_DISCOVERY_V1` JSON | `Discovery/PairingService.swift` |
| JPEG frames | UDP `data_port` | `<IHHIHHHHBBQQI`, 42 B header + 1400 B chunks | `Transport/UdpSender.swift` |
| intrinsics | UDP `data_port` | `<IHHI`, 12 B envelope + JSON | `Capture/CameraIntrinsics.swift` |
| clock probe reply | UDP `data_port` | `<IHHQQQ`, 32 B, kind 2 | `Transport/UdpSender.swift` |
| H.264 | TCP `data_port` | `<4sIIHHQQQI`, 44 B header + Annex-B | `Transport/AvcTcpSender.swift` |

`data_port` comes from the PC's offer and is used for **both** UDP and TCP — the
PC binds its TCP listener to the same port number as its UDP socket
(`h264_stream.py:118`). Default 5007.

### High-speed mode (H.264 over TCP)

`AVCaptureVideoDataOutput` → `CVPixelBuffer` → `VTCompressionSession`, matching
`AvcEncoder.kt`:

| setting | Android | iOS |
|---|---|---|
| realtime | `KEY_LATENCY = 0` | `kVTCompressionPropertyKey_RealTime = true` |
| no B-frames | `KEY_MAX_B_FRAMES = 0` | `AllowFrameReordering = false` + `MaxFrameDelayCount = 1` |
| profile | `AVCProfileBaseline` | `kVTProfileLevel_H264_Baseline_AutoLevel` + CAVLC |
| keyframe interval | `KEY_I_FRAME_INTERVAL = 1` | `MaxKeyFrameInterval = fps`, `MaxKeyFrameIntervalDuration = 1` |
| bitrate | `w*h*fps/8`, clamped 2–80 Mbps | identical expression and clamp |
| socket | `TCP_NODELAY`, `SO_SNDBUF` 32 KiB | identical |
| queue | 16 packets, drop oldest | identical |

**The one conversion that matters:** VideoToolbox returns **AVCC** access units
with SPS/PPS in the format description, while the PC decodes with
`av.CodecContext.create('h264', 'r')` and no extradata, so it parses **Annex-B**
only. `Capture/AnnexB.swift` rewrites the length prefixes to 4-byte start codes,
and `Capture/H264SampleBridge.swift` extracts SPS/PPS and emits them as a
`flags & 2` packet — which the PC stores and prepends to every keyframe packet.

Keyframe detection reads the IDR NAL type out of the bitstream rather than
`kCMSampleAttachmentKey_NotSync`, which avoids CoreMedia attachment bridging and
is unit-testable without an encoder.

`flags & 8` (partial access unit) is never set: the PC raises on it, so a
malformed access unit is dropped instead.

### Camera mode (JPEG over UDP)

Crop-then-compress, never resize — the PC rejects a JPEG whose decoded size does
not equal the packet header (`camera.py:80`). The crop is computed by the exact
port of `CropPercent.pixels` in `Model/CropPercent.swift`, including the
even-pixel rounding `(round(size * pct / 100) and -2)` and the inverse mapping
from the rotated view into unrotated stream coordinates.

iOS has no software NV21 encoder, so the retained CoreImage extent is rendered
straight to JPEG with `CIContext.jpegRepresentation`. That is a GPU round trip;
Camera mode runs at 60 FPS or less and is not the latency-critical route, and the
high-speed route never touches CoreImage.

iOS delivers biplanar **NV12** (UV order), which is not byte-compatible with the
PC's NV21 reader, so `format = 2` (JPEG) is the only UDP format used.

### Rotation and intrinsics

The phone **never rotates its own buffers**. The capture connection is pinned to
`videoRotationAngle = 0` and the clockwise angle is reported as `frameRotation`,
so the PC applies rotation exactly once, to both pixels and intrinsics — the same
division of labour as the Android app. `AVCaptureDevice.RotationCoordinator`
provides the angle on iOS 17+, and the UI exposes a manual 0/90/180/270 override.

Intrinsics are sent at stream start and refreshed every 2 s, like Android.

**This is the one place iOS cannot mirror Android exactly.** AVFoundation exposes
no per-format factory calibration for the video path, so:

| field | iOS source |
|---|---|
| `fx`, `fy` | derived from `activeFormat.videoFieldOfView`, scaled by digital zoom and by `streamWidth / formatWidth` |
| `cx`, `cy` | frame centre (no calibrated optical axis is available) |
| `distortion` | empty (no lookup table on the video path) |
| `source` | `ios_avfoundation_video_field_of_view_horizontal` or `..._diagonal` |

Apple documents `videoFieldOfView` only as "the format's field of view, in
degrees" without naming the axis, and the horizontal and diagonal readings differ
by ~15% at 16:9 — a linear error in `fx`. The UI therefore exposes a
**Field-of-view reference axis** toggle (default: horizontal). **Verify it on the
device**; if the gaze projection looks scaled or rotated, switch the axis before
recalibrating.

`AVCameraCalibrationData` (via `AVCapturePhotoOutput` with
`isCameraCalibrationDataDeliveryEnabled`) is the only exact intrinsics source on
iOS, but it requires a still capture and is not available for the front camera in
video mode. It is a documented future improvement, not implemented here.

The PC validates what it receives (`camera.py:505`): focal lengths within
0.1–10 × the frame dimension, a principal point inside a plausible window, and a
`softwareCrop` that adds up. `CameraIntrinsicsTests` re-implements that validator
and asserts the payload passes it for every offered geometry, every rotation, and
both reference axes.

### Clock domain

Android sends `SystemClock.elapsedRealtimeNanos()` in `sensor_time_ns`,
`encoded_time_ns`, `phone_send_time_ns` and in clock-probe replies, so all of them
share one clock. iOS uses `mach_absolute_time()` converted through the mach
timebase, because `AVCaptureVideoDataOutput` presentation timestamps are in the
host clock — the same domain. `mach_continuous_time` is deliberately **not**
mixed in; doing so would silently break the PC's clock-offset estimate.

Difference, documented rather than hidden: `mach_absolute_time` does not advance
while the device sleeps, whereas Android's `elapsedRealtimeNanos` does. Streaming
keeps the display awake (`isIdleTimerDisabled`), so the two behave identically for
the duration of a session.

---

## 5. Performance monitor

Phone-side only, mirroring the Android labels. Byte accounting matches Android
exactly: header bytes count as sent (`data + 44` for AVC, `payload + chunks * 42`
for JPEG), and codec-configuration packets do not increment the sent-frame count.

| metric | source |
|---|---|
| capture FPS | `AVCaptureVideoDataOutput` callbacks |
| sent FPS | packets actually written |
| bitrate | wire bytes × 8 / elapsed |
| capture → encoded | host-clock delta between the sample buffer PTS and the encoder output callback |
| exposure | `AVCaptureDevice.exposureDuration`, sampled per frame |

Exposure is a limitation: iOS does not attach a per-frame exposure duration to
video data output, so this is the device's current setting, not the value applied
to that specific frame.

Network transit, PC decode and display latency are absent because they cannot be
observed from the phone; the PC control centre reports those.

---

## 6. What is verified where

| check | where | status |
|---|---|---|
| Swift/PC byte layout, magic numbers, struct sizes, geometry gate | `Tools/check_wire_contract.py` | **passes**: 31 checks, locally and in CI |
| Swift unit tests (wire vectors, crop math, option selection, Annex-B, intrinsics validator, rotation, subnet arithmetic) | `OpenGazeLinkTests` | **passes**: 85 tests, 0 failures, on the CI simulator |
| source compiles for `iphoneos` arm64, deployment target 17.0 | CI macOS runner, Xcode 16.2 | **passes** |
| unsigned Release and Debug IPAs package and publish | CI macOS runner | **passes** |
| camera formats actually advertised by the device | **device only** | must be measured |
| whether the front camera exposes 1080p120 through `videoSupportedFrameRateRanges` | **device only** | must be measured |
| H.264 hardware encoder throughput at the selected rate | **device only** | must be measured |
| `videoFieldOfView` reference axis | **device only** | must be verified |
| end-to-end pairing with the Windows provider | **device + PC** | must be verified |
| gaze accuracy after recalibration | **device + PC** | must be verified |

Compiling and passing unit tests proves the protocol and the capture-option logic
are correct. It says nothing about whether the camera pipeline runs, which is
what the device checklist below is for.

### Device test checklist

1. Pair: start the PC control centre, press **Find PC**. If the sweep finds
   nothing, type the PC's LAN address and port.
2. Confirm the phone in the PC control centre so the PC allowlists its address.
3. High-speed mode, front camera, 1280 × 720, 120 FPS. Confirm the PC shows a
   live image and that its reported rotation matches what you see.
4. If 120 FPS is not offered, the format does not advertise it; 60 FPS is the
   fallback and needs no configuration.
5. Check the Performance monitor: capture FPS should track the target, and
   capture → encoded should stay in the low single-digit milliseconds.
6. Switch to Camera mode, set a crop, confirm the PC reports the expected
   received view size and that no resize is applied.
7. Recalibrate on the PC after any change to camera, resolution, rotation, crop
   or mounting position — the same rule as the Android app.

---

## 7. Known limitations

- **Broadcast discovery needs a paid Apple account** for the multicast
  entitlement. Unicast sweep and manual entry work regardless (§3).
- **H.264 is 1280 × 720 only** because the PC rejects everything else (§1).
- **1080p120 H.264 is not achievable on iPhone hardware.** 1080p and 4K are
  available in Camera/JPEG mode instead.
- **HEVC is not supported.** The wire format has no codec field, so HEVC would be
  a protocol extension (e.g. a new `AVC_HEADER.flags` bit) plus a PC decoder
  change. Not attempted here.
- **Intrinsics are derived, not calibrated** (§4).
- **The preview is not mirrored**, matching the wire frame. The PC mirrors if its
  own config says to.
- **The app must stay in the foreground**, like the Android app. The screen stays
  awake while streaming.
- **Camera mode JPEG encoding is a GPU round trip**, not a hardware JPEG block.
- **The Xcode project is generated.** Opening `OpenGazeLink.xcodeproj` requires
  running `xcodegen generate` first.

---

## 8. Layout

```text
ios-app/
  project.yml                     XcodeGen spec (the tracked source of truth)
  README.md
  Tools/check_wire_contract.py    Swift <-> PC protocol contract check
  OpenGazeLink/
    Info.plist                    camera + local network usage strings
    OpenGazeLink.entitlements     multicast entitlement, commented out
    App/                          app entry point and observable model
    Core/                         WireFormat, MonotonicClock, L10n
    Model/                        capture options, crop math, metrics
    Capture/                      AVFoundation session, Annex-B, VideoToolbox, JPEG
    Transport/                    BSD sockets: UDP JPEG + clock reply, TCP AVC
    Discovery/                    interface enumeration, pairing, persistence
    UI/                           SwiftUI views and the preview layer
  OpenGazeLinkTests/              unit tests, runnable on the simulator
```
