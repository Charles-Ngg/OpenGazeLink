import AVFoundation
import Foundation

/// Resolves the clockwise rotation the PC should apply to the received frame.
///
/// Android computes `(sensorOrientation + displayRotation) % 360` for the front
/// camera and `(sensorOrientation - displayRotation + 360) % 360` for the back
/// camera, with a manual override. iOS 17 exposes the equivalent directly:
/// `AVCaptureDevice.RotationCoordinator.videoRotationAngleForHorizonLevelCapture`
/// is already the clockwise angle that makes the capture horizon-level, and it
/// accounts for lens position and device orientation.
///
/// The phone never rotates its own buffers — the capture connection is pinned
/// to `videoRotationAngle = 0` so the wire always carries sensor-native frames
/// and the PC applies rotation once to both pixels and intrinsics, exactly as
/// the Android app does.
enum FrameRotation {

    /// Rotation values the PC accepts; anything else falls back to 270 in
    /// `camera.py:228`.
    static let supported: [Int] = [0, 90, 180, 270]

    /// Snaps a CoreVideo/CoreAnimation angle to the nearest supported value.
    static func normalize(degrees: Double) -> Int {
        guard degrees.isFinite else { return 0 }
        var value = Int((degrees / 90.0).rounded()) * 90
        value %= 360
        if value < 0 { value += 360 }
        return supported.contains(value) ? value : 0
    }

    /// Clockwise rotation for the given device, or nil when the coordinator
    /// cannot be created.
    static func automatic(device: AVCaptureDevice, previewLayer: AVCaptureVideoPreviewLayer?) -> Int? {
        let coordinator = AVCaptureDevice.RotationCoordinator(device: device, previewLayer: previewLayer)
        let angle = coordinator.videoRotationAngleForHorizonLevelCapture
        guard angle.isFinite else { return nil }
        return normalize(degrees: Double(angle))
    }
}
