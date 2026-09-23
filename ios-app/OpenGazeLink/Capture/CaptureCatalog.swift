import AVFoundation
import CoreMedia
import Foundation

/// One camera plus every size/rate pair it advertises.
///
/// `bindings` carries the `AVCaptureDevice.Format` and the exact
/// `CMTime` frame duration for each option key. They live outside
/// `CaptureOption` so that type stays a plain `Hashable` value that unit tests
/// can build without AVFoundation.
struct CameraChoice: Identifiable {
    let id: String
    let device: AVCaptureDevice
    let displayName: String
    let position: AVCaptureDevice.Position
    let options: [CaptureOption]
    let bindings: [String: CaptureFormatBinding]

    var isFront: Bool { position == .front }
}

struct CaptureFormatBinding {
    let format: AVCaptureDevice.Format
    let frameDuration: CMTime
}

/// Enumerates AVFoundation capture capabilities, mirroring `CameraCatalog.kt`.
///
/// Android reads `StreamConfigurationMap` and, for high-speed mode, additionally
/// proves that a hardware H.264 encoder exists for the size. iOS has no
/// equivalent "does an encoder exist for this size" query, and the real gate is
/// the PC's receiver, so `CaptureOptions.ordered` substitutes
/// `WireFormat.acceptsH264`. That keeps the guarantee the user actually cares
/// about: an option offered in High-speed mode is one the PC will decode.
enum CaptureCatalog {

    /// Physical cameras only. Virtual devices (`builtInTripleCamera`,
    /// `builtInDualWideCamera`) switch between physical lenses automatically,
    /// which would change focal length — and therefore intrinsics — mid-stream.
    static let deviceTypes: [AVCaptureDevice.DeviceType] = [
        .builtInWideAngleCamera,
        .builtInUltraWideCamera,
        .builtInTelephotoCamera,
        .builtInTrueDepthCamera,
    ]

    /// Rates worth offering inside a wider advertised range.
    private static let commonFrameRates: [Double] = [240, 120, 60, 30, 24, 15]

    static func read() -> [CameraChoice] {
        let discovery = AVCaptureDevice.DiscoverySession(
            deviceTypes: deviceTypes,
            mediaType: .video,
            position: .unspecified
        )
        let choices = discovery.devices.compactMap { device -> CameraChoice? in
            guard let choice = read(device: device) else { return nil }
            return choice
        }
        // Front camera first, then by name — the Android ordering.
        return choices.sorted { lhs, rhs in
            if lhs.isFront != rhs.isFront { return lhs.isFront }
            return lhs.displayName < rhs.displayName
        }
    }

    static func read(device: AVCaptureDevice) -> CameraChoice? {
        var options: [CaptureOption] = []
        var bindings: [String: CaptureFormatBinding] = [:]

        for format in device.formats {
            let dimensions = CMVideoFormatDescriptionGetDimensions(format.formatDescription)
            let width = Int(dimensions.width)
            let height = Int(dimensions.height)
            guard width > 0, height > 0, width % 2 == 0, height % 2 == 0 else { continue }
            guard !format.videoSupportedFrameRateRanges.isEmpty else { continue }

            for range in format.videoSupportedFrameRateRanges {
                for rate in candidateFrameRates(for: range) {
                    let duration = frameDuration(for: rate, in: range)
                    // Both routes are described by the same advertised pair; the
                    // mode-specific filters in `CaptureOptions.ordered` decide
                    // which of them each mode actually offers.
                    for mode in SessionMode.allCases {
                        let option = CaptureOption(
                            cameraID: device.uniqueID,
                            mode: mode,
                            width: width,
                            height: height,
                            fps: rate,
                            fpsLower: max(1, Int(range.minFrameRate.rounded()))
                        )
                        options.append(option)
                        // First format wins for a given size/rate pair. Formats
                        // sharing a size and rate are equivalent for capture
                        // purposes, and `device.formats` lists the default
                        // format first.
                        if bindings[option.key] == nil {
                            bindings[option.key] = CaptureFormatBinding(format: format, frameDuration: duration)
                        }
                    }
                }
            }
        }

        let ordered = CaptureOptions.ordered(options)
        guard !ordered.isEmpty else { return nil }
        return CameraChoice(
            id: device.uniqueID,
            device: device,
            displayName: device.localizedName,
            position: device.position,
            options: ordered,
            bindings: bindings
        )
    }

    /// Discrete rates to offer inside one advertised range.
    ///
    /// Every returned value lies inside `[minFrameRate, maxFrameRate]`, so this
    /// never invents a rate the format did not advertise — the same guarantee
    /// Android gets from enumerating discrete `CONTROL_AE_AVAILABLE_TARGET_FPS_RANGES`
    /// pairs instead of crossing a size list with a rate list.
    static func candidateFrameRates(for range: AVFrameRateRange) -> [Int] {
        let lower = max(1.0, range.minFrameRate)
        let upper = range.maxFrameRate
        guard upper.isFinite, upper >= lower else { return [] }

        var rates = Set<Int>()
        let roundedUpper = upper.rounded()
        // 59.94 must stay 60 rather than becoming 59: the rate is a label, and
        // the exact duration comes from the range itself.
        rates.insert(abs(upper - roundedUpper) < 0.5 ? Int(roundedUpper) : Int(upper.rounded(.down)))
        for common in commonFrameRates where common >= lower - 0.001 && common <= upper + 0.001 {
            rates.insert(Int(common))
        }
        return rates.filter { $0 > 0 }.sorted(by: >)
    }

    /// Exact duration for a rate.
    ///
    /// For the range's own maximum the advertised `minFrameDuration` is used
    /// verbatim, so a 59.94 Hz format is never asked for exactly 1/60 s.
    static func frameDuration(for rate: Int, in range: AVFrameRateRange) -> CMTime {
        let upper = range.maxFrameRate
        if upper.isFinite, abs(upper - Double(rate)) < 0.5, range.minFrameDuration.isValid {
            return range.minFrameDuration
        }
        return CMTime(value: 1, timescale: CMTimeScale(max(1, rate)))
    }
}
