import Foundation

/// The two capture routes, matching Android's `SessionMode`.
///
/// - `highSpeed`: full frame, hardware H.264, TCP. The PC decodes every
///   reference frame and never resizes.
/// - `camera`: optional pre-encode crop, software JPEG, UDP.
enum SessionMode: String, CaseIterable, Identifiable {
    case highSpeed = "HIGH_SPEED"
    case camera = "CAMERA"

    var id: String { rawValue }

    /// Android labels these "High-speed session" and "Camera".
    var titleKey: L10n.Key {
        self == .highSpeed ? .modeHighSpeed : .modeCamera
    }
}

/// A single advertised size/rate pair.
///
/// Port of `CaptureOption.kt`. The pair is never synthesised from a cross
/// product of unrelated size and rate lists, because most cameras only
/// advertise a subset of the combinations.
struct CaptureResolution: Hashable, Comparable {
    let width: Int
    let height: Int

    var pixels: Int { width * height }
    var label: String { "\(width) × \(height)" }

    static func < (lhs: CaptureResolution, rhs: CaptureResolution) -> Bool {
        if lhs.pixels != rhs.pixels { return lhs.pixels > rhs.pixels }
        if lhs.width != rhs.width { return lhs.width > rhs.width }
        return lhs.height > rhs.height
    }
}

struct CaptureOption: Hashable, Identifiable {
    let cameraID: String
    let mode: SessionMode
    let width: Int
    let height: Int
    /// Upper bound of the advertised `AVFrameRateRange`; the rate we pin.
    let fps: Int
    /// Lower bound of the same range, kept for display only.
    let fpsLower: Int

    var id: String { key }
    var key: String { "\(cameraID):\(mode.rawValue):\(width):\(height):\(fps)" }
    var resolution: CaptureResolution { CaptureResolution(width: width, height: height) }
    var label: String { "\(resolution.label) · \(fps) FPS" }

    /// True when the PC's H.264 receiver will accept this geometry.
    /// Only meaningful for `highSpeed`; the UDP JPEG path is size-agnostic.
    var isPCH264Compatible: Bool { WireFormat.acceptsH264(width: width, height: height) }
}

/// Port of `CaptureOptions` in `CaptureOption.kt`.
///
/// The ordering, tie-breaking and "never invent a cross product" rules are kept
/// identical so that a phone moved between platforms resolves the same saved
/// selection to the same pair.
enum CaptureOptions {

    /// Preferred rate per mode, used for ordering and for the distance metric.
    static func preferredFPS(for mode: SessionMode) -> Int {
        mode == .highSpeed ? 120 : 30
    }

    /// Port of `CaptureOptions.ordered`.
    ///
    /// Differences from Android, deliberate and documented:
    /// - Android required `fps >= 120 && fpsLower == fps` for high speed because
    ///   Camera2 constrained-high-speed sessions only expose fixed ranges, and
    ///   it verified a hardware H.264 encoder for the size. iOS exposes ordinary
    ///   `AVFrameRateRange`s, so the high-speed filter is `fps >= 60` (keeping
    ///   the 60 FPS fallback the user asked for) and the hardware check is
    ///   replaced by `WireFormat.acceptsH264`, which is what actually decides
    ///   whether the PC will decode the stream.
    static func ordered(_ options: [CaptureOption]) -> [CaptureOption] {
        let filtered = options.filter { option in
            guard option.width > 0, option.height > 0,
                  option.width % 2 == 0, option.height % 2 == 0,
                  option.fps > 0,
                  option.fpsLower >= 1, option.fpsLower <= option.fps
            else { return false }
            switch option.mode {
            case .highSpeed:
                return option.fps >= 60 && option.isPCH264Compatible
            case .camera:
                // Android caps the JPEG route at 60 FPS; UDP chunking a
                // full-resolution frame faster than that only adds loss.
                return option.fps <= 60
            }
        }

        // Each mode is sorted separately. The "preferred rate first" rule is
        // only a valid ordering when `preferred` is fixed for the whole list,
        // so mixing modes in one sort would give `sorted(by:)` an inconsistent
        // comparator. Order across modes is irrelevant: every caller filters to
        // a single mode first.
        func sort(_ mode: SessionMode) -> [CaptureOption] {
            let preferred = preferredFPS(for: mode)
            return filtered.filter { $0.mode == mode }.sorted { lhs, rhs in
                if (lhs.fps == preferred) != (rhs.fps == preferred) {
                    return lhs.fps == preferred
                }
                let lhsAreaDistance = abs(lhs.width * lhs.height - 1280 * 720)
                let rhsAreaDistance = abs(rhs.width * rhs.height - 1280 * 720)
                if lhsAreaDistance != rhsAreaDistance { return lhsAreaDistance < rhsAreaDistance }
                if abs(lhs.fps - preferred) != abs(rhs.fps - preferred) {
                    return abs(lhs.fps - preferred) < abs(rhs.fps - preferred)
                }
                if (lhs.fps - lhs.fpsLower) != (rhs.fps - rhs.fpsLower) {
                    return (lhs.fps - lhs.fpsLower) < (rhs.fps - rhs.fpsLower)
                }
                if lhs.width != rhs.width { return lhs.width < rhs.width }
                return lhs.height < rhs.height
            }
        }

        let sorted = sort(.highSpeed) + sort(.camera)
        var seen = Set<String>()
        return sorted.filter { seen.insert($0.key).inserted }
    }

    /// Port of `CaptureOptions.selected`.
    static func selected(_ options: [CaptureOption], savedKey: String?) -> CaptureOption? {
        if let savedKey, let match = options.first(where: { $0.key == savedKey }) { return match }
        return options.first
    }

    /// Port of `CaptureOptions.resolutions` — distinct values, largest first.
    static func resolutions(_ options: [CaptureOption]) -> [CaptureResolution] {
        var seen = Set<CaptureResolution>()
        let distinct = options.compactMap { seen.insert($0.resolution).inserted ? $0.resolution : nil }
        return distinct.sorted()
    }

    /// Port of `CaptureOptions.frameRates` — distinct values, fastest first.
    static func frameRates(_ options: [CaptureOption]) -> [Int] {
        Array(Set(options.map(\.fps))).sorted(by: >)
    }

    /// Port of `CaptureOptions.selectResolution`: keep the frame rate when the
    /// new size offers it, otherwise take the nearest rate and prefer the lower
    /// one on a tie so a size change never silently doubles the load.
    static func selectResolution(
        _ options: [CaptureOption],
        resolution: CaptureResolution,
        preferred: CaptureOption?
    ) -> CaptureOption? {
        let matching = options.filter { $0.resolution == resolution }
        guard let preferred else { return matching.first }
        return matching.min { lhs, rhs in
            let lhsDistance = abs(lhs.fps - preferred.fps)
            let rhsDistance = abs(rhs.fps - preferred.fps)
            if lhsDistance != rhsDistance { return lhsDistance < rhsDistance }
            return lhs.fps < rhs.fps
        }
    }

    /// Port of `CaptureOptions.selectFrameRate`: keep the resolution when it
    /// offers the new rate, otherwise take the nearest size.
    static func selectFrameRate(
        _ options: [CaptureOption],
        fps: Int,
        preferred: CaptureOption?
    ) -> CaptureOption? {
        let matching = options.filter { $0.fps == fps }
        guard let preferred else { return matching.first }
        return matching.min { lhs, rhs in
            let lhsSame = lhs.resolution == preferred.resolution
            let rhsSame = rhs.resolution == preferred.resolution
            if lhsSame != rhsSame { return lhsSame }
            let lhsPixels = abs(lhs.resolution.pixels - preferred.resolution.pixels)
            let rhsPixels = abs(rhs.resolution.pixels - preferred.resolution.pixels)
            if lhsPixels != rhsPixels { return lhsPixels < rhsPixels }
            let lhsDistance = abs(lhs.width - preferred.width) + abs(lhs.height - preferred.height)
            let rhsDistance = abs(rhs.width - preferred.width) + abs(rhs.height - preferred.height)
            if lhsDistance != rhsDistance { return lhsDistance < rhsDistance }
            if lhs.resolution.pixels != rhs.resolution.pixels {
                return lhs.resolution.pixels < rhs.resolution.pixels
            }
            if lhs.width != rhs.width { return lhs.width < rhs.width }
            return lhs.height < rhs.height
        }
    }
}
