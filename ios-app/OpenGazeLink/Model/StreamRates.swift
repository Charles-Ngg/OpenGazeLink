import Foundation

/// Live phone-side metrics, mirroring Android's `StreamRates`.
struct StreamRates: Equatable {
    var captureFPS: Double = 0
    var sentFPS: Double = 0
    var megabitsPerSecond: Double = 0
    /// Capture → encoded packet age. Nil until a measurement is available.
    var encodingAgeMs: Double?
    /// Current device exposure duration. Nil until a measurement is available.
    var exposureMs: Double?

    static let empty = StreamRates()
}

/// Failure classification, mirroring Android's `StreamFailure` so the UI can
/// pick the same message for the same cause.
enum StreamFailure {
    case connection
    case camera
    case combination
    case intrinsics

    var messageKey: L10n.Key {
        switch self {
        case .connection: return .connectionFailed
        case .camera: return .cameraFailed
        case .combination: return .combinationFailed
        case .intrinsics: return .intrinsicsFailed
        }
    }
}

/// Accumulates per-interval counters and emits one `StreamRates` per second.
///
/// Byte accounting matches Android exactly: the packet header is counted as
/// sent bytes (`data.size + 44` for AVC, `bytes + chunks * 42` for JPEG), and
/// codec-configuration packets do not increment the sent-frame counter.
struct StreamStatistics {
    private var captured: UInt64 = 0
    private var sent: UInt64 = 0
    private var sentBytes: UInt64 = 0
    private var lastCaptured: UInt64 = 0
    private var lastSent: UInt64 = 0
    private var lastSentBytes: UInt64 = 0
    private var lastSampleTimeNs: UInt64 = 0

    private var encodingAgeTotalMs: Double = 0
    private var encodingAgeCount: Int = 0
    private var exposureTotalMs: Double = 0
    private var exposureCount: Int = 0

    mutating func recordCapture() {
        captured &+= 1
    }

    mutating func recordSent(bytes: Int, countsAsFrame: Bool) {
        sentBytes &+= UInt64(max(0, bytes))
        if countsAsFrame { sent &+= 1 }
    }

    mutating func recordEncodingAge(ms: Double) {
        guard ms.isFinite, ms >= 0 else { return }
        encodingAgeTotalMs += ms
        encodingAgeCount += 1
    }

    mutating func recordExposure(ms: Double) {
        guard ms.isFinite, ms >= 0 else { return }
        exposureTotalMs += ms
        exposureCount += 1
    }

    /// Resets the interval and returns the rates for the elapsed period.
    mutating func sample(nowNs: UInt64) -> StreamRates {
        guard lastSampleTimeNs != 0 else {
            lastSampleTimeNs = nowNs
            lastCaptured = captured
            lastSent = sent
            lastSentBytes = sentBytes
            return .empty
        }
        let elapsedNs = nowNs >= lastSampleTimeNs ? nowNs - lastSampleTimeNs : 0
        let elapsedSeconds = max(Double(elapsedNs) / 1_000_000_000.0, 0.001)

        let rates = StreamRates(
            captureFPS: Double(captured - lastCaptured) / elapsedSeconds,
            sentFPS: Double(sent - lastSent) / elapsedSeconds,
            megabitsPerSecond: Double(sentBytes - lastSentBytes) * 8.0 / elapsedSeconds / 1_000_000.0,
            encodingAgeMs: encodingAgeCount > 0 ? encodingAgeTotalMs / Double(encodingAgeCount) : nil,
            exposureMs: exposureCount > 0 ? exposureTotalMs / Double(exposureCount) : nil
        )

        lastSampleTimeNs = nowNs
        lastCaptured = captured
        lastSent = sent
        lastSentBytes = sentBytes
        encodingAgeTotalMs = 0
        encodingAgeCount = 0
        exposureTotalMs = 0
        exposureCount = 0
        return rates
    }

    mutating func reset() {
        self = StreamStatistics()
    }
}
