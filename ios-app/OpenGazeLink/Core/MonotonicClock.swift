import CoreMedia
import Foundation
import QuartzCore

/// One nanosecond clock for every timestamp that crosses the wire.
///
/// Android sends `SystemClock.elapsedRealtimeNanos()` in `sensor_time_ns`,
/// `encoded_time_ns`, `phone_send_time_ns` and in clock-probe replies, and the
/// PC subtracts them from each other (`camera.py`, `h264_stream.py`,
/// `transport_clock.py`). All of them must therefore share one clock domain.
///
/// AVCaptureVideoDataOutput presentation timestamps are expressed in the host
/// clock, which is `mach_absolute_time()` on Darwin. We therefore use
/// `mach_absolute_time()` here as well, converted through the mach timebase.
///
/// Difference from Android, documented rather than hidden: `mach_absolute_time`
/// does not advance while the device is asleep, while Android's
/// `elapsedRealtimeNanos` does. Streaming keeps the display awake
/// (`isIdleTimerDisabled`), so the two behave identically for the duration of a
/// session. `mach_continuous_time` must NOT be mixed in here — doing so would
/// silently break the PC's clock-offset estimate.
enum MonotonicClock {

    private static let timebase: mach_timebase_info_data_t = {
        var info = mach_timebase_info_data_t()
        mach_timebase_info(&info)
        return info
    }()

    /// Nanoseconds since boot, in the same domain as capture presentation
    /// timestamps.
    static func nowNs() -> UInt64 {
        let ticks = mach_absolute_time()
        let numer = UInt64(timebase.numer)
        let denom = UInt64(timebase.denom)
        if numer == denom { return ticks }
        return ticks / denom * numer + (ticks % denom) * numer / denom
    }

    /// Host-clock nanoseconds for a CoreMedia timestamp.
    ///
    /// `CMTimeGetSeconds` rounds through Double; for the 44-bit nanosecond
    /// values used here that is exact to the nanosecond at current uptimes, and
    /// the PC only ever uses differences.
    static func nanoseconds(from time: CMTime) -> UInt64 {
        guard time.isValid, time.timescale != 0 else { return 0 }
        let seconds = CMTimeGetSeconds(time)
        guard seconds.isFinite, seconds >= 0 else { return 0 }
        return UInt64((seconds * 1_000_000_000.0).rounded())
    }
}
