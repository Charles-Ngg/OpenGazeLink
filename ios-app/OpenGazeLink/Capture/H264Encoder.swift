import CoreMedia
import CoreVideo
import Foundation
import VideoToolbox

/// Real-time hardware H.264 encoder, matching Android's `AvcEncoder`.
///
/// Settings are chosen so that one input frame always yields exactly one
/// decodable access unit in submission order:
/// - `RealTime = true` — low latency mode.
/// - `AllowFrameReordering = false` — the PC's `AvcDecoder.decode_native`
///   raises when a packet does not decode to exactly one frame whose PTS equals
///   the header's `sensor_time_ns` (h264_stream.py:44), so B-frames and
///   pyramid reordering are fatal, not merely undesirable.
/// - Baseline profile + CAVLC — matches `AVCProfileBaseline` and
///   `KEY_MAX_B_FRAMES = 0` on Android.
/// - `MaxKeyFrameInterval = fps`, `MaxKeyFrameIntervalDuration = 1` — Android's
///   `KEY_I_FRAME_INTERVAL = 1`.
/// - `AverageBitRate` — Android's `(w * h * fps / 8).coerceIn(2 Mbps, 80 Mbps)`.
///
/// VideoToolbox emits AVCC access units with SPS/PPS in the format description.
/// `H264SampleBridge` converts both to the Annex-B the PC expects.
final class H264Encoder {

    struct Packet {
        let data: [UInt8]
        /// `WireFormat.avcFlag*` bits, matching Android's MediaCodec flags.
        let flags: UInt32
        let sensorTimeNs: UInt64
        let encodedTimeNs: UInt64
    }

    struct Diagnostics {
        var encodedFrames: UInt64 = 0
        var droppedFrames: UInt64 = 0
        var codecConfigPackets: UInt64 = 0
        var pendingFrames: Int = 0
        var lastAccessUnitBytes: Int = 0
    }

    enum Failure: LocalizedError {
        case sessionCreation(OSStatus)
        case unsupportedProperty(String, OSStatus)
        case encode(OSStatus)

        var errorDescription: String? {
            switch self {
            case .sessionCreation(let status):
                return "VideoToolbox could not create an H.264 session (status \(status))"
            case .unsupportedProperty(let key, let status):
                return "H.264 encoder rejected \(key) (status \(status))"
            case .encode(let status):
                return "H.264 encode failed (status \(status))"
            }
        }
    }

    let width: Int
    let height: Int
    let fps: Int
    let bitrate: Int

    /// Called on VideoToolbox's internal serial queue, in submission order.
    var onPacket: ((Packet) -> Void)?
    /// Called once, on the first fatal encoder error.
    var onFailure: ((Error) -> Void)?

    private let session: VTCompressionSession
    private let lock = NSLock()
    private var lastCodecConfig: [UInt8]?
    private var diagnostics = Diagnostics()
    private var failed = false
    private var invalidated = false

    init(width: Int, height: Int, fps: Int) throws {
        self.width = width
        self.height = height
        self.fps = max(1, fps)
        // Same heuristic and clamp as AvcEncoder.format().
        self.bitrate = min(max(width * height * max(1, fps) / 8, 2_000_000), 80_000_000)

        // Hardware-encoder hints. The specification key is only available from
        // iOS 17.4, so it is added when the running OS supports it; VideoToolbox
        // already prefers the hardware H.264 encoder when the key is absent.
        var encoderSpecification: [CFString: Any] = [:]
        if #available(iOS 17.4, *) {
            encoderSpecification[kVTVideoEncoderSpecification_EnableHardwareAcceleratedVideoEncoder] = true
        }
        let sourceAttributes: [CFString: Any] = [
            kCVPixelBufferPixelFormatTypeKey: Int(kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange),
            kCVPixelBufferWidthKey: width,
            kCVPixelBufferHeightKey: height,
            kCVPixelBufferIOSurfacePropertiesKey: [:] as [CFString: Any],
        ]

        var created: VTCompressionSession?
        let status = VTCompressionSessionCreate(
            allocator: kCFAllocatorDefault,
            width: Int32(width),
            height: Int32(height),
            codecType: CMVideoCodecType(kCMVideoCodecType_H264),
            encoderSpecification: encoderSpecification as CFDictionary,
            imageBufferAttributes: sourceAttributes as CFDictionary,
            compressedDataAllocator: nil,
            outputCallback: nil,
            refcon: nil,
            compressionSessionOut: &created
        )
        guard status == noErr, let session = created else {
            throw Failure.sessionCreation(status)
        }
        self.session = session

        do {
            try H264Encoder.configure(session, fps: self.fps, bitrate: self.bitrate)
        } catch {
            VTCompressionSessionInvalidate(session)
            throw error
        }
        _ = VTCompressionSessionPrepareToEncodeFrames(session)
    }

    deinit {
        invalidate()
    }

    /// Encodes one captured frame. The pixel buffer is retained by
    /// VideoToolbox until the output handler runs, so the caller may release it
    /// as soon as this returns.
    func encode(pixelBuffer: CVPixelBuffer, presentationTimeNs: UInt64) {
        lock.lock()
        if invalidated || failed {
            lock.unlock()
            return
        }
        diagnostics.pendingFrames += 1
        lock.unlock()

        // Nanosecond timebase so the PTS travels through VideoToolbox and back
        // unchanged, and the header value is bit-identical to the decoder's PTS.
        let presentationTime = CMTime(value: CMTimeValue(presentationTimeNs), timescale: 1_000_000_000)
        let duration = CMTime(value: 1, timescale: CMTimeScale(fps))

        let status = VTCompressionSessionEncodeFrameWithOutputHandler(
            session,
            imageBuffer: pixelBuffer,
            presentationTimeStamp: presentationTime,
            duration: duration,
            frameProperties: nil,
            infoFlagsOut: nil
        ) { [weak self] status, infoFlags, sampleBuffer in
            self?.handleOutput(status: status, infoFlags: infoFlags, sampleBuffer: sampleBuffer)
        }

        if status != noErr {
            lock.lock()
            diagnostics.pendingFrames = max(0, diagnostics.pendingFrames - 1)
            lock.unlock()
            report(Failure.encode(status))
        }
    }

    func invalidate() {
        lock.lock()
        let alreadyDone = invalidated
        invalidated = true
        lock.unlock()
        guard !alreadyDone else { return }
        _ = VTCompressionSessionCompleteFrames(session, untilPresentationTimeStamp: .invalid)
        VTCompressionSessionInvalidate(session)
    }

    func currentDiagnostics() -> Diagnostics {
        lock.lock()
        defer { lock.unlock() }
        return diagnostics
    }

    // MARK: - Output handling

    private func handleOutput(status: OSStatus, infoFlags: VTEncodeInfoFlags, sampleBuffer: CMSampleBuffer?) {
        lock.lock()
        if diagnostics.pendingFrames > 0 { diagnostics.pendingFrames -= 1 }
        lock.unlock()

        guard status == noErr else {
            report(Failure.encode(status))
            return
        }
        if infoFlags.contains(.frameDropped) {
            // A dropped frame emits no packet, so the PC sees no sequence gap:
            // the sender assigns sequence numbers at write time.
            lock.lock()
            diagnostics.droppedFrames += 1
            lock.unlock()
            return
        }
        guard let sampleBuffer, CMSampleBufferDataIsReady(sampleBuffer),
              let formatDescription = CMSampleBufferGetFormatDescription(sampleBuffer),
              let parameterSets = H264SampleBridge.parameterSets(formatDescription: formatDescription)
        else { return }

        let sensorTimeNs = MonotonicClock.nanoseconds(from: CMSampleBufferGetPresentationTimeStamp(sampleBuffer))
        let encodedTimeNs = MonotonicClock.nowNs()

        // Emit the SPS/PPS access unit whenever it changes, including the first
        // frame. The PC stores it and prepends it to every keyframe packet.
        let codecConfig = parameterSets.annexBCodecConfig
        lock.lock()
        let codecConfigChanged = codecConfig != lastCodecConfig
        if codecConfigChanged { lastCodecConfig = codecConfig }
        lock.unlock()

        if codecConfigChanged {
            lock.lock()
            diagnostics.codecConfigPackets += 1
            lock.unlock()
            onPacket?(Packet(
                data: codecConfig,
                flags: WireFormat.avcFlagCodecConfig,
                sensorTimeNs: sensorTimeNs,
                encodedTimeNs: encodedTimeNs
            ))
        }

        guard let accessUnit = H264SampleBridge.annexBAccessUnit(
            from: sampleBuffer, formatDescription: formatDescription
        ) else { return }

        // Bit 3 would make the PC raise "Partial H.264 access units are
        // unsupported" (h264_stream.py:39), so it is never set here; a partial
        // access unit is dropped instead of poisoning the stream.
        if infoFlags.contains(.asynchronous) {
            // Asynchronous mode only reports that the callback was not invoked
            // synchronously; the sample is complete and still in order.
        }
        var flags: UInt32 = 0
        if AnnexB.containsIDR(annexB: accessUnit) {
            flags |= WireFormat.avcFlagKeyframe
        }

        lock.lock()
        diagnostics.encodedFrames += 1
        diagnostics.lastAccessUnitBytes = accessUnit.count
        lock.unlock()

        onPacket?(Packet(
            data: accessUnit,
            flags: flags,
            sensorTimeNs: sensorTimeNs,
            encodedTimeNs: encodedTimeNs
        ))
    }

    private func report(_ error: Error) {
        lock.lock()
        let shouldReport = !failed
        failed = true
        lock.unlock()
        guard shouldReport else { return }
        onFailure?(error)
    }

    // MARK: - Session properties

    private static func configure(_ session: VTCompressionSession, fps: Int, bitrate: Int) throws {
        // Required: a rejected property here means the encoder cannot honour the
        // one-frame-in / one-frame-out contract the PC depends on.
        try set(session, kVTCompressionPropertyKey_RealTime, kCFBooleanTrue, required: true)
        try set(session, kVTCompressionPropertyKey_AllowFrameReordering, kCFBooleanFalse, required: true)
        try set(session, kVTCompressionPropertyKey_ProfileLevel,
                kVTProfileLevel_H264_Baseline_AutoLevel, required: true)
        try set(session, kVTCompressionPropertyKey_MaxKeyFrameInterval, fps as CFNumber, required: true)
        try set(session, kVTCompressionPropertyKey_MaxKeyFrameIntervalDuration, 1 as CFNumber, required: true)
        try set(session, kVTCompressionPropertyKey_ExpectedFrameRate, fps as CFNumber, required: true)
        try set(session, kVTCompressionPropertyKey_AverageBitRate, bitrate as CFNumber, required: true)

        // Optional: rejected on some encoders. Failing to set one of these only
        // costs a little latency, so it must not abort the session.
        try? set(session, kVTCompressionPropertyKey_MaxFrameDelayCount, 1 as CFNumber, required: false)
        try? set(session, kVTCompressionPropertyKey_PrioritizeEncodingSpeedOverQuality,
                kCFBooleanTrue, required: false)
        try? set(session, kVTCompressionPropertyKey_H264EntropyMode,
                kVTH264EntropyMode_CAVLC, required: false)
        try? set(session, kVTCompressionPropertyKey_AllowTemporalCompression,
                kCFBooleanTrue, required: false)
    }

    private static func set(
        _ session: VTCompressionSession,
        _ key: CFString,
        _ value: CFTypeRef?,
        required: Bool
    ) throws {
        let status = VTSessionSetProperty(session, key: key, value: value)
        if status != noErr && required {
            throw Failure.unsupportedProperty(key as String, status)
        }
    }
}
