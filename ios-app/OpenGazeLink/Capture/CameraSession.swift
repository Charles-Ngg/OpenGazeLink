import AVFoundation
import CoreMedia
import CoreVideo
import Foundation

/// Owns one capture session and its transport for the whole streaming lifetime.
///
/// Mirrors Android's `CameraStream`: a single object owns the device, the
/// session, the encoder or JPEG path, the senders and the counters, and a
/// failure tears all of them down before reporting. Restarting means building a
/// new instance, which is why nothing here is reused across sessions.
///
/// Threading:
/// - `sessionQueue` — device/session configuration and `startRunning()`.
/// - `videoQueue` — `AVCaptureVideoDataOutput` sample buffers.
/// - `transportQueue` — JPEG encode plus UDP send, with a latest-only handoff.
/// - the TCP sender runs its own thread inside `AvcTcpSender`.
final class CameraSession {

    struct Configuration {
        let device: AVCaptureDevice
        let format: AVCaptureDevice.Format
        let frameDuration: CMTime
        let option: CaptureOption
        let host: String
        let port: UInt16
        let rotation: Int
        let cropPercent: CropPercent
        let jpegQualityPercent: Int
        let fovReference: FovReference
    }

    var onReady: (() -> Void)?
    var onRates: ((StreamRates) -> Void)?
    var onFailure: ((StreamFailure, String) -> Void)?
    /// Reports the `source` string that was last sent in an intrinsics packet.
    var onIntrinsicsSource: ((String) -> Void)?
    /// Hands the configured `AVCaptureSession` to the UI for preview.
    var onSessionCreated: ((AVCaptureSession) -> Void)?

    /// Matches Android's 2 s intrinsics heartbeat.
    private static let intrinsicsRefreshIntervalNs: UInt64 = 2_000_000_000
    /// Discard an encoding-age sample this far outside the plausible range
    /// instead of reporting a bogus number.
    private static let maximumPlausibleEncodingAgeMs: Double = 5_000

    private let configuration: Configuration
    /// Retained crop region in unrotated stream coordinates.
    ///
    /// High-speed mode always sends the full frame, exactly like Android: the
    /// PC's H.264 receiver compares the header size against the decoded size and
    /// only accepts 1280x720, so cropping there would break the stream rather
    /// than reduce bandwidth.
    private let softwareCrop: PixelCrop

    private let captureSession = AVCaptureSession()
    private let sessionQueue = DispatchQueue(label: "opengazelink.camera-session")
    private let videoQueue = DispatchQueue(label: "opengazelink.camera-video", qos: .userInitiated)
    private let transportQueue = DispatchQueue(label: "opengazelink.camera-transport", qos: .userInitiated)
    private let statisticsQueue = DispatchQueue(label: "opengazelink.camera-stats")
    private let resourceLock = NSLock()

    private var udpSender: UdpSender?
    private var avcSender: AvcTcpSender?
    private var encoder: H264Encoder?
    private let jpegEncoder = JpegEncoder()
    private var videoOutput: AVCaptureVideoDataOutput?
    private var relay: SampleBufferRelay?
    private var statisticsTimer: DispatchSourceTimer?

    private var statistics = StreamStatistics()
    private var intrinsicsSentAtNs: UInt64 = 0
    private var intrinsicsSourceName = ""
    private var isClosed = false
    private var failureReported = false

    // Latest-only JPEG handoff, matching Android's `pending`/`sending` pair.
    private let frameLock = NSLock()
    private var pendingFrame: (pixelBuffer: CVPixelBuffer, sensorTimeNs: UInt64)?
    private var jpegSending = false

    init(configuration: Configuration) {
        self.configuration = configuration
        let cropPercent = configuration.option.mode == .camera ? configuration.cropPercent : CropPercent.none
        self.softwareCrop = cropPercent.pixels(
            width: configuration.option.width,
            height: configuration.option.height,
            rotation: configuration.rotation
        )
    }

    deinit {
        close()
    }

    // MARK: - Lifecycle

    func start() {
        sessionQueue.async { [weak self] in
            guard let self, !self.closed else { return }

            do {
                try self.startTransports()
            } catch {
                self.fail(.connection, error)
                return
            }

            do {
                try self.configureSession()
            } catch let failure as CameraSessionFailure {
                self.fail(failure.kind, failure)
                return
            } catch {
                self.fail(.camera, error)
                return
            }

            self.captureSession.startRunning()
            if self.closed {
                self.captureSession.stopRunning()
                return
            }
            self.startStatisticsTimer()
            DispatchQueue.main.async { [weak self] in
                guard let self, !self.closed else { return }
                self.onReady?()
            }
        }
    }

    func stop() {
        sessionQueue.async { [weak self] in self?.close() }
    }

    private var closed: Bool {
        resourceLock.lock()
        defer { resourceLock.unlock() }
        return isClosed
    }

    /// Idempotent teardown. Order matters: the transports are closed first so a
    /// blocked write cannot keep the capture pipeline alive, then the session is
    /// stopped, then the encoder is drained.
    private func close() {
        resourceLock.lock()
        if isClosed {
            resourceLock.unlock()
            return
        }
        isClosed = true
        let udp = udpSender
        let avc = avcSender
        let activeEncoder = encoder
        udpSender = nil
        avcSender = nil
        encoder = nil
        resourceLock.unlock()

        statisticsTimer?.cancel()
        statisticsTimer = nil
        udp?.close()
        avc?.close()
        activeEncoder?.invalidate()

        frameLock.lock()
        pendingFrame = nil
        frameLock.unlock()

        if captureSession.isRunning {
            captureSession.stopRunning()
        }
        if let videoOutput {
            videoOutput.setSampleBufferDelegate(nil, queue: nil)
            captureSession.beginConfiguration()
            captureSession.removeOutput(videoOutput)
            captureSession.commitConfiguration()
        }
        videoOutput = nil
        relay = nil
    }

    // MARK: - Setup

    private func startTransports() throws {
        let udp = try UdpSender(host: configuration.host, port: configuration.port)
        resourceLock.lock()
        if isClosed {
            resourceLock.unlock()
            udp.close()
            return
        }
        udpSender = udp
        resourceLock.unlock()

        guard configuration.option.mode == .highSpeed else { return }

        let encoder = try H264Encoder(
            width: configuration.option.width,
            height: configuration.option.height,
            fps: configuration.option.fps
        )
        encoder.onPacket = { [weak self] packet in
            self?.handleEncoded(packet)
        }
        encoder.onFailure = { [weak self] error in
            self?.fail(.combination, error)
        }

        let avc = try AvcTcpSender(
            host: configuration.host,
            port: configuration.port,
            width: configuration.option.width,
            height: configuration.option.height
        )
        avc.onSent = { [weak self] bytes in
            self?.recordSent(bytes: bytes, countsAsFrame: true)
        }
        avc.onError = { [weak self] error in
            self?.fail(.connection, error)
        }

        resourceLock.lock()
        if isClosed {
            resourceLock.unlock()
            avc.close()
            encoder.invalidate()
            return
        }
        self.encoder = encoder
        avcSender = avc
        resourceLock.unlock()
    }

    private func configureSession() throws {
        let device = configuration.device
        let format = configuration.format
        let frameDuration = configuration.frameDuration

        // Reject a duration the active format did not advertise before touching
        // the device: AVFoundation raises for an unsupported frame duration.
        let requestedRate = 1.0 / max(frameDuration.seconds, 1e-9)
        let advertised = format.videoSupportedFrameRateRanges.contains { range in
            requestedRate >= range.minFrameRate - 0.01 && requestedRate <= range.maxFrameRate + 0.01
        }
        guard advertised else {
            throw CameraSessionFailure(
                kind: .combination,
                message: "\(configuration.option.label) is outside the format's advertised frame-rate ranges"
            )
        }

        // The device lock is held only for device properties. AVFoundation
        // advises against reconfiguring the session while holding it, and a
        // `defer` here would keep it held across `beginConfiguration()`.
        try device.lockForConfiguration()
        if device.activeFormat != format {
            device.activeFormat = format
        }
        device.activeVideoMinFrameDuration = frameDuration
        device.activeVideoMaxFrameDuration = frameDuration

        // Low-latency tuning, mirroring CameraTuning.kt. Each knob is guarded so
        // an unsupported setting is skipped rather than raising.
        if format.isVideoHDRSupported {
            // 10-bit HDR would change the pixel format the encoder sees and add
            // a tone-mapping stage; the Android path disables it too.
            device.automaticallyAdjustsVideoHDREnabled = false
            device.isVideoHDREnabled = false
        }
        if device.isLowLightBoostSupported {
            // Low-light boost lengthens exposure, which fights a 120 FPS target.
            device.automaticallyEnablesLowLightBoostWhenAvailable = false
        }
        if device.isSubjectAreaChangeMonitoringSupported {
            device.isSubjectAreaChangeMonitoringEnabled = false
        }
        // Nothing between here and the matching unlock may throw, so the lock
        // cannot be leaked.
        device.unlockForConfiguration()

        let output = AVCaptureVideoDataOutput()
        output.alwaysDiscardsLateVideoFrames = true
        output.videoSettings = [
            kCVPixelBufferPixelFormatTypeKey as String:
                Int(kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange),
        ]
        let relay = SampleBufferRelay { [weak self] pixelBuffer, presentationTimeNs in
            self?.handle(pixelBuffer: pixelBuffer, presentationTimeNs: presentationTimeNs)
        }
        output.setSampleBufferDelegate(relay, queue: videoQueue)

        captureSession.beginConfiguration()
        // inputPriority lets the device's activeFormat win instead of the
        // session preset silently clamping the frame rate.
        captureSession.sessionPreset = .inputPriority

        // Built before beginConfiguration() so a throwing initialiser cannot
        // leave the session in a half-configured state.
        let input: AVCaptureDeviceInput
        do {
            input = try AVCaptureDeviceInput(device: device)
        } catch {
            captureSession.commitConfiguration()
            throw CameraSessionFailure(
                kind: .camera,
                message: "Camera input could not be created: \(error.localizedDescription)"
            )
        }
        guard captureSession.canAddInput(input) else {
            captureSession.commitConfiguration()
            throw CameraSessionFailure(kind: .combination, message: "Camera input was rejected by the session")
        }
        captureSession.addInput(input)

        guard captureSession.canAddOutput(output) else {
            captureSession.commitConfiguration()
            throw CameraSessionFailure(kind: .combination, message: "Video data output was rejected by the session")
        }
        captureSession.addOutput(output)

        if let connection = output.connection(with: .video) {
            // The wire always carries sensor-native frames; the PC applies
            // rotation once to both pixels and intrinsics, exactly as Android
            // does. Leaving the connection rotation at a non-zero default would
            // rotate twice.
            if connection.isVideoRotationAngleSupported(0), connection.videoRotationAngle != 0 {
                connection.videoRotationAngle = 0
            }
            if connection.isVideoMirroringSupported, connection.isVideoMirrored {
                connection.automaticallyAdjustsVideoMirroring = false
                connection.isVideoMirrored = false
            }
            if connection.isVideoStabilizationSupported {
                connection.preferredVideoStabilizationMode = .off
            }
        }

        captureSession.commitConfiguration()

        videoOutput = output
        self.relay = relay
        DispatchQueue.main.async { [weak self] in
            guard let self else { return }
            self.onSessionCreated?(self.captureSession)
        }
    }

    // MARK: - Frame handling

    private func handle(pixelBuffer: CVPixelBuffer, presentationTimeNs: UInt64) {
        guard !closed else { return }
        recordCapture()

        let nowNs = MonotonicClock.nowNs()
        sendIntrinsicsIfDue(
            streamWidth: CVPixelBufferGetWidth(pixelBuffer),
            streamHeight: CVPixelBufferGetHeight(pixelBuffer),
            nowNs: nowNs
        )
        recordExposure()

        switch configuration.option.mode {
        case .highSpeed:
            // VideoToolbox retains the buffer until its output handler runs, so
            // no copy is needed here.
            resourceLock.lock()
            let encoder = self.encoder
            resourceLock.unlock()
            encoder?.encode(pixelBuffer: pixelBuffer, presentationTimeNs: presentationTimeNs)
        case .camera:
            enqueueJPEGFrame(pixelBuffer: pixelBuffer, sensorTimeNs: presentationTimeNs)
        }
    }

    private func handleEncoded(_ packet: H264Encoder.Packet) {
        guard !closed else { return }
        resourceLock.lock()
        let sender = avcSender
        resourceLock.unlock()
        guard let sender else { return }

        if packet.flags & WireFormat.avcFlagCodecConfig == 0, packet.sensorTimeNs > 0,
           packet.encodedTimeNs >= packet.sensorTimeNs {
            let ageMs = Double(packet.encodedTimeNs - packet.sensorTimeNs) / 1_000_000.0
            // A different clock domain on some future device would show up as an
            // implausible age; drop the sample instead of publishing nonsense.
            if ageMs <= CameraSession.maximumPlausibleEncodingAgeMs {
                recordEncodingAge(ms: ageMs)
            }
        }

        sender.offer(
            data: packet.data,
            sensorTimeNs: packet.sensorTimeNs,
            encodedTimeNs: packet.encodedTimeNs,
            flags: packet.flags
        )
    }

    private func enqueueJPEGFrame(pixelBuffer: CVPixelBuffer, sensorTimeNs: UInt64) {
        frameLock.lock()
        if closed {
            frameLock.unlock()
            return
        }
        let shouldStartSending = !jpegSending
        // Replacing the slot releases the previous buffer back to the capture
        // pool, which is what keeps latency bounded under load.
        pendingFrame = (pixelBuffer, sensorTimeNs)
        if shouldStartSending { jpegSending = true }
        frameLock.unlock()

        if shouldStartSending {
            transportQueue.async { [weak self] in self?.sendLatestJPEGFrames() }
        }
    }

    private func sendLatestJPEGFrames() {
        while true {
            frameLock.lock()
            let frame = pendingFrame
            pendingFrame = nil
            if frame == nil { jpegSending = false }
            let stopped = isClosed
            frameLock.unlock()

            guard let frame, !stopped else { return }

            resourceLock.lock()
            let sender = udpSender
            resourceLock.unlock()
            guard let sender else { return }

            guard let data = jpegEncoder.jpegData(
                from: frame.pixelBuffer,
                crop: softwareCrop,
                qualityPercent: configuration.jpegQualityPercent
            ) else { continue }

            let payload = [UInt8](data)
            guard let stats = sender.sendFrame(
                payload: payload,
                width: softwareCrop.width,
                height: softwareCrop.height,
                sensorTimeNs: frame.sensorTimeNs,
                format: WireFormat.formatJPEG
            ) else { continue }

            let nowNs = MonotonicClock.nowNs()
            if nowNs >= frame.sensorTimeNs {
                let ageMs = Double(nowNs - frame.sensorTimeNs) / 1_000_000.0
                if ageMs <= CameraSession.maximumPlausibleEncodingAgeMs {
                    recordEncodingAge(ms: ageMs)
                }
            }
            recordSent(bytes: stats.wireBytes, countsAsFrame: true)
        }
    }

    // MARK: - Intrinsics

    private func sendIntrinsicsIfDue(streamWidth: Int, streamHeight: Int, nowNs: UInt64) {
        if intrinsicsSentAtNs != 0, nowNs &- intrinsicsSentAtNs < CameraSession.intrinsicsRefreshIntervalNs {
            return
        }
        intrinsicsSentAtNs = nowNs

        let dimensions = CMVideoFormatDescriptionGetDimensions(configuration.format.formatDescription)
        let input = IntrinsicsInput(
            cameraID: configuration.device.uniqueID,
            lensFacing: lensFacingName(configuration.device.position),
            deviceModel: DeviceIdentity.machineIdentifier,
            deviceType: configuration.device.deviceType.rawValue,
            formatWidth: Int(dimensions.width),
            formatHeight: Int(dimensions.height),
            streamWidth: streamWidth,
            streamHeight: streamHeight,
            videoFieldOfViewDegrees: Double(configuration.format.videoFieldOfView),
            videoZoomFactor: Double(configuration.device.videoZoomFactor),
            fovReference: configuration.fovReference,
            frameRotation: configuration.rotation,
            softwareCrop: softwareCrop
        )

        guard let json = CameraIntrinsics.jsonData(input) else {
            fail(.intrinsics, CameraSessionFailure(kind: .intrinsics, message: "Intrinsics JSON could not be encoded"))
            return
        }
        resourceLock.lock()
        let sender = udpSender
        resourceLock.unlock()
        guard let sender, sender.sendIntrinsics(json) else {
            fail(.intrinsics, CameraSessionFailure(kind: .intrinsics, message: "Intrinsics packet could not be sent"))
            return
        }
        if intrinsicsSourceName != input.fovReference.sourceName {
            intrinsicsSourceName = input.fovReference.sourceName
            DispatchQueue.main.async { [weak self] in
                self?.onIntrinsicsSource?(input.fovReference.sourceName)
            }
        }
    }

    private func lensFacingName(_ position: AVCaptureDevice.Position) -> String {
        switch position {
        case .front: return "front"
        case .back: return "back"
        default: return "unknown"
        }
    }

    // MARK: - Statistics

    private func startStatisticsTimer() {
        let timer = DispatchSource.makeTimerSource(queue: statisticsQueue)
        timer.schedule(deadline: .now() + 1.0, repeating: 1.0)
        timer.setEventHandler { [weak self] in
            guard let self, !self.closed else { return }
            // Already on statisticsQueue, so the counters are mutated directly.
            // Re-entering through sync() here would deadlock.
            let rates = self.statistics.sample(nowNs: MonotonicClock.nowNs())
            DispatchQueue.main.async { [weak self] in
                self?.onRates?(rates)
            }
        }
        statisticsTimer = timer
        timer.resume()
    }

    private func recordCapture() {
        statisticsQueue.sync { statistics.recordCapture() }
    }

    private func recordSent(bytes: Int, countsAsFrame: Bool) {
        statisticsQueue.sync { statistics.recordSent(bytes: bytes, countsAsFrame: countsAsFrame) }
    }

    private func recordEncodingAge(ms: Double) {
        statisticsQueue.sync { statistics.recordEncodingAge(ms: ms) }
    }

    private func recordExposure() {
        let duration = configuration.device.exposureDuration
        let seconds = CMTimeGetSeconds(duration)
        guard seconds.isFinite, seconds >= 0 else { return }
        let milliseconds = seconds * 1000.0
        statisticsQueue.sync { statistics.recordExposure(ms: milliseconds) }
    }

    // MARK: - Failure

    private func fail(_ kind: StreamFailure, _ error: Error) {
        resourceLock.lock()
        let shouldReport = !isClosed && !failureReported
        if shouldReport { failureReported = true }
        resourceLock.unlock()
        guard shouldReport else { return }

        let message = (error as? LocalizedError)?.errorDescription ?? error.localizedDescription
        // Teardown is always hopped to sessionQueue. `fail` can run inside a
        // capture callback or a VideoToolbox output handler, and calling
        // AVCaptureSession.stopRunning() there would wait on the very callback
        // that is still on the stack.
        sessionQueue.async { [weak self] in self?.close() }
        DispatchQueue.main.async { [weak self] in
            self?.onFailure?(kind, message)
        }
    }
}

/// Tagged failure so `start()` can classify a setup error the way Android's
/// `StreamFailure` does.
struct CameraSessionFailure: LocalizedError {
    let kind: StreamFailure
    let message: String
    var errorDescription: String? { message }
}

/// Bridges `AVCaptureVideoDataOutput` callbacks to a closure so `CameraSession`
/// does not have to be an `NSObject` subclass.
private final class SampleBufferRelay: NSObject, AVCaptureVideoDataOutputSampleBufferDelegate {
    private let handler: (CVPixelBuffer, UInt64) -> Void

    init(handler: @escaping (CVPixelBuffer, UInt64) -> Void) {
        self.handler = handler
    }

    func captureOutput(
        _ output: AVCaptureOutput,
        didOutput sampleBuffer: CMSampleBuffer,
        from connection: AVCaptureConnection
    ) {
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        let presentationTimeNs = MonotonicClock.nanoseconds(
            from: CMSampleBufferGetPresentationTimeStamp(sampleBuffer)
        )
        handler(pixelBuffer, presentationTimeNs)
    }
}
