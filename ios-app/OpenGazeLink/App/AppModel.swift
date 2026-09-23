import AVFoundation
import Combine
import Foundation
import UIKit

/// All mutable app state and the streaming lifecycle.
///
/// Deliberately not `@MainActor`: the target is Swift 5 language mode, where
/// main-actor isolation is not enforced for synchronous callbacks and would
/// give a false sense of safety. Instead every mutation is funnelled through
/// `DispatchQueue.main.async`, and every background callback in this file does
/// so explicitly.
final class AppModel: ObservableObject {

    enum ConnectionState: Equatable {
        case idle
        case searching
        case sweeping(Int)
        case found(name: String, address: String)
        case paired(name: String, address: String, port: UInt16)
        case failed(String)

        static func == (lhs: ConnectionState, rhs: ConnectionState) -> Bool {
            switch (lhs, rhs) {
            case (.idle, .idle), (.searching, .searching): return true
            case (.sweeping(let a), .sweeping(let b)): return a == b
            case (.found(let a1, let a2), .found(let b1, let b2)): return a1 == b1 && a2 == b2
            case (.paired(let a1, let a2, let a3), .paired(let b1, let b2, let b3)):
                return a1 == b1 && a2 == b2 && a3 == b3
            case (.failed(let a), .failed(let b)): return a == b
            default: return false
            }
        }
    }

    // MARK: - Published state

    @Published var language: AppLanguage {
        didSet { store.language = language }
    }
    @Published var cameras: [CameraChoice] = []
    @Published var selectedCameraID: String = ""
    @Published var mode: SessionMode = .highSpeed {
        didSet { rotationCache = nil }
    }
    @Published var selectedOptionKey: String?
    @Published var rotationOverride: Int = -1
    @Published var cropLeft: String = "0"
    @Published var cropRight: String = "0"
    @Published var cropTop: String = "0"
    @Published var cropBottom: String = "0"
    @Published var jpegQuality: Int = JpegQuality.default
    @Published var fovReference: FovReference {
        didSet { store.fovReference = fovReference }
    }
    @Published var host: String {
        didSet { store.host = host }
    }
    @Published var portText: String = String(WireFormat.defaultDataPort)
    @Published var connectionState: ConnectionState = .idle
    @Published var statusText: String = ""
    @Published var rates: StreamRates = .empty
    @Published var isStreaming = false
    @Published var isLoadingCameras = false
    @Published var intrinsicsSource: String = ""
    @Published var previewSession: AVCaptureSession?
    @Published var errorText: String?

    // MARK: - Private state

    private let store: PairingStore
    private let pairingService: PairingService
    private var cameraSession: CameraSession?
    private var excludedCombinations: Set<String> = []
    private var rotationCache: (cameraID: String, value: Int)?
    private var hasAppeared = false

    init(store: PairingStore = PairingStore()) {
        self.store = store
        self.pairingService = PairingService(store: store)
        self.language = store.language
        self.fovReference = store.fovReference
        self.host = store.host
        if let port = store.port { self.portText = String(port) }
        self.statusText = L10n(language: store.language).text(.ready)
    }

    // MARK: - Localization

    var l10n: L10n { L10n(language: language) }

    func text(_ key: L10n.Key) -> String { l10n.text(key) }

    func text(_ key: L10n.Key, _ arguments: CVarArg...) -> String {
        String(format: l10n.text(key), arguments: arguments)
    }

    // MARK: - Derived capture state

    var selectedCamera: CameraChoice? {
        cameras.first { $0.id == selectedCameraID }
    }

    /// Options for the current camera and mode, minus combinations the camera
    /// already refused during this app run.
    var options: [CaptureOption] {
        guard let camera = selectedCamera else { return [] }
        return camera.options.filter { $0.mode == mode && !excludedCombinations.contains($0.key) }
    }

    var resolutions: [CaptureResolution] { CaptureOptions.resolutions(options) }
    var frameRates: [Int] { CaptureOptions.frameRates(options) }

    var currentOption: CaptureOption? {
        CaptureOptions.selected(options, savedKey: selectedOptionKey)
    }

    var currentBinding: CaptureFormatBinding? {
        guard let option = currentOption, let camera = selectedCamera else { return nil }
        return camera.bindings[option.key]
    }

    /// Resolved clockwise rotation. A cached coordinator reading keeps the
    /// value stable for the whole session, matching Android, which samples the
    /// display rotation once when the stream starts.
    var resolvedRotation: Int {
        if rotationOverride >= 0 { return rotationOverride }
        guard let camera = selectedCamera else { return 0 }
        if let cached = rotationCache, cached.cameraID == camera.id { return cached.value }
        let value = FrameRotation.automatic(device: camera.device, previewLayer: nil) ?? 0
        rotationCache = (camera.id, value)
        return value
    }

    var rotationOptions: [(label: String, value: Int)] {
        [(text(.rotationAuto), -1), ("0°", 0), ("90°", 90), ("180°", 180), ("270°", 270)]
    }

    var selectedRotationLabel: String {
        rotationOptions.first { $0.value == rotationOverride }?.label ?? text(.rotationAuto)
    }

    /// Parsed crop, or nil when any field is out of range.
    var cropPercent: CropPercent? {
        guard let left = Double(cropLeft.replacingOccurrences(of: ",", with: ".")),
              let right = Double(cropRight.replacingOccurrences(of: ",", with: ".")),
              let top = Double(cropTop.replacingOccurrences(of: ",", with: ".")),
              let bottom = Double(cropBottom.replacingOccurrences(of: ",", with: "."))
        else { return nil }
        return CropPercent(left: left, right: right, top: top, bottom: bottom)
    }

    /// Retained region in unrotated stream coordinates, for the size readout.
    var outputCrop: PixelCrop? {
        guard let option = currentOption, let crop = cropPercent else { return nil }
        let effective = mode == .camera ? crop : CropPercent.none
        return effective.pixels(width: option.width, height: option.height, rotation: resolvedRotation)
    }

    /// What the PC will actually display, after its own rotation.
    var outputSizeDescription: String? {
        guard let option = currentOption, let crop = outputCrop else { return nil }
        let rotation = resolvedRotation
        let swapped = rotation == 90 || rotation == 270
        let width = swapped ? crop.height : crop.width
        let height = swapped ? crop.width : crop.height
        return text(.cropOutput, width, height, rotation)
    }

    var port: UInt16? {
        guard let value = Int(portText.trimmingCharacters(in: .whitespaces)), (1...65535).contains(value) else {
            return nil
        }
        return UInt16(value)
    }

    var canStart: Bool {
        !isStreaming && !isLoadingCameras && currentOption != nil && currentBinding != nil
    }

    // MARK: - Lifecycle

    func onAppear() {
        guard !hasAppeared else { return }
        hasAppeared = true
        setStatus(.ready)
        refreshCameras()
        startDiscovery()
    }

    func onDisappear() {
        stopStreaming()
        pairingService.stop()
    }

    /// Keeps the display awake for the whole session. Android holds a
    /// `WIFI_MODE_FULL_LOW_LATENCY` lock instead, which has no iOS equivalent.
    private func applyIdleTimerPolicy() {
        UIApplication.shared.isIdleTimerDisabled = isStreaming
    }

    // MARK: - Cameras

    func refreshCameras() {
        guard !isStreaming, !isLoadingCameras else { return }
        isLoadingCameras = true
        setStatus(.readingCameras)

        let proceed = { [weak self] in
            guard let self else { return }
            DispatchQueue.global(qos: .userInitiated).async {
                let choices = CaptureCatalog.read()
                DispatchQueue.main.async {
                    self.isLoadingCameras = false
                    self.apply(cameraChoices: choices)
                }
            }
        }

        switch AVCaptureDevice.authorizationStatus(for: .video) {
        case .authorized:
            proceed()
        case .notDetermined:
            AVCaptureDevice.requestAccess(for: .video) { granted in
                DispatchQueue.main.async {
                    if granted {
                        proceed()
                    } else {
                        self.isLoadingCameras = false
                        self.setStatus(.permissionDenied)
                        self.errorText = self.text(.permissionDenied)
                    }
                }
            }
        default:
            isLoadingCameras = false
            setStatus(.permissionDenied)
            errorText = text(.permissionDenied)
        }
    }

    private func apply(cameraChoices choices: [CameraChoice]) {
        cameras = choices
        if !choices.contains(where: { $0.id == selectedCameraID }) {
            let withOptions = choices.filter { choice in choice.options.contains { $0.mode == mode } }
            let fallback = withOptions.first { $0.isFront }
                ?? withOptions.first
                ?? choices.first { $0.isFront }
                ?? choices.first
            selectedCameraID = fallback?.id ?? ""
            selectedOptionKey = nil
            rotationCache = nil
        }
        if choices.isEmpty {
            setStatus(.noCameras)
        } else if currentOption == nil {
            selectedOptionKey = options.first?.key
            setStatus(.ready)
        }
    }

    func selectCamera(_ id: String) {
        guard !isStreaming else { return }
        selectedCameraID = id
        selectedOptionKey = nil
        rotationCache = nil
    }

    func selectResolution(_ resolution: CaptureResolution) {
        guard !isStreaming else { return }
        if let chosen = CaptureOptions.selectResolution(options, resolution: resolution, preferred: currentOption) {
            selectedOptionKey = chosen.key
        }
    }

    func selectFrameRate(_ fps: Int) {
        guard !isStreaming else { return }
        if let chosen = CaptureOptions.selectFrameRate(options, fps: fps, preferred: currentOption) {
            selectedOptionKey = chosen.key
        }
    }

    // MARK: - Discovery

    func startDiscovery() {
        pairingService.start { [weak self] event in
            DispatchQueue.main.async { self?.apply(discoveryEvent: event) }
        }
    }

    private func apply(discoveryEvent event: PairingService.Event) {
        switch event {
        case .searching:
            if case .failed = connectionState { connectionState = .idle }
            if case .paired = connectionState { return }
            connectionState = .searching
            setStatus(.discoveryActive)

        case .sweeping(let count):
            connectionState = .sweeping(count)
            setStatus(.sweepProgress, count)

        case .offer(let offer):
            if offer.accepted {
                if isStreaming, host != offer.address {
                    connectionState = .paired(name: offer.pcName, address: offer.address, port: offer.dataPort)
                    setStatus(.pairingChanged)
                    return
                }
                host = offer.address
                portText = String(offer.dataPort)
                store.pairedInstanceID = offer.instanceID
                store.port = offer.dataPort
                connectionState = .paired(name: offer.pcName, address: offer.address, port: offer.dataPort)
                setStatus(.paired, offer.pcName, offer.address, Int(offer.dataPort))
            } else {
                connectionState = .found(name: offer.pcName, address: offer.address)
                setStatus(.pcFound, offer.pcName, offer.address)
            }

        case .failed(let message):
            if case .paired = connectionState { return }
            connectionState = .failed(message)
            setStatus(.discoveryFailed)
        }
    }

    func changePairedPC() {
        guard !isStreaming else { return }
        store.forgetPairedPC()
        host = ""
        portText = String(WireFormat.defaultDataPort)
        connectionState = .idle
        setStatus(.discoveryActive)
        startDiscovery()
    }

    // MARK: - Streaming

    func toggleStreaming() {
        if isStreaming { stopStreaming() } else { startStreaming() }
    }

    func startStreaming() {
        guard !isStreaming else { return }

        let trimmedHost = host.trimmingCharacters(in: .whitespaces)
        guard !trimmedHost.isEmpty else {
            errorText = text(.hostRequired)
            setStatus(.hostRequired)
            return
        }
        guard let port else {
            errorText = text(.portInvalid)
            setStatus(.portInvalid)
            return
        }
        guard let option = currentOption, let binding = currentBinding, let camera = selectedCamera else {
            errorText = text(mode == .highSpeed ? .noHighSpeed : .noCameraOptions)
            return
        }
        if mode == .camera, cropPercent == nil {
            errorText = text(.cropInvalid)
            return
        }

        // Rotation is resolved fresh for each session, then frozen.
        rotationCache = nil
        let rotation = resolvedRotation
        let crop = cropPercent ?? CropPercent.none

        let configuration = CameraSession.Configuration(
            device: camera.device,
            format: binding.format,
            frameDuration: binding.frameDuration,
            option: option,
            host: trimmedHost,
            port: port,
            rotation: rotation,
            cropPercent: crop,
            jpegQualityPercent: jpegQuality,
            fovReference: fovReference
        )

        let session = CameraSession(configuration: configuration)
        session.onReady = { [weak self] in
            DispatchQueue.main.async {
                guard let self else { return }
                self.isStreaming = true
                self.applyIdleTimerPolicy()
                self.setStatus(.streaming, option.label)
            }
        }
        session.onRates = { [weak self] rates in
            DispatchQueue.main.async { self?.rates = rates }
        }
        session.onIntrinsicsSource = { [weak self] source in
            DispatchQueue.main.async { self?.intrinsicsSource = source }
        }
        session.onSessionCreated = { [weak self] captureSession in
            DispatchQueue.main.async { self?.previewSession = captureSession }
        }
        session.onFailure = { [weak self] kind, message in
            DispatchQueue.main.async { self?.handleStreamFailure(kind: kind, message: message) }
        }

        cameraSession = session
        setStatus(.opening, option.label)
        session.start()
    }

    func stopStreaming() {
        guard let session = cameraSession else {
            isStreaming = false
            previewSession = nil
            applyIdleTimerPolicy()
            return
        }
        cameraSession = nil
        session.stop()
        isStreaming = false
        previewSession = nil
        rates = .empty
        applyIdleTimerPolicy()
        setStatus(.stopped)
        startDiscovery()
    }

    private func handleStreamFailure(kind: StreamFailure, message: String) {
        cameraSession = nil
        isStreaming = false
        previewSession = nil
        rates = .empty
        applyIdleTimerPolicy()
        errorText = message
        setStatus(kind.messageKey)

        // A rejected size/rate pair is remembered for this run so the picker
        // stops offering it, exactly like Android's excludedCombinations.
        if kind == .combination, let option = currentOption {
            excludedCombinations.insert(option.key)
            selectedOptionKey = options.first?.key
        }
    }

    func dismissError() {
        errorText = nil
    }

    private func setStatus(_ key: L10n.Key, _ arguments: CVarArg...) {
        statusText = String(format: l10n.text(key), arguments: arguments)
    }
}
