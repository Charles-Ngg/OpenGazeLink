import SwiftUI

/// Connection, capture, orientation and performance controls.
///
/// Section order and wording follow the Android activity so a user moving
/// between the two platforms sees the same workflow: connect, choose capture,
/// start, watch metrics.
struct ContentView: View {
    @EnvironmentObject private var model: AppModel

    var body: some View {
        NavigationStack {
            Form {
                Section {
                    previewBlock
                    streamingControls
                }

                connectionSection
                captureSection
                orientationSection
                intrinsicsSection
                PerformanceSectionView()
            }
            .navigationTitle("OpenGazeLink")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    Picker(model.text(.languageLabel), selection: $model.language) {
                        ForEach(AppLanguage.allCases) { language in
                            Text(language.label).tag(language)
                        }
                    }
                    .pickerStyle(.menu)
                }
            }
            .alert(
                "OpenGazeLink",
                isPresented: Binding(
                    get: { model.errorText != nil },
                    set: { if !$0 { model.dismissError() } }
                ),
                actions: {
                    Button(model.text(.dismiss)) { model.dismissError() }
                },
                message: {
                    Text(model.errorText ?? "")
                }
            )
            .onDisappear { model.onDisappear() }
        }
    }

    // MARK: - Preview and streaming

    private var previewBlock: some View {
        ZStack {
            RoundedRectangle(cornerRadius: 10)
                .fill(Color.black)
            CameraPreviewView(session: model.previewSession)
            if model.previewSession == nil {
                Text(model.text(.streamHint))
                    .font(.footnote)
                    .foregroundStyle(.white.opacity(0.7))
                    .multilineTextAlignment(.center)
                    .padding(12)
            }
        }
        .frame(height: 200)
        .listRowInsets(EdgeInsets())
        .listRowBackground(Color.clear)
    }

    private var streamingControls: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(model.statusText)
                .font(.subheadline)
                .foregroundStyle(.secondary)

            HStack {
                Button(model.text(model.isStreaming ? .stopStream : .startStream)) {
                    model.toggleStreaming()
                }
                .buttonStyle(.borderedProminent)
                .tint(model.isStreaming ? .red : .accentColor)
                .disabled(!model.isStreaming && !model.canStart)

                if model.isStreaming {
                    Text(model.text(.streamHint))
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                }
            }
        }
        .padding(.vertical, 4)
    }

    // MARK: - Connection

    private var connectionSection: some View {
        Section(model.text(.connectionTitle)) {
            TextField(model.text(.hostHint), text: $model.host)
                .textInputAutocapitalization(.never)
                .autocorrectionDisabled()
                .keyboardType(.URL)
                .disabled(model.isStreaming)

            TextField(model.text(.portLabel), text: $model.portText)
                .keyboardType(.numberPad)
                .disabled(model.isStreaming)

            HStack {
                Button(model.text(.findPC)) { model.startDiscovery() }
                    .disabled(model.isStreaming)
                Spacer()
                Button(model.text(.changePairedPC)) { model.changePairedPC() }
                    .disabled(model.isStreaming)
                    .foregroundStyle(.red)
            }

            connectionStatusRow

            Text(model.text(.discoveryEntitlementHint))
                .font(.caption2)
                .foregroundStyle(.secondary)
        }
    }

    @ViewBuilder
    private var connectionStatusRow: some View {
        switch model.connectionState {
        case .idle:
            Text(model.text(.connectionIdle)).font(.footnote).foregroundStyle(.secondary)
        case .searching:
            Text(model.text(.discoveryActive)).font(.footnote).foregroundStyle(.secondary)
        case .sweeping(let count):
            Text(model.text(.sweepProgress, count)).font(.footnote).foregroundStyle(.secondary)
        case .found(let name, let address):
            Text(model.text(.pcFound, name, address)).font(.footnote)
        case .paired(let name, let address, let port):
            Text(model.text(.paired, name, address, Int(port))).font(.footnote)
        case .failed(let message):
            Text(message).font(.caption2).foregroundStyle(.orange)
        }
    }

    // MARK: - Capture

    private var captureSection: some View {
        Section(model.text(.captureTitle)) {
            Picker(model.text(.modeHighSpeed), selection: $model.mode) {
                ForEach(SessionMode.allCases) { mode in
                    Text(model.text(mode.titleKey)).tag(mode)
                }
            }
            .pickerStyle(.segmented)
            .disabled(model.isStreaming)

            Text(model.text(model.mode == .highSpeed ? .highSpeedHelp : .cameraHelp))
                .font(.caption2)
                .foregroundStyle(.secondary)

            Picker(model.text(.cameraLabel), selection: Binding(
                get: { model.selectedCameraID },
                set: { model.selectCamera($0) }
            )) {
                ForEach(model.cameras) { camera in
                    Text(cameraTitle(camera)).tag(camera.id)
                }
            }
            .disabled(model.isStreaming || model.cameras.isEmpty)

            if model.cameras.isEmpty {
                Text(model.text(model.isLoadingCameras ? .readingCameras : .noCameras))
                    .font(.footnote)
                    .foregroundStyle(.secondary)
            }

            Picker(model.text(.resolutionLabel), selection: Binding(
                get: { model.currentOption?.resolution ?? model.resolutions.first },
                set: { value in if let value { model.selectResolution(value) } }
            )) {
                ForEach(model.resolutions, id: \.self) { resolution in
                    Text(resolution.label).tag(Optional(resolution))
                }
            }
            .disabled(model.isStreaming || model.resolutions.isEmpty)

            Picker(model.text(.frameRateLabel), selection: Binding(
                get: { model.currentOption?.fps ?? model.frameRates.first },
                set: { value in if let value { model.selectFrameRate(value) } }
            )) {
                ForEach(model.frameRates, id: \.self) { fps in
                    Text("\(fps) FPS").tag(Optional(fps))
                }
            }
            .disabled(model.isStreaming || model.frameRates.isEmpty)

            if model.resolutions.isEmpty {
                Text(model.text(model.mode == .highSpeed ? .noHighSpeed : .noCameraOptions))
                    .font(.footnote)
                    .foregroundStyle(.secondary)
            } else {
                Text(model.text(.formatsHelp, model.resolutions.count, model.frameRates.count))
                    .font(.caption2)
                    .foregroundStyle(.secondary)
            }

            if model.mode == .highSpeed {
                Text(model.text(.h264PCSupportedNote))
                    .font(.caption2)
                    .foregroundStyle(.secondary)
            }

            Button(model.text(.refreshFormats)) { model.refreshCameras() }
                .disabled(model.isStreaming || model.isLoadingCameras)
        }
    }

    private func cameraTitle(_ camera: CameraChoice) -> String {
        let key: L10n.Key = camera.isFront ? .frontCamera : .backCamera
        return model.text(key, camera.displayName)
    }

    // MARK: - Orientation and crop

    private var orientationSection: some View {
        Section(model.text(.orientationTitle)) {
            Picker(model.text(.rotationLabel), selection: Binding(
                get: { model.rotationOverride },
                set: { model.rotationOverride = $0 }
            )) {
                ForEach(model.rotationOptions, id: \.value) { option in
                    Text(option.label).tag(option.value)
                }
            }
            .disabled(model.isStreaming)

            if model.mode == .camera {
                HStack {
                    cropField(model.text(.cropLeft), text: $model.cropLeft)
                    cropField(model.text(.cropRight), text: $model.cropRight)
                }
                HStack {
                    cropField(model.text(.cropTop), text: $model.cropTop)
                    cropField(model.text(.cropBottom), text: $model.cropBottom)
                }

                Picker(model.text(.jpegQualityLabel), selection: $model.jpegQuality) {
                    ForEach(JpegQuality.levels, id: \.self) { level in
                        Text("Q\(level)").tag(level)
                    }
                }
                .disabled(model.isStreaming)

                Text(model.text(.jpegQualityHelp))
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                Text(model.text(.cropHelp))
                    .font(.caption2)
                    .foregroundStyle(.secondary)
            }

            if let description = model.outputSizeDescription {
                Text(description).font(.footnote)
            } else if model.mode == .camera {
                Text(model.text(.cropInvalid)).font(.footnote).foregroundStyle(.orange)
            }
        }
    }

    private func cropField(_ title: String, text: Binding<String>) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(title).font(.caption2).foregroundStyle(.secondary)
            TextField("0", text: text)
                .keyboardType(.decimalPad)
                .textFieldStyle(.roundedBorder)
                .disabled(model.isStreaming)
        }
    }

    // MARK: - Intrinsics

    private var intrinsicsSection: some View {
        Section(model.text(.intrinsicsTitle)) {
            Picker(model.text(.intrinsicsFovReference), selection: $model.fovReference) {
                ForEach(FovReference.allCases) { reference in
                    Text(model.text(reference.titleKey)).tag(reference)
                }
            }
            .disabled(model.isStreaming)

            if model.intrinsicsSource.isEmpty {
                Text(model.text(.notAvailable)).font(.footnote).foregroundStyle(.secondary)
            } else {
                Text(model.text(.intrinsicsSource, model.intrinsicsSource)).font(.footnote)
            }

            Text(model.text(.intrinsicsLimitation))
                .font(.caption2)
                .foregroundStyle(.secondary)
        }
    }
}
