import AVFoundation
import SwiftUI

/// Live camera preview.
///
/// iOS has no equivalent of Android's `TextureView` for a `Surface`-fed
/// encoder; `AVCaptureVideoPreviewLayer` on the same session is the idiomatic
/// replacement and costs nothing extra because the frames are already being
/// delivered to `AVCaptureVideoDataOutput`.
struct CameraPreviewView: UIViewRepresentable {
    let session: AVCaptureSession?

    func makeUIView(context: Context) -> PreviewContainerView {
        PreviewContainerView()
    }

    func updateUIView(_ uiView: PreviewContainerView, context: Context) {
        uiView.attach(session: session)
    }
}

final class PreviewContainerView: UIView {

    override class var layerClass: AnyClass { AVCaptureVideoPreviewLayer.self }

    private var previewLayer: AVCaptureVideoPreviewLayer? {
        layer as? AVCaptureVideoPreviewLayer
    }

    /// The layer is not mirrored and not rotated, so the preview matches what
    /// the PC receives before its own rotation is applied.
    func attach(session: AVCaptureSession?) {
        guard let previewLayer else { return }
        if previewLayer.session !== session {
            previewLayer.session = session
        }
        previewLayer.videoGravity = .resizeAspect
        previewLayer.connection?.isEnabled = session != nil
    }
}
