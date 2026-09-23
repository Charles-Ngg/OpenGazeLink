import CoreImage
import CoreVideo
import Foundation
// kCGImageDestinationLossyCompressionQuality is declared in ImageIO, and is the
// documented quality key for CIContext.jpegRepresentation.
import ImageIO

/// Crops a captured frame and JPEG-encodes it without resizing.
///
/// Android's Camera mode copies the retained YUV region into an NV21 buffer and
/// calls `YuvImage.compressToJpeg`. iOS has no equivalent software NV21
/// encoder, so the crop is expressed as a CoreImage extent and rendered
/// straight to JPEG. Two properties matter for PC compatibility:
///
/// 1. The output is **cropped, never scaled**. `camera.decode_udp_frame`
///    rejects a JPEG whose decoded size does not equal the packet header
///    (camera.py:80).
/// 2. The retained rectangle is expressed in unrotated stream coordinates, so
///    the PC can apply `frameRotation` once to both pixels and intrinsics.
///
/// Cost note: unlike Android this is a GPU round trip. Camera mode runs at
/// 60 FPS or less and is not the latency-critical route, so that is acceptable;
/// the high-speed route never touches CoreImage.
final class JpegEncoder {

    private let context: CIContext
    private let colorSpace = CGColorSpaceCreateDeviceRGB()

    init() {
        // No intermediate caching: every frame is a new buffer, and holding
        // rendered intermediates would only add memory pressure and latency.
        context = CIContext(options: [.cacheIntermediates: false])
    }

    /// - Parameter qualityPercent: one of `JpegQuality.levels`.
    func jpegData(from pixelBuffer: CVPixelBuffer, crop: PixelCrop, qualityPercent: Int) -> Data? {
        let bufferWidth = CVPixelBufferGetWidth(pixelBuffer)
        let bufferHeight = CVPixelBufferGetHeight(pixelBuffer)
        guard crop.validationFailure(frameWidth: bufferWidth, frameHeight: bufferHeight) == nil else {
            return nil
        }

        let image = CIImage(cvPixelBuffer: pixelBuffer)
        // CIImage coordinates put the origin at the bottom-left, while crop
        // insets are expressed from the top-left of the captured frame.
        let extent = CGRect(
            x: CGFloat(crop.x),
            y: CGFloat(bufferHeight - crop.y - crop.height),
            width: CGFloat(crop.width),
            height: CGFloat(crop.height)
        )
        let cropped = image
            .cropped(to: extent)
            .transformed(by: CGAffineTransform(translationX: -extent.origin.x, y: -extent.origin.y))

        let options: [CIImageRepresentationOption: Any] = [
            CIImageRepresentationOption(rawValue: kCGImageDestinationLossyCompressionQuality as String):
                JpegQuality.normalized(qualityPercent),
        ]
        return context.jpegRepresentation(of: cropped, colorSpace: colorSpace, options: options)
    }
}
