import CoreMedia
import Foundation

/// Which axis `AVCaptureDevice.Format.videoFieldOfView` describes.
///
/// Apple documents the property only as "the format's field of view, in
/// degrees" without naming the axis, and the horizontal and diagonal readings
/// differ by about 15% at 16:9 (the diagonal is `hypot(W, H) / W` times the
/// width). `fx` scales linearly with that error, and the PC projects the 3D eye
/// plane through `fx` to build the 64x36 normalized eye patch, so the choice
/// matters. Horizontal is the default because it is what the
/// equivalent-focal-length numbers imply; the UI exposes the toggle so it can
/// be settled on the device instead of guessed here.
enum FovReference: String, CaseIterable, Identifiable {
    case horizontal
    case diagonal

    var id: String { rawValue }

    var titleKey: L10n.Key {
        self == .horizontal ? .intrinsicsFovHorizontal : .intrinsicsFovDiagonal
    }

    /// `camera.py` displays this string verbatim as `intrinsicsSource`.
    var sourceName: String {
        switch self {
        case .horizontal: return "ios_avfoundation_video_field_of_view_horizontal"
        case .diagonal: return "ios_avfoundation_video_field_of_view_diagonal"
        }
    }
}

/// Everything needed to build a PC-compatible intrinsics message.
///
/// Kept as plain values so the mapping from AVFoundation to the Camera2 schema
/// is unit-testable without a camera.
struct IntrinsicsInput {
    let cameraID: String
    let lensFacing: String
    let deviceModel: String
    let deviceType: String
    /// Dimensions reported by the active format's format description.
    let formatWidth: Int
    let formatHeight: Int
    /// Dimensions actually delivered by AVCaptureVideoDataOutput. Usually equal
    /// to the format dimensions; scaled for rather than assumed.
    let streamWidth: Int
    let streamHeight: Int
    let videoFieldOfViewDegrees: Double
    let videoZoomFactor: Double
    let fovReference: FovReference
    let frameRotation: Int
    /// Retained region in unrotated stream coordinates.
    let softwareCrop: PixelCrop
}

/// Maps iOS camera geometry onto the schema the PC already parses.
///
/// The PC reads exactly four things out of the intrinsics JSON
/// (camera.py:505):
/// - `streamIntrinsics.{width,height,fx,fy,cx,cy}`
/// - `frameRotation`
/// - `softwareCrop.{left,top,right,bottom,width,height,sourceWidth,sourceHeight}`
/// - `source` and `distortion` for reporting
///
/// Everything else it keeps as opaque metadata, so the extra `ios*` keys below
/// are free diagnostics rather than protocol surface.
enum CameraIntrinsics {

    static let schema = "opengazelink-ios-avfoundation-intrinsics-v1"

    /// Pinhole focal length in stream pixels.
    ///
    /// - Horizontal reference: `fx = (W / 2) / tan(hFov / 2)`.
    /// - Diagonal reference: the diagonal of the image maps to the reported
    ///   field of view, so `fx = hypot(W, H) / 2 / tan(dFov / 2)`.
    ///
    /// Then scaled by the digital zoom, and by `streamWidth / formatWidth` so a
    /// downscaled data output still reports intrinsics for the frame it
    /// actually sends.
    static func focalLengthPixels(_ input: IntrinsicsInput) -> Double {
        let halfAngle = input.videoFieldOfViewDegrees * Double.pi / 180.0 / 2.0
        let tangent = tan(halfAngle)
        guard tangent.isFinite, tangent > 1e-6 else { return Double(max(input.streamWidth, input.streamHeight)) }

        let formatWidth = Double(input.formatWidth)
        let formatHeight = Double(input.formatHeight)
        let reference: Double
        switch input.fovReference {
        case .horizontal:
            reference = formatWidth / 2.0
        case .diagonal:
            reference = hypot(formatWidth, formatHeight) / 2.0
        }

        let zoom = input.videoZoomFactor.isFinite && input.videoZoomFactor > 0 ? input.videoZoomFactor : 1.0
        let fxAtFormat = reference / tangent * zoom
        guard formatWidth > 0 else { return fxAtFormat }
        let scale = Double(input.streamWidth) / formatWidth
        return fxAtFormat * scale
    }

    /// JSON body placed inside the `EYCI` envelope.
    static func json(_ input: IntrinsicsInput) -> [String: Any] {
        let fx = focalLengthPixels(input)
        // iOS sensors have square pixels and no skew, so fy mirrors fx and the
        // skew term is zero. The PC reads the skew term but does not use it.
        let fy = fx
        // The principal point is the frame centre: AVFoundation exposes no
        // calibrated optical axis for the video path.
        let baseCx = (Double(input.streamWidth) - 1.0) * 0.5
        let baseCy = (Double(input.streamHeight) - 1.0) * 0.5
        let crop = input.softwareCrop
        // Cropping translates the optical axis; it must never re-centre it.
        let cx = baseCx - Double(crop.x)
        let cy = baseCy - Double(crop.y)

        let right = input.streamWidth - crop.x - crop.width
        let bottom = input.streamHeight - crop.y - crop.height

        return [
            "schema": schema,
            "cameraId": input.cameraID,
            "lensFacing": input.lensFacing,
            "frameRotation": input.frameRotation,
            "source": input.fovReference.sourceName,
            "timestampNs": Int(MonotonicClock.nowNs()),
            "activeArray": [
                "left": 0, "top": 0,
                "right": input.streamWidth, "bottom": input.streamHeight,
                "width": input.streamWidth, "height": input.streamHeight,
            ],
            "preCorrectionActiveArray": [
                "left": 0, "top": 0,
                "right": input.streamWidth, "bottom": input.streamHeight,
                "width": input.streamWidth, "height": input.streamHeight,
            ],
            "captureCropRegion": [
                "left": 0, "top": 0,
                "right": input.formatWidth, "bottom": input.formatHeight,
                "width": input.formatWidth, "height": input.formatHeight,
            ],
            "effectiveStreamCrop": [
                "left": 0, "top": 0,
                "width": input.streamWidth, "height": input.streamHeight,
            ],
            "softwareCrop": [
                "coordinateSystem": "unrotated_stream",
                "left": crop.x, "top": crop.y,
                "right": right, "bottom": bottom,
                "width": crop.width, "height": crop.height,
                "sourceWidth": input.streamWidth, "sourceHeight": input.streamHeight,
            ],
            // Android sends the factory and derived vectors too. iOS has no
            // equivalent, so both are null rather than a fabricated matrix.
            "factoryIntrinsic": NSNull(),
            "derivedIntrinsic": NSNull(),
            "selectedSensorIntrinsic": [fx, fy, baseCx, baseCy, 0.0],
            "principalPointSource": "avfoundation_frame_center_before_software_crop",
            "focalLengthsMm": NSNull(),
            "sensorPhysicalSizeMm": NSNull(),
            "pixelArraySize": ["width": input.formatWidth, "height": input.formatHeight],
            "distortion": [],
            "streamIntrinsics": [
                "width": crop.width,
                "height": crop.height,
                "fx": fx,
                "fy": fy,
                "cx": cx,
                "cy": cy,
                "skew": 0.0,
            ],
            // iOS-only diagnostics. The PC keeps these in `sourceMetadata`.
            "iosFormatWidth": input.formatWidth,
            "iosFormatHeight": input.formatHeight,
            "iosStreamWidth": input.streamWidth,
            "iosStreamHeight": input.streamHeight,
            "iosVideoFieldOfViewDegrees": input.videoFieldOfViewDegrees,
            "iosFovReference": input.fovReference.rawValue,
            "iosVideoZoomFactor": input.videoZoomFactor,
            "iosDeviceModel": input.deviceModel,
            "iosDeviceType": input.deviceType,
            "iosTimestampClock": "mach_absolute_time",
        ]
    }

    /// Serialises the JSON body, or nil when the payload would not encode.
    static func jsonData(_ input: IntrinsicsInput) -> [UInt8]? {
        guard let data = try? JSONSerialization.data(withJSONObject: json(input), options: []) else {
            return nil
        }
        return [UInt8](data)
    }
}
