import XCTest
@testable import OpenGazeLink

/// The intrinsics message is the one place where iOS cannot mirror Android
/// exactly, because AVFoundation exposes no factory calibration for the video
/// path. These tests pin the mapping and, more importantly, re-implement the
/// PC's validation from `camera.py:505` so a schema drift fails here rather
/// than as a rejected packet in the field.
final class CameraIntrinsicsTests: XCTestCase {

    private func input(
        formatWidth: Int = 1280,
        formatHeight: Int = 720,
        streamWidth: Int = 1280,
        streamHeight: Int = 720,
        fov: Double = 76.0,
        zoom: Double = 1.0,
        reference: FovReference = .horizontal,
        rotation: Int = 0,
        crop: PixelCrop = PixelCrop(x: 0, y: 0, width: 1280, height: 720)
    ) -> IntrinsicsInput {
        IntrinsicsInput(
            cameraID: "device-1",
            lensFacing: "front",
            deviceModel: "iPhone17,2",
            deviceType: "AVCaptureDeviceTypeBuiltInTrueDepthCamera",
            formatWidth: formatWidth,
            formatHeight: formatHeight,
            streamWidth: streamWidth,
            streamHeight: streamHeight,
            videoFieldOfViewDegrees: fov,
            videoZoomFactor: zoom,
            fovReference: reference,
            frameRotation: rotation,
            softwareCrop: crop
        )
    }

    // MARK: - Focal length mapping

    func testHorizontalReferenceUsesThePinholeModel() {
        let fx = CameraIntrinsics.focalLengthPixels(input(fov: 76.0))
        let expected = 640.0 / tan(76.0 * Double.pi / 180.0 / 2.0)
        XCTAssertEqual(fx, expected, accuracy: 1e-9)
        // Sanity: a 76 degree horizontal field of view at 720p is a plausible
        // phone camera focal length in pixels.
        XCTAssertGreaterThan(fx, 700)
        XCTAssertLessThan(fx, 950)
    }

    func testDiagonalReferenceUsesTheImageDiagonal() {
        // 640x480 is a 3-4-5 triangle, so the half diagonal is exactly 400 and
        // a 90 degree diagonal field of view gives fx = 400.
        let fx = CameraIntrinsics.focalLengthPixels(
            input(formatWidth: 640, formatHeight: 480, streamWidth: 640, streamHeight: 480,
                  fov: 90.0, reference: .diagonal)
        )
        XCTAssertEqual(fx, 400.0, accuracy: 1e-9)
    }

    func testDiagonalAndHorizontalReferencesDifferByTheExpectedRatio() {
        let horizontal = CameraIntrinsics.focalLengthPixels(input(fov: 70.0, reference: .horizontal))
        let diagonal = CameraIntrinsics.focalLengthPixels(input(fov: 70.0, reference: .diagonal))
        // Reading a diagonal figure as horizontal overstates fx by the ratio of
        // the diagonal to the width, which is why the UI exposes the choice.
        let expectedRatio = hypot(1280.0, 720.0) / 1280.0
        XCTAssertEqual(diagonal / horizontal, expectedRatio, accuracy: 1e-9)
        XCTAssertGreaterThan(diagonal / horizontal, 1.14)
    }

    func testStreamScaleAndZoomAreApplied() {
        // A 4K format delivered as 1080p at 2x digital zoom.
        let fx = CameraIntrinsics.focalLengthPixels(
            input(formatWidth: 3840, formatHeight: 2160, streamWidth: 1920, streamHeight: 1080,
                  fov: 76.0, zoom: 2.0)
        )
        let expected = (3840.0 / 2.0) / tan(76.0 * Double.pi / 180.0 / 2.0) * 2.0 * (1920.0 / 3840.0)
        XCTAssertEqual(fx, expected, accuracy: 1e-9)
    }

    func testDegenerateFieldOfViewFallsBackInsteadOfProducingInfinity() {
        let fx = CameraIntrinsics.focalLengthPixels(input(fov: 0.0))
        XCTAssertTrue(fx.isFinite)
        XCTAssertEqual(fx, 1280.0, accuracy: 1e-9)
    }

    // MARK: - JSON shape

    func testJSONCarriesEveryFieldThePCReads() throws {
        let json = CameraIntrinsics.json(input(rotation: 90))
        XCTAssertEqual(json["schema"] as? String, CameraIntrinsics.schema)
        XCTAssertEqual(json["cameraId"] as? String, "device-1")
        XCTAssertEqual(json["lensFacing"] as? String, "front")
        XCTAssertEqual(json["frameRotation"] as? Int, 90)
        XCTAssertEqual(json["source"] as? String, "ios_avfoundation_video_field_of_view_horizontal")
        XCTAssertNotNil(json["timestampNs"] as? Int)

        let stream = try XCTUnwrap(json["streamIntrinsics"] as? [String: Any])
        for key in ["width", "height", "fx", "fy", "cx", "cy"] {
            XCTAssertNotNil(stream[key], "streamIntrinsics is missing \(key)")
        }

        let crop = try XCTUnwrap(json["softwareCrop"] as? [String: Any])
        for key in ["left", "top", "right", "bottom", "width", "height", "sourceWidth", "sourceHeight"] {
            XCTAssertNotNil(crop[key], "softwareCrop is missing \(key)")
        }
        XCTAssertEqual(crop["coordinateSystem"] as? String, "unrotated_stream")
        XCTAssertNotNil(json["distortion"] as? [Any])
    }

    func testUncroppedStreamDescribesAFullFrameSoftwareCrop() {
        let json = CameraIntrinsics.json(input())
        let crop = json["softwareCrop"] as? [String: Any]
        XCTAssertEqual(crop?["left"] as? Int, 0)
        XCTAssertEqual(crop?["top"] as? Int, 0)
        XCTAssertEqual(crop?["right"] as? Int, 0)
        XCTAssertEqual(crop?["bottom"] as? Int, 0)
        XCTAssertEqual(crop?["width"] as? Int, 1280)
        XCTAssertEqual(crop?["sourceWidth"] as? Int, 1280)

        let stream = json["streamIntrinsics"] as? [String: Any]
        XCTAssertEqual(stream?["width"] as? Int, 1280)
        XCTAssertEqual(stream?["height"] as? Int, 720)
    }

    func testCropTranslatesThePrincipalPointWithoutRescalingTheFocals() {
        let full = CameraIntrinsics.json(input())
        let cropped = CameraIntrinsics.json(input(crop: PixelCrop(x: 80, y: 10, width: 1120, height: 700)))

        let fullStream = full["streamIntrinsics"] as? [String: Any]
        let croppedStream = cropped["streamIntrinsics"] as? [String: Any]

        // Focal lengths are properties of the lens, not of the retained region.
        XCTAssertEqual(croppedStream?["fx"] as? Double, fullStream?["fx"] as? Double)
        XCTAssertEqual(croppedStream?["fy"] as? Double, fullStream?["fy"] as? Double)
        // The optical axis moves with the crop; it is never re-centred.
        let fullCx = try! XCTUnwrap(fullStream?["cx"] as? Double)
        let croppedCx = try! XCTUnwrap(croppedStream?["cx"] as? Double)
        XCTAssertEqual(croppedCx, fullCx - 80.0, accuracy: 1e-9)
        let fullCy = try! XCTUnwrap(fullStream?["cy"] as? Double)
        let croppedCy = try! XCTUnwrap(croppedStream?["cy"] as? Double)
        XCTAssertEqual(croppedCy, fullCy - 10.0, accuracy: 1e-9)
    }

    func testJSONSerialisesToTheEnvelopePayload() throws {
        let bytes = try XCTUnwrap(CameraIntrinsics.jsonData(input()))
        let object = try JSONSerialization.jsonObject(with: Data(bytes)) as? [String: Any]
        XCTAssertNotNil(object?["streamIntrinsics"])
    }

    // MARK: - PC compatibility

    /// Re-implementation of `camera.py:505` `_handle_intrinsics`. Any payload
    /// this accepts will be accepted by the PC's validator too.
    private func pcValidationFailure(_ json: [String: Any]) -> String? {
        func number(_ value: Any?) -> Double? {
            (value as? NSNumber)?.doubleValue
        }
        func integer(_ value: Any?) -> Int? {
            (value as? NSNumber)?.intValue
        }

        guard let stream = json["streamIntrinsics"] as? [String: Any] else { return "missing streamIntrinsics" }
        guard let width = integer(stream["width"]), let height = integer(stream["height"]),
              let fx = number(stream["fx"]), let fy = number(stream["fy"]),
              let cx = number(stream["cx"]), let cy = number(stream["cy"])
        else { return "streamIntrinsics is incomplete" }

        var checkWidth = width
        var checkHeight = height
        var checkCx = cx
        var checkCy = cy

        if let crop = json["softwareCrop"] as? [String: Any] {
            if crop["sourceWidth"] != nil || crop["sourceHeight"] != nil {
                guard let sourceWidth = integer(crop["sourceWidth"]),
                      let sourceHeight = integer(crop["sourceHeight"]),
                      let left = integer(crop["left"]), let top = integer(crop["top"]),
                      let right = integer(crop["right"]), let bottom = integer(crop["bottom"]),
                      let cropWidth = integer(crop["width"]), let cropHeight = integer(crop["height"])
                else { return "softwareCrop is incomplete" }

                checkWidth = sourceWidth
                checkHeight = sourceHeight
                guard sourceWidth > 0, sourceWidth <= 65535, sourceHeight > 0, sourceHeight <= 65535,
                      min(left, top, right, bottom) >= 0,
                      left + width + right == sourceWidth,
                      top + height + bottom == sourceHeight,
                      cropWidth == width, cropHeight == height
                else { return "software crop metadata is inconsistent" }
                checkCx += Double(left)
                checkCy += Double(top)
            }
        }

        let maximum = Double(max(checkWidth, checkHeight))
        guard width > 0, height > 0,
              0.1 * maximum <= fx, fx <= 10.0 * maximum,
              0.1 * maximum <= fy, fy <= 10.0 * maximum,
              -Double(checkWidth) <= checkCx, checkCx <= 2.0 * Double(checkWidth),
              -Double(checkHeight) <= checkCy, checkCy <= 2.0 * Double(checkHeight)
        else { return "intrinsics failed sanity checks" }
        return nil
    }

    func testPayloadPassesThePCsValidatorForEveryOfferedGeometry() {
        // The FOV is swept across the plausible range for a phone camera and
        // both reference axes, because an `fx` outside 0.1...10 x the frame
        // dimension would be rejected outright by camera.py:549.
        for reference in FovReference.allCases {
            for fov in stride(from: 30.0, through: 120.0, by: 5.0) {
                for rotation in FrameRotation.supported {
                    for crop in [
                        PixelCrop(x: 0, y: 0, width: 1280, height: 720),
                        PixelCrop(x: 0, y: 0, width: 640, height: 360),
                        PixelCrop(x: 80, y: 10, width: 1120, height: 700),
                        PixelCrop(x: 320, y: 180, width: 640, height: 360),
                    ] {
                        let json = CameraIntrinsics.json(input(
                            fov: fov, reference: reference, rotation: rotation, crop: crop
                        ))
                        XCTAssertNil(
                            pcValidationFailure(json),
                            "rejected: reference=\(reference.rawValue) fov=\(fov) rotation=\(rotation) crop=\(crop)"
                        )
                    }
                }
            }
        }
    }

    func testPayloadPassesThePCsValidatorForAHighResolutionJpegCrop() {
        // 4K Camera-mode crop: the UDP path is dimension agnostic, so the
        // intrinsics must stay valid well away from 1280x720.
        let json = CameraIntrinsics.json(input(
            formatWidth: 3840, formatHeight: 2160,
            streamWidth: 3840, streamHeight: 2160,
            fov: 76.0, reference: .diagonal,
            crop: PixelCrop(x: 640, y: 360, width: 2560, height: 1440)
        ))
        XCTAssertNil(pcValidationFailure(json))
    }

    func testValidatorItselfRejectsAnInconsistentCrop() {
        // Guard against a test that always passes: the validator must be able
        // to fail. This payload claims a crop that does not add up.
        var json = CameraIntrinsics.json(input(crop: PixelCrop(x: 80, y: 10, width: 1120, height: 700)))
        var crop = json["softwareCrop"] as? [String: Any] ?? [:]
        crop["right"] = 999
        json["softwareCrop"] = crop
        XCTAssertEqual(pcValidationFailure(json), "software crop metadata is inconsistent")
    }
}
