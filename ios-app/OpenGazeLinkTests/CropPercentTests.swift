import XCTest
@testable import OpenGazeLink

/// Port of `CameraCropTest.kt`. The expected values are the Android test's
/// expected values, so the two platforms crop identically.
final class CropPercentTests: XCTestCase {

    private let asymmetric = CropPercent(left: 10, right: 20, top: 30, bottom: 40)!

    func testAsymmetricDisplayInsetsMapToAllFourRotations() {
        XCTAssertEqual(asymmetric.pixels(width: 200, height: 100, rotation: 0), PixelCrop(x: 20, y: 30, width: 140, height: 30))
        XCTAssertEqual(asymmetric.pixels(width: 200, height: 100, rotation: 90), PixelCrop(x: 60, y: 20, width: 60, height: 70))
        XCTAssertEqual(asymmetric.pixels(width: 200, height: 100, rotation: 180), PixelCrop(x: 40, y: 40, width: 140, height: 30))
        XCTAssertEqual(asymmetric.pixels(width: 200, height: 100, rotation: 270), PixelCrop(x: 80, y: 10, width: 60, height: 70))
    }

    /// Every retained pixel must be the same set as "rotate the whole frame,
    /// then crop the rotated view" — the property the inverse mapping exists to
    /// preserve.
    func testEveryRetainedPixelMatchesRotateThenCrop() {
        let width = 200
        let height = 100
        for rotation in [0, 90, 180, 270] {
            let rotatedWidth = rotation % 180 == 0 ? width : height
            let rotatedHeight = rotation % 180 == 0 ? height : width
            let display = asymmetric.pixels(width: rotatedWidth, height: rotatedHeight, rotation: 0)
            let raw = asymmetric.pixels(width: width, height: height, rotation: rotation)

            for y in 0..<height {
                for x in 0..<width {
                    let rotated: (x: Int, y: Int)
                    switch rotation {
                    case 90: rotated = (height - 1 - y, x)
                    case 180: rotated = (width - 1 - x, height - 1 - y)
                    case 270: rotated = (y, width - 1 - x)
                    default: rotated = (x, y)
                    }
                    let inRaw = x >= raw.x && x < raw.x + raw.width && y >= raw.y && y < raw.y + raw.height
                    let inDisplay = rotated.x >= display.x && rotated.x < display.x + display.width
                        && rotated.y >= display.y && rotated.y < display.y + display.height
                    XCTAssertEqual(inRaw, inDisplay, "rotation \(rotation) pixel (\(x),\(y))")
                }
            }
        }
    }

    func testCropIsEvenBoundedAndNonemptyAtLimits() {
        for width in [4, 6, 640, 1920] {
            for height in [4, 10, 480, 1080] {
                for rotation in [0, 90, 180, 270] {
                    let extreme = CropPercent(left: 45, right: 45, top: 45, bottom: 45)!
                    let crop = extreme.pixels(width: width, height: height, rotation: rotation)
                    XCTAssertNil(crop.validationFailure(frameWidth: width, frameHeight: height))
                    XCTAssertEqual(CropPercent.none.pixels(width: width, height: height, rotation: rotation),
                                   PixelCrop.full(width: width, height: height))
                }
            }
        }
    }

    func testPrincipalPointIsTranslatedNotRecentered() {
        let crop = PixelCrop(x: 80, y: 10, width: 60, height: 70)
        let factory = crop.principalPoint(streamWidth: 200, streamHeight: 100, cx: 123, cy: 42, estimated: false)
        XCTAssertEqual(factory.x, 43)
        XCTAssertEqual(factory.y, 32)

        let estimated = crop.principalPoint(streamWidth: 200, streamHeight: 100, cx: 0, cy: 0, estimated: true)
        XCTAssertEqual(estimated.x, 19.5)
        XCTAssertEqual(estimated.y, 39.5)

        // A valid optical axis can lie outside a retained off-axis ROI.
        let offAxis = crop.principalPoint(streamWidth: 200, streamHeight: 100, cx: 10, cy: 5, estimated: false)
        XCTAssertEqual(offAxis.x, -70)
        XCTAssertEqual(offAxis.y, -5)
    }

    func testInvalidPercentIsRejected() {
        for value in [-1.0, 46.0, Double.nan, Double.infinity, -Double.infinity] {
            XCTAssertNil(CropPercent(left: value, right: 0, top: 0, bottom: 0), "left = \(value)")
            XCTAssertNil(CropPercent(left: 0, right: value, top: 0, bottom: 0), "right = \(value)")
            XCTAssertNil(CropPercent(left: 0, right: 0, top: value, bottom: 0), "top = \(value)")
            XCTAssertNil(CropPercent(left: 0, right: 0, top: 0, bottom: value), "bottom = \(value)")
        }
        XCTAssertNotNil(CropPercent(left: 0, right: 45, top: 45, bottom: 0))
    }

    func testCropValidationReportsTheFirstFailure() {
        XCTAssertNotNil(PixelCrop(x: -1, y: 0, width: 100, height: 100).validationFailure(frameWidth: 200, frameHeight: 200))
        XCTAssertNotNil(PixelCrop(x: 0, y: 0, width: 101, height: 100).validationFailure(frameWidth: 200, frameHeight: 200))
        // Exceeds the frame on the right and the bottom.
        XCTAssertNotNil(PixelCrop(x: 0, y: 0, width: 202, height: 200).validationFailure(frameWidth: 200, frameHeight: 200))
        XCTAssertNotNil(PixelCrop(x: 0, y: 0, width: 2, height: 100).validationFailure(frameWidth: 200, frameHeight: 200))
        // A full-frame crop is valid: it is what High-speed mode always sends.
        XCTAssertNil(PixelCrop(x: 0, y: 0, width: 200, height: 200).validationFailure(frameWidth: 200, frameHeight: 200))
        XCTAssertNil(PixelCrop(x: 0, y: 0, width: 100, height: 100).validationFailure(frameWidth: 200, frameHeight: 200))
    }

    func testJpegQualityTiersMatchAndroid() {
        XCTAssertEqual(JpegQuality.levels, [50, 65, 80, 90, 95, 100])
        XCTAssertEqual(JpegQuality.default, 80)
        XCTAssertEqual(JpegQuality.normalized(80), 0.8, accuracy: 1e-9)
    }
}
