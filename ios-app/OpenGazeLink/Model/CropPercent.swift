import Foundation

/// Even-pixel region inside the unrotated capture stream.
///
/// Port of `PixelCrop` in `CameraCrop.kt`.
struct PixelCrop: Hashable {
    let x: Int
    let y: Int
    let width: Int
    let height: Int

    static func full(width: Int, height: Int) -> PixelCrop {
        PixelCrop(x: 0, y: 0, width: width, height: height)
    }

    /// Port of `PixelCrop.validateIn`. Returns the reason instead of throwing so
    /// the UI can show it and tests can assert on it.
    func validationFailure(frameWidth: Int, frameHeight: Int) -> String? {
        guard x >= 0, y >= 0, width >= 4, height >= 4 else {
            return "crop must start inside the frame and keep at least 4x4 pixels"
        }
        guard [x, y, width, height].allSatisfy({ $0 % 2 == 0 }) else {
            return "crop edges must land on even pixels"
        }
        guard x + width <= frameWidth, y + height <= frameHeight else {
            return "crop extends past \(frameWidth)x\(frameHeight)"
        }
        return nil
    }

    /// Port of `PixelCrop.principalPoint`.
    ///
    /// The fallback principal point belongs to the ORIGINAL stream and is never
    /// re-centred after cropping; a crop translates the optical axis, it does
    /// not move it. A valid axis may end up outside a retained off-axis ROI.
    func principalPoint(
        streamWidth: Int,
        streamHeight: Int,
        cx: Float,
        cy: Float,
        estimated: Bool
    ) -> (x: Float, y: Float) {
        let baseX = estimated ? (Float(streamWidth) - 1) * 0.5 : cx
        let baseY = estimated ? (Float(streamHeight) - 1) * 0.5 : cy
        return (baseX - Float(x), baseY - Float(y))
    }
}

/// Port of `CropPercent` in `CameraCrop.kt`.
///
/// Percentages are insets in the displayed (clockwise-rotated, not mirrored)
/// frame. `pixels(width:height:rotation:)` inverse-maps them into unrotated
/// stream coordinates so the phone never has to rotate or copy a whole frame
/// just to crop it.
struct CropPercent: Hashable {
    let left: Double
    let right: Double
    let top: Double
    let bottom: Double

    static let none = CropPercent(left: 0, right: 0, top: 0, bottom: 0)!

    /// Port of the Kotlin `init { require(...) }` block.
    init?(left: Double, right: Double, top: Double, bottom: Double) {
        let values = [left, right, top, bottom]
        guard values.allSatisfy({ $0.isFinite && $0 >= 0.0 && $0 <= 45.0 }) else { return nil }
        self.left = left
        self.right = right
        self.top = top
        self.bottom = bottom
    }

    /// Port of `CropPercent.pixels`.
    ///
    /// `rotation` is the clockwise rotation the PC will apply to the received
    /// frame. Insets are therefore rotated into raw stream coordinates first.
    func pixels(width: Int, height: Int, rotation: Int) -> PixelCrop {
        precondition(width >= 4 && height >= 4 && width % 2 == 0 && height % 2 == 0,
                     "stream dimensions must be even and at least 4x4")

        let raw: CropPercent
        switch rotation {
        case 0:
            raw = self
        case 90:
            raw = CropPercent(left: top, right: bottom, top: right, bottom: left)!
        case 180:
            raw = CropPercent(left: right, right: left, top: bottom, bottom: top)!
        case 270:
            raw = CropPercent(left: bottom, right: top, top: left, bottom: right)!
        default:
            preconditionFailure("rotation must be 0, 90, 180 or 270")
        }

        // Kotlin: ((size * percent / 100).roundToInt() and -2).coerceIn(0, size - 4)
        func inset(_ size: Int, _ percent: Double) -> Int {
            let rounded = Int((Double(size) * percent / 100.0).rounded())
            let even = rounded & -2
            return min(max(even, 0), size - 4)
        }

        let x = inset(width, raw.left)
        let y = inset(height, raw.top)
        let right = min(inset(width, raw.right), width - x - 4)
        let bottom = min(inset(height, raw.bottom), height - y - 4)
        return PixelCrop(x: x, y: y, width: width - x - right, height: height - y - bottom)
    }
}

/// Port of `JpegQuality` in `CameraCrop.kt`.
enum JpegQuality {
    /// The Android tiers, expressed as CoreImage's 0...1 quality.
    static let levels: [Int] = [50, 65, 80, 90, 95, 100]
    static let `default` = 80

    static func normalized(_ percent: Int) -> Double {
        Double(percent) / 100.0
    }
}
