import XCTest
@testable import OpenGazeLink

/// Port of `CaptureOptionTest.kt` and `CaptureSelectionTest.kt`, adapted to the
/// two deliberate iOS differences documented on `CaptureOptions.ordered`:
/// the high-speed gate is the PC's accepted geometry instead of a hardware
/// encoder query, and the rate floor is 60 FPS instead of 120 so the 60 FPS
/// fallback the user asked for is reachable.
final class CaptureOptionTests: XCTestCase {

    private func high(_ width: Int, _ height: Int, _ fps: Int, low: Int? = nil) -> CaptureOption {
        CaptureOption(cameraID: "front", mode: .highSpeed, width: width, height: height,
                      fps: fps, fpsLower: low ?? fps)
    }

    private func camera(_ width: Int, _ height: Int, _ fps: Int, low: Int? = nil) -> CaptureOption {
        CaptureOption(cameraID: "front", mode: .camera, width: width, height: height,
                      fps: fps, fpsLower: low ?? fps)
    }

    // MARK: - Filtering

    func testHighSpeedRequiresThePCSizeAndAtLeastSixtyFPS() {
        let accepted = high(1280, 720, 120)
        let options = CaptureOptions.ordered([
            accepted,
            high(1920, 1080, 120),   // the PC rejects anything but 1280x720
            high(640, 480, 120),     // same
            high(1280, 720, 30),     // below the 60 FPS floor
            high(641, 720, 120),     // odd width
            high(1280, 720, 120, low: 240),  // impossible range
        ])
        XCTAssertEqual(options, [accepted])
    }

    func testHighSpeedKeepsTheSixtyFPSFallback() {
        let options = CaptureOptions.ordered([high(1280, 720, 120), high(1280, 720, 60)])
        XCTAssertEqual(Set(options.map(\.fps)), [120, 60])
        XCTAssertEqual(options.first?.fps, 120)
    }

    func testCameraModeCapsAtSixtyFPS() {
        let sixty = camera(1280, 720, 60)
        let thirty = camera(1280, 720, 30)
        let options = CaptureOptions.ordered([camera(1280, 720, 120), sixty, thirty])
        XCTAssertEqual(Set(options), Set([sixty, thirty]))
        XCTAssertFalse(options.contains { $0.fps > 60 })
    }

    func testCameraModeIsNotRestrictedToThePCH264Size() {
        // The UDP JPEG path is dimension agnostic, so 1080p and 4K stay usable
        // exactly where they are actually useful.
        let options = CaptureOptions.ordered([
            camera(1920, 1080, 30), camera(3840, 2160, 30), camera(1280, 720, 30),
        ])
        XCTAssertEqual(options.count, 3)
    }

    func testPreferredRateIsOrderedFirstPerMode() {
        let cameraOptions = CaptureOptions.ordered([camera(1280, 720, 60), camera(1280, 720, 30)])
        XCTAssertEqual(cameraOptions.first?.fps, 30)
        let highOptions = CaptureOptions.ordered([high(1280, 720, 60), high(1280, 720, 120)])
        XCTAssertEqual(highOptions.first?.fps, 120)
    }

    func testClosestToTheModelResolutionWinsWhenThePreferredRateTies() {
        let options = CaptureOptions.ordered([camera(1920, 1080, 30), camera(1280, 720, 30)])
        XCTAssertEqual(options.first?.resolution, CaptureResolution(width: 1280, height: 720))
    }

    func testDuplicatesCollapseOnTheOptionKey() {
        let options = CaptureOptions.ordered([camera(1280, 720, 30), camera(1280, 720, 30), camera(1280, 720, 30)])
        XCTAssertEqual(options.count, 1)
    }

    func testDoNotInventSizeRateCrossProducts() {
        let declared = [camera(1920, 1080, 60), camera(1280, 720, 24)]
        let options = CaptureOptions.ordered(declared)
        XCTAssertEqual(Set(options), Set(declared))
        XCTAssertFalse(options.contains { $0.width == 1920 && $0.fps == 24 })
        XCTAssertFalse(options.contains { $0.width == 1280 && $0.fps == 60 })
    }

    // MARK: - Picker values

    func testPickerValuesAreUniqueAndSorted() {
        let options = [
            camera(1280, 720, 60), camera(1280, 720, 30),
            camera(1920, 1080, 60), camera(640, 480, 60), camera(1280, 720, 60),
        ]
        XCTAssertEqual(CaptureOptions.resolutions(options), [
            CaptureResolution(width: 1920, height: 1080),
            CaptureResolution(width: 1280, height: 720),
            CaptureResolution(width: 640, height: 480),
        ])
        XCTAssertEqual(CaptureOptions.frameRates(options), [60, 30])
    }

    // MARK: - Selection

    func testChangingResolutionPreservesCompatibleFrameRate() {
        let current = camera(1280, 720, 60)
        let target = camera(1920, 1080, 60)
        let options = CaptureOptions.ordered([current, target, camera(1920, 1080, 30)])
        XCTAssertEqual(CaptureOptions.selectResolution(options, resolution: target.resolution, preferred: current), target)
    }

    func testChangingFrameRatePreservesCompatibleResolution() {
        let current = camera(1920, 1080, 30)
        let target = camera(1920, 1080, 60)
        let options = CaptureOptions.ordered([current, target, camera(1280, 720, 60)])
        XCTAssertEqual(CaptureOptions.selectFrameRate(options, fps: 60, preferred: current), target)
    }

    func testResolutionFallbackPrefersLowerRateOnTie() {
        // 24 and 60 are equidistant from 42, so the lower rate must win rather
        // than silently doubling the load.
        let current = camera(1280, 720, 42)
        let low = camera(1920, 1080, 24)
        let high = camera(1920, 1080, 60)
        XCTAssertEqual(CaptureOptions.selectResolution([high, low, current], resolution: low.resolution, preferred: current), low)
    }

    func testFrameRateFallbackPrefersSmallerSizeOnTie() {
        let current = camera(1280, 720, 30)
        let small = camera(1024, 720, 60)
        let large = camera(1920, 1080, 60)
        XCTAssertEqual(CaptureOptions.selectFrameRate([large, small, current], fps: 60, preferred: current), small)
    }

    func testSamePixelCountDoesNotConfusePortraitAndLandscape() {
        let current = camera(1080, 1920, 30)
        let target = camera(1080, 1920, 60)
        let options = [camera(1920, 1080, 60), target, current]
        XCTAssertEqual(CaptureOptions.selectFrameRate(options, fps: 60, preferred: current), target)
    }

    func testEitherPickerCanReachDisconnectedCombinations() {
        let largeSlow = camera(1920, 1080, 30)
        let smallFast = camera(1280, 720, 60)
        let options = [largeSlow, smallFast]
        XCTAssertEqual(CaptureOptions.selectFrameRate(options, fps: 60, preferred: largeSlow), smallFast)
        XCTAssertEqual(CaptureOptions.selectResolution(options, resolution: smallFast.resolution, preferred: largeSlow), smallFast)
        XCTAssertEqual(CaptureOptions.selectFrameRate(options, fps: 30, preferred: smallFast), largeSlow)
        XCTAssertEqual(CaptureOptions.selectResolution(options, resolution: largeSlow.resolution, preferred: smallFast), largeSlow)
        XCTAssertEqual(CaptureOptions.resolutions(options).count, 2)
        XCTAssertEqual(CaptureOptions.frameRates(options).count, 2)
    }

    func testEitherFieldCanBeSelectedWithoutAnExistingPreference() {
        let options = [camera(1920, 1080, 30), camera(1280, 720, 60)]
        XCTAssertEqual(CaptureOptions.selectFrameRate(options, fps: 60, preferred: nil), options[1])
        XCTAssertEqual(CaptureOptions.selectResolution(options, resolution: options[1].resolution, preferred: nil), options[1])
    }

    func testAbsentAndExcludedValuesNeverCreateUnsupportedCombinations() {
        let options = [camera(1920, 1080, 30)]
        let excluded = camera(1280, 720, 60)
        XCTAssertNil(CaptureOptions.selectFrameRate(options, fps: excluded.fps, preferred: options[0]))
        XCTAssertNil(CaptureOptions.selectResolution(options, resolution: excluded.resolution, preferred: options[0]))
        XCTAssertEqual(CaptureOptions.frameRates([]), [])
        XCTAssertEqual(CaptureOptions.resolutions([]), [])
        XCTAssertNil(CaptureOptions.selectFrameRate([], fps: 30, preferred: options[0]))
        XCTAssertNil(CaptureOptions.selectResolution([], resolution: options[0].resolution, preferred: options[0]))
    }

    func testSavedSelectionIsValidatedAgainstCurrentCapabilities() {
        let options = CaptureOptions.ordered([high(1280, 720, 120), camera(1280, 720, 60)])
        XCTAssertEqual(CaptureOptions.selected(options, savedKey: options[1].key), options[1])
        XCTAssertEqual(CaptureOptions.selected(options, savedKey: "stale:cropped:mode"), options.first)
        XCTAssertNil(CaptureOptions.selected([], savedKey: options[0].key))
    }

    func testHundredsOfPairsRemainReachableInBothOrdersWithoutInventingAny() {
        var declared: [CaptureOption] = []
        for size in 1...40 {
            for (index, fps) in [15, 24, 30, 60].enumerated() where (size + index) % 4 != 0 {
                declared.append(camera(640 + size * 16, 480 + size * 12, fps))
            }
        }
        let options = CaptureOptions.ordered(declared)
        XCTAssertGreaterThan(options.count, 100)
        let resolutions = CaptureOptions.resolutions(options)
        let frameRates = CaptureOptions.frameRates(options)

        for initial in [options.first!, options[options.count / 2], options.last!] {
            for target in options {
                guard let sizeFirst = CaptureOptions.selectResolution(options, resolution: target.resolution, preferred: initial),
                      let rateFirst = CaptureOptions.selectFrameRate(options, fps: target.fps, preferred: initial)
                else {
                    return XCTFail("selection returned nil for \(target.key)")
                }
                XCTAssertTrue(options.contains(sizeFirst))
                XCTAssertTrue(options.contains(rateFirst))
                XCTAssertEqual(sizeFirst.resolution, target.resolution)
                XCTAssertEqual(rateFirst.fps, target.fps)
                XCTAssertEqual(CaptureOptions.selectFrameRate(options, fps: target.fps, preferred: sizeFirst), target)
                XCTAssertEqual(CaptureOptions.selectResolution(options, resolution: target.resolution, preferred: rateFirst), target)
            }
        }
        XCTAssertEqual(resolutions.count, Set(resolutions).count)
        XCTAssertEqual(frameRates.count, Set(frameRates).count)
    }

    // MARK: - Derived properties

    func testOptionKeyAndLabelMatchTheAndroidShape() {
        let option = high(1280, 720, 120)
        XCTAssertEqual(option.key, "front:HIGH_SPEED:1280:720:120")
        XCTAssertEqual(option.label, "1280 × 720 · 120 FPS")
        XCTAssertTrue(option.isPCH264Compatible)
        XCTAssertFalse(camera(1920, 1080, 30).isPCH264Compatible)
    }

    func testResolutionSortsLargestFirstThenWidest() {
        let sorted = [CaptureResolution(width: 1280, height: 720),
                      CaptureResolution(width: 1920, height: 1080),
                      CaptureResolution(width: 640, height: 480)].sorted()
        XCTAssertEqual(sorted.map(\.width), [1920, 1280, 640])
    }
}
