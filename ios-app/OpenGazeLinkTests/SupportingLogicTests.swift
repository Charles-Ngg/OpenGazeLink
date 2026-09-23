import XCTest
@testable import OpenGazeLink

/// Rotation, subnet arithmetic and metric accounting: the three pieces of pure
/// logic that decide whether the PC receives a correct stream.
final class SupportingLogicTests: XCTestCase {

    // MARK: - FrameRotation

    func testRotationNormalisesToTheFourValuesThePCAccepts() {
        // camera.py:228 falls back to 270 for anything else, so a value outside
        // this set would silently rotate the stream the wrong way.
        XCTAssertEqual(FrameRotation.supported, [0, 90, 180, 270])
        XCTAssertEqual(FrameRotation.normalize(degrees: 0), 0)
        XCTAssertEqual(FrameRotation.normalize(degrees: 90), 90)
        XCTAssertEqual(FrameRotation.normalize(degrees: 180), 180)
        XCTAssertEqual(FrameRotation.normalize(degrees: 270), 270)
        XCTAssertEqual(FrameRotation.normalize(degrees: 360), 0)
        XCTAssertEqual(FrameRotation.normalize(degrees: -90), 270)
        XCTAssertEqual(FrameRotation.normalize(degrees: -180), 180)
        XCTAssertEqual(FrameRotation.normalize(degrees: 89.9), 90)
        XCTAssertEqual(FrameRotation.normalize(degrees: 91.0), 90)
        XCTAssertEqual(FrameRotation.normalize(degrees: 44.9), 0)
    }

    func testRotationRejectsNonFiniteAngles() {
        XCTAssertEqual(FrameRotation.normalize(degrees: Double.nan), 0)
        XCTAssertEqual(FrameRotation.normalize(degrees: Double.infinity), 0)
        XCTAssertEqual(FrameRotation.normalize(degrees: -Double.infinity), 0)
    }

    // MARK: - LocalNetwork

    private func interface(_ name: String, _ address: String, _ netmask: String) -> LocalNetwork.Interface {
        LocalNetwork.Interface(name: name, address: address, netmask: netmask, broadcast: "")
    }

    func testSweepCoversTheTwentyFourBitSliceExactlyOnce() {
        let candidates = LocalNetwork.sweepCandidates(
            over: [interface("en0", "192.168.50.222", "255.255.255.0")]
        )
        XCTAssertEqual(candidates.count, 253)
        XCTAssertEqual(candidates.first, "192.168.50.1")
        XCTAssertEqual(candidates.last, "192.168.50.254")
        XCTAssertFalse(candidates.contains("192.168.50.222"), "the phone's own address must not be probed")
        XCTAssertFalse(candidates.contains("192.168.50.255"), "the broadcast address must not be probed")
        XCTAssertFalse(candidates.contains("192.168.50.0"), "the network address must not be probed")
        XCTAssertEqual(Set(candidates).count, candidates.count, "no duplicates")
    }

    func testSweepHonoursANarrowHotspotNetmask() {
        // iOS Personal Hotspot hands the phone 172.20.10.1/28 and the tethered
        // PC one of 172.20.10.2 ... 172.20.10.14.
        let candidates = LocalNetwork.sweepCandidates(
            over: [interface("bridge100", "172.20.10.1", "255.255.255.240")]
        )
        XCTAssertEqual(candidates, (2...14).map { "172.20.10.\($0)" })
    }

    func testSweepNeverEscapesTheInterfacesOwnNetwork() {
        // A /16 must not produce 65k datagrams; only the /24 slice is swept.
        let candidates = LocalNetwork.sweepCandidates(
            over: [interface("en0", "10.5.7.9", "255.255.0.0")]
        )
        XCTAssertEqual(candidates.count, 253)
        XCTAssertTrue(candidates.allSatisfy { $0.hasPrefix("10.5.7.") })
        XCTAssertFalse(candidates.contains("10.5.7.9"))
    }

    func testSweepSkipsLinkLocalInterfaces() {
        let candidates = LocalNetwork.sweepCandidates(
            over: [interface("en0", "169.254.10.5", "255.255.0.0")]
        )
        XCTAssertTrue(candidates.isEmpty)
    }

    func testSweepDeduplicatesAcrossInterfaces() {
        let candidates = LocalNetwork.sweepCandidates(over: [
            interface("en0", "192.168.1.10", "255.255.255.0"),
            interface("en1", "192.168.1.11", "255.255.255.0"),
        ])
        XCTAssertEqual(Set(candidates).count, candidates.count)
        // Both of the phone's own addresses are excluded, not just the one
        // belonging to the interface being swept.
        XCTAssertFalse(candidates.contains("192.168.1.10"))
        XCTAssertFalse(candidates.contains("192.168.1.11"))
        XCTAssertEqual(candidates.count, 252)
    }

    func testBroadcastAddressesExcludeLimitedAndPointToPoint() {
        let addresses = LocalNetwork.broadcastAddresses([
            interface("en0", "192.168.50.222", "255.255.255.0"),
            interface("utun3", "10.0.0.5", "255.255.255.255"),
        ])
        XCTAssertEqual(addresses, ["192.168.50.255"])
    }

    func testBroadcastAddressesAreDeduplicated() {
        let addresses = LocalNetwork.broadcastAddresses([
            interface("en0", "192.168.50.10", "255.255.255.0"),
            interface("en1", "192.168.50.11", "255.255.255.0"),
        ])
        XCTAssertEqual(addresses, ["192.168.50.255"])
    }

    func testUsableLANAddressFilter() {
        XCTAssertTrue(SocketSupport.isUsableLANAddress("192.168.50.222"))
        XCTAssertTrue(SocketSupport.isUsableLANAddress("10.0.0.5"))
        XCTAssertTrue(SocketSupport.isUsableLANAddress("172.20.10.2"))
        XCTAssertFalse(SocketSupport.isUsableLANAddress("127.0.0.1"))
        XCTAssertFalse(SocketSupport.isUsableLANAddress("0.0.0.0"))
        XCTAssertFalse(SocketSupport.isUsableLANAddress("224.0.0.1"))
        XCTAssertFalse(SocketSupport.isUsableLANAddress("255.255.255.255"))
        XCTAssertFalse(SocketSupport.isUsableLANAddress("169.254.1.1"))
        XCTAssertFalse(SocketSupport.isUsableLANAddress("not-an-address"))
    }

    func testIPv4AddressRejectsHostNames() {
        XCTAssertNotNil(SocketSupport.ipv4Address(host: "192.168.50.222", port: 5007))
        // inet_pton only accepts literals; names are resolved explicitly so a
        // DNS stall can never block the capture queue.
        XCTAssertNil(SocketSupport.ipv4Address(host: "desktop.local", port: 5007))
    }

    // MARK: - StreamStatistics

    func testStatisticsMatchTheAndroidByteAccounting() {
        var statistics = StreamStatistics()
        let start: UInt64 = 1_000_000_000
        XCTAssertEqual(statistics.sample(nowNs: start), .empty, "the first sample only primes the interval")

        statistics.recordCapture()
        statistics.recordCapture()
        // A frame packet, then a codec-configuration packet: the config packet
        // adds bytes but must not count as a sent frame.
        statistics.recordSent(bytes: 1044, countsAsFrame: true)
        statistics.recordSent(bytes: 100, countsAsFrame: false)
        statistics.recordEncodingAge(ms: 4.0)
        statistics.recordExposure(ms: 8.0)

        let rates = statistics.sample(nowNs: start + 1_000_000_000)
        XCTAssertEqual(rates.captureFPS, 2.0, accuracy: 1e-9)
        XCTAssertEqual(rates.sentFPS, 1.0, accuracy: 1e-9)
        XCTAssertEqual(rates.megabitsPerSecond, 1144.0 * 8.0 / 1_000_000.0, accuracy: 1e-9)
        XCTAssertEqual(rates.encodingAgeMs ?? 0, 4.0, accuracy: 1e-9)
        XCTAssertEqual(rates.exposureMs ?? 0, 8.0, accuracy: 1e-9)
    }

    func testSecondIntervalStartsFromZero() {
        var statistics = StreamStatistics()
        _ = statistics.sample(nowNs: 1_000_000_000)
        statistics.recordCapture()
        _ = statistics.sample(nowNs: 2_000_000_000)

        let idle = statistics.sample(nowNs: 3_000_000_000)
        XCTAssertEqual(idle.captureFPS, 0.0, accuracy: 1e-9)
        XCTAssertEqual(idle.sentFPS, 0.0, accuracy: 1e-9)
        XCTAssertEqual(idle.megabitsPerSecond, 0.0, accuracy: 1e-9)
        XCTAssertNil(idle.encodingAgeMs, "no samples means no average, not zero")
        XCTAssertNil(idle.exposureMs)
    }

    func testStatisticsIgnoreImplausibleSamples() {
        var statistics = StreamStatistics()
        _ = statistics.sample(nowNs: 1_000_000_000)
        statistics.recordEncodingAge(ms: -1.0)
        statistics.recordEncodingAge(ms: Double.nan)
        statistics.recordExposure(ms: Double.infinity)
        statistics.recordSent(bytes: -5, countsAsFrame: true)

        let rates = statistics.sample(nowNs: 2_000_000_000)
        XCTAssertNil(rates.encodingAgeMs)
        XCTAssertNil(rates.exposureMs)
        XCTAssertEqual(rates.sentFPS, 1.0, accuracy: 1e-9)
        XCTAssertEqual(rates.megabitsPerSecond, 0.0, accuracy: 1e-9)
    }

    // MARK: - DeviceIdentity

    func testPhoneNameIsNeverEmpty() {
        // pairing.py:83 defaults a missing phone_name to "Android phone", so an
        // empty name would mislabel the device in the PC control centre.
        let name = DeviceIdentity.phoneName
        XCTAssertFalse(name.isEmpty)
        XCTAssertFalse(DeviceIdentity.machineIdentifier.isEmpty)
    }

    // MARK: - Localization

    func testEveryKeyHasAnEnglishAndChineseString() {
        let l10nKeys: [L10n.Key] = [
            .connectionTitle, .hostLabel, .portLabel, .findPC, .captureTitle,
            .modeHighSpeed, .modeCamera, .cameraLabel, .resolutionLabel, .frameRateLabel,
            .rotationLabel, .cropLabel, .jpegQualityLabel, .intrinsicsTitle,
            .startStream, .stopStream, .performanceTitle, .languageLabel,
        ]
        for key in l10nKeys {
            for language in AppLanguage.allCases {
                let text = L10n(language: language).text(key)
                XCTAssertFalse(text.isEmpty, "\(key.rawValue) is empty for \(language.rawValue)")
                XCTAssertNotEqual(text, key.rawValue, "\(key.rawValue) is missing for \(language.rawValue)")
            }
        }
    }

    func testLocalizedFormattingMatchesTheAndroidPlaceholderOrder() {
        let text = L10n(language: .english).text(.paired, "DESKTOP", "192.168.50.68", 5007)
        XCTAssertEqual(text, "Paired with DESKTOP · 192.168.50.68:5007")
    }
}
