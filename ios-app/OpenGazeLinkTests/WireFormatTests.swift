import XCTest
@testable import OpenGazeLink

/// Golden-byte tests against the PC's `struct.Struct` definitions.
///
/// Every expected array below was produced by Python's `struct.pack` with the
/// exact format string the PC unpacks with, so a change to any field width or
/// byte order fails here instead of silently desynchronising the receiver.
final class WireFormatTests: XCTestCase {

    // MARK: - UDP frame header, camera.py HEADER = struct.Struct("<IHHIHHHHBBQQI")

    func testFrameHeaderLayoutMatchesPCStruct() {
        let bytes = WireFormat.frameHeader(
            sequence: 1,
            chunkIndex: 0,
            chunkCount: 1,
            width: 1280,
            height: 720,
            format: WireFormat.formatJPEG,
            flags: 0,
            sensorTimeNs: 0x0102_0304_0506_0708,
            frameSendTimeNs: 0x1112_1314_1516_1718,
            payloadSize: 3
        )
        let expected: [UInt8] = [
            0x45, 0x59, 0x55, 0x56, 0x01, 0x00, 0x2A, 0x00,
            0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00,
            0x00, 0x05, 0xD0, 0x02, 0x02, 0x00,
            0x08, 0x07, 0x06, 0x05, 0x04, 0x03, 0x02, 0x01,
            0x18, 0x17, 0x16, 0x15, 0x14, 0x13, 0x12, 0x11,
            0x03, 0x00, 0x00, 0x00,
        ]
        XCTAssertEqual(bytes.count, 42)
        XCTAssertEqual(bytes, expected)
    }

    func testFrameHeaderSizeIsWhatThePCValidates() {
        let bytes = WireFormat.frameHeader(
            sequence: 0, chunkIndex: 0, chunkCount: 1, width: 1280, height: 720,
            format: WireFormat.formatJPEG, sensorTimeNs: 0, frameSendTimeNs: 0, payloadSize: 0
        )
        XCTAssertEqual(bytes.count, WireFormat.frameHeaderSize)
        // camera.py:347 rejects the packet unless header_size == HEADER.size.
        XCTAssertEqual(UInt16(bytes[6]) | (UInt16(bytes[7]) << 8), UInt16(42))
        // version == 1
        XCTAssertEqual(UInt16(bytes[4]) | (UInt16(bytes[5]) << 8), 1)
    }

    // MARK: - Intrinsics envelope, camera.py INTRINSICS_HEADER = struct.Struct("<IHHI")

    func testIntrinsicsEnvelopeLayoutMatchesPCStruct() {
        let bytes = WireFormat.intrinsicsEnvelope(json: [UInt8](repeating: 0x41, count: 9))
        let expected: [UInt8] = [0x45, 0x59, 0x43, 0x49, 0x01, 0x00, 0x0C, 0x00, 0x09, 0x00, 0x00, 0x00]
        XCTAssertEqual(bytes.count, 12)
        XCTAssertEqual(Array(bytes[0..<12]), expected)
        XCTAssertEqual(bytes.count, WireFormat.intrinsicsHeaderSize + 9)
    }

    // MARK: - TCP AVC header, h264_stream.py AVC_HEADER = struct.Struct('<4sIIHHQQQI')

    func testAvcHeaderLayoutMatchesPCStruct() {
        let bytes = WireFormat.avcHeader(
            sequence: 7,
            flags: 3,
            width: 1280,
            height: 720,
            sensorTimeNs: 0x0102_0304_0506_0708,
            encodedTimeNs: 0x1112_1314_1516_1718,
            phoneSendTimeNs: 0x2122_2324_2526_2728,
            payloadSize: 5
        )
        let expected: [UInt8] = [
            0x41, 0x56, 0x43, 0x31, 0x07, 0x00, 0x00, 0x00,
            0x03, 0x00, 0x00, 0x00, 0x00, 0x05, 0xD0, 0x02,
            0x08, 0x07, 0x06, 0x05, 0x04, 0x03, 0x02, 0x01,
            0x18, 0x17, 0x16, 0x15, 0x14, 0x13, 0x12, 0x11,
            0x28, 0x27, 0x26, 0x25, 0x24, 0x23, 0x22, 0x21,
            0x05, 0x00, 0x00, 0x00,
        ]
        XCTAssertEqual(bytes.count, 44)
        XCTAssertEqual(bytes, expected)
        XCTAssertEqual(bytes.count, WireFormat.avcHeaderSize)
        XCTAssertEqual(Array(bytes[0..<4]), WireFormat.avcMagic)
    }

    // MARK: - Clock probe, transport_clock.py CLOCK_PACKET = struct.Struct("<IHHQQQ")

    func testClockReplyLayoutMatchesPCStruct() {
        let bytes = WireFormat.clockReply(
            t1: 0x0102_0304_0506_0708,
            t2: 0x1112_1314_1516_1718,
            t3: 0x2122_2324_2526_2728
        )
        let expected: [UInt8] = [
            0x45, 0x59, 0x43, 0x54, 0x01, 0x00, 0x02, 0x00,
            0x08, 0x07, 0x06, 0x05, 0x04, 0x03, 0x02, 0x01,
            0x18, 0x17, 0x16, 0x15, 0x14, 0x13, 0x12, 0x11,
            0x28, 0x27, 0x26, 0x25, 0x24, 0x23, 0x22, 0x21,
        ]
        XCTAssertEqual(bytes.count, 32)
        XCTAssertEqual(bytes, expected)
        XCTAssertEqual(bytes.count, WireFormat.clockPacketSize)
    }

    func testClockRequestParsingRejectsAnythingThePCWouldNotSend() {
        // The PC sends kind 1 with t2 and t3 zeroed (transport_clock.py:16).
        var request: [UInt8] = [
            0x45, 0x59, 0x43, 0x54, 0x01, 0x00, 0x01, 0x00,
            0x08, 0x07, 0x06, 0x05, 0x04, 0x03, 0x02, 0x01,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        ]
        let parsed = WireFormat.parseClockRequest(request)
        XCTAssertEqual(parsed?.t1, 0x0102_0304_0506_0708)

        // Wrong kind: a reply must never be answered with a reply.
        request[6] = 2
        XCTAssertNil(WireFormat.parseClockRequest(request))
        request[6] = 1

        // Wrong magic.
        request[0] = 0x00
        XCTAssertNil(WireFormat.parseClockRequest(request))
        request[0] = 0x45

        // Short packet.
        XCTAssertNil(WireFormat.parseClockRequest(Array(request[0..<16])))
    }

    // MARK: - Discovery payloads

    func testDiscoveryQueryMatchesThePCMagicAndKeys() {
        let query = WireFormat.discoveryQuery(
            phoneID: "phone-1", phoneName: "iPhone 16 Pro Max",
            instanceID: "instance-9", nonce: "nonce-3"
        )
        XCTAssertEqual(query["magic"] as? String, "EYETRACING_DISCOVERY_V1")
        XCTAssertEqual(query["type"] as? String, "discover")
        XCTAssertEqual(query["version"] as? Int, 1)
        XCTAssertEqual(query["phone_id"] as? String, "phone-1")
        XCTAssertEqual(query["phone_name"] as? String, "iPhone 16 Pro Max")
        XCTAssertEqual(query["instance_id"] as? String, "instance-9")
        XCTAssertEqual(query["nonce"] as? String, "nonce-3")
        XCTAssertEqual(WireFormat.discoveryPort, 5006)
    }

    private func offerJSON(
        nonce: String = "n1",
        instanceID: String = "i1",
        dataPort: Int = 5007,
        accepted: Bool = true,
        type: String = "offer",
        version: Int = 1,
        magic: String = "EYETRACING_DISCOVERY_V1"
    ) -> Data {
        let object: [String: Any] = [
            "magic": magic,
            "type": type,
            "version": version,
            "nonce": nonce,
            "instance_id": instanceID,
            "pc_name": "DESKTOP",
            "data_port": dataPort,
            "accepted": accepted,
        ]
        return try! JSONSerialization.data(withJSONObject: object)
    }

    func testOfferParsingAcceptsAValidReply() {
        let offer = WireFormat.parseDiscoveryOffer(offerJSON(), nonce: "n1", expectedInstanceID: "i1")
        XCTAssertEqual(offer?.instanceID, "i1")
        XCTAssertEqual(offer?.pcName, "DESKTOP")
        XCTAssertEqual(offer?.dataPort, 5007)
        XCTAssertEqual(offer?.accepted, true)
    }

    func testOfferParsingRejectsAMismatchedNonce() {
        // MainActivity.kt:628 requires the nonce echo, which is what stops a
        // stale offer from a previous discovery run being accepted.
        XCTAssertNil(WireFormat.parseDiscoveryOffer(offerJSON(), nonce: "other", expectedInstanceID: "i1"))
    }

    func testOfferParsingRejectsAChangedInstanceWhenOneIsStored() {
        XCTAssertNil(WireFormat.parseDiscoveryOffer(offerJSON(), nonce: "n1", expectedInstanceID: "different"))
        // An empty stored instance accepts any PC, matching the Android client.
        XCTAssertNotNil(WireFormat.parseDiscoveryOffer(offerJSON(), nonce: "n1", expectedInstanceID: ""))
    }

    func testOfferParsingRejectsMalformedReplies() {
        XCTAssertNil(WireFormat.parseDiscoveryOffer(offerJSON(type: "discover"), nonce: "n1", expectedInstanceID: ""))
        XCTAssertNil(WireFormat.parseDiscoveryOffer(offerJSON(version: 2), nonce: "n1", expectedInstanceID: ""))
        XCTAssertNil(WireFormat.parseDiscoveryOffer(offerJSON(magic: "OTHER"), nonce: "n1", expectedInstanceID: ""))
        XCTAssertNil(WireFormat.parseDiscoveryOffer(offerJSON(dataPort: 0), nonce: "n1", expectedInstanceID: ""))
        XCTAssertNil(WireFormat.parseDiscoveryOffer(offerJSON(dataPort: 70000), nonce: "n1", expectedInstanceID: ""))
        XCTAssertNil(WireFormat.parseDiscoveryOffer(offerJSON(instanceID: ""), nonce: "n1", expectedInstanceID: ""))
        XCTAssertNil(WireFormat.parseDiscoveryOffer(Data([0x00, 0x01]), nonce: "n1", expectedInstanceID: ""))
    }

    // MARK: - H.264 geometry gate

    func testOnlyThePCSupportedH264GeometryIsAccepted() {
        XCTAssertTrue(WireFormat.acceptsH264(width: 1280, height: 720))
        // h264_stream.py:205 rejects every other pair before decoding.
        XCTAssertFalse(WireFormat.acceptsH264(width: 1920, height: 1080))
        XCTAssertFalse(WireFormat.acceptsH264(width: 3840, height: 2160))
        XCTAssertFalse(WireFormat.acceptsH264(width: 720, height: 1280))
    }

    func testAVCFlagValuesMatchMediaCodecBufferFlags() {
        // AvcEncoder.kt forwards MediaCodec's BufferInfo flags verbatim, and
        // h264_stream.py tests the same bit values.
        XCTAssertEqual(WireFormat.avcFlagKeyframe, 1)
        XCTAssertEqual(WireFormat.avcFlagCodecConfig, 2)
        XCTAssertEqual(WireFormat.avcFlagPartialFrame, 8)
    }

    // MARK: - Byte helpers

    func testByteReaderRoundTripsEveryWidth() {
        var writer = ByteWriter()
        writer.putUInt8(0xAB)
        writer.putUInt16(0x1234)
        writer.putUInt32(0xDEAD_BEEF)
        writer.putUInt64(0x0102_0304_0506_0708)

        var reader = ByteReader(writer.bytes)
        XCTAssertEqual(reader.readUInt8(), 0xAB)
        XCTAssertEqual(reader.readUInt16(), 0x1234)
        XCTAssertEqual(reader.readUInt32(), 0xDEAD_BEEF)
        XCTAssertEqual(reader.readUInt64(), 0x0102_0304_0506_0708)
        XCTAssertEqual(reader.remaining, 0)
        XCTAssertNil(reader.readUInt8())
    }

    func testByteReaderRefusesPartialReads() {
        var reader = ByteReader([0x01, 0x02, 0x03])
        XCTAssertNil(reader.readUInt32())
        XCTAssertEqual(reader.remaining, 3)
    }
}
