import XCTest
@testable import OpenGazeLink

/// VideoToolbox emits AVCC access units; the PC's PyAV decoder has no extradata
/// and therefore parses Annex-B only (h264_stream.py:31,44). These tests pin the
/// conversion that makes the two agree.
final class AnnexBTests: XCTestCase {

    func testConvertsLengthPrefixedAccessUnitToAnnexB() {
        // IDR (0x65 -> type 5) then a non-IDR slice (0x41 -> type 1).
        let avcc: [UInt8] = [
            0x00, 0x00, 0x00, 0x03, 0x65, 0x11, 0x22,
            0x00, 0x00, 0x00, 0x02, 0x41, 0x33,
        ]
        let expected: [UInt8] = [
            0x00, 0x00, 0x00, 0x01, 0x65, 0x11, 0x22,
            0x00, 0x00, 0x00, 0x01, 0x41, 0x33,
        ]
        XCTAssertEqual(AnnexB.convert(avcc: avcc, nalUnitHeaderLength: 4), expected)
    }

    func testConvertsTwoByteLengthPrefixes() {
        let avcc: [UInt8] = [0x00, 0x03, 0x65, 0x11, 0x22, 0x00, 0x02, 0x41, 0x33]
        let expected: [UInt8] = [
            0x00, 0x00, 0x00, 0x01, 0x65, 0x11, 0x22,
            0x00, 0x00, 0x00, 0x01, 0x41, 0x33,
        ]
        XCTAssertEqual(AnnexB.convert(avcc: avcc, nalUnitHeaderLength: 2), expected)
    }

    func testConvertsOneByteLengthPrefixes() {
        let avcc: [UInt8] = [0x03, 0x65, 0x11, 0x22]
        XCTAssertEqual(AnnexB.convert(avcc: avcc, nalUnitHeaderLength: 1),
                       [0x00, 0x00, 0x00, 0x01, 0x65, 0x11, 0x22])
    }

    func testRejectsATruncatedAccessUnit() {
        // Declared length runs past the buffer: emitting it would desynchronise
        // the PC decoder, so the frame must be dropped instead.
        XCTAssertNil(AnnexB.convert(avcc: [0x00, 0x00, 0x00, 0x09, 0x65, 0x11], nalUnitHeaderLength: 4))
        // Trailing partial length field.
        XCTAssertNil(AnnexB.convert(avcc: [0x00, 0x00, 0x00, 0x01, 0x65, 0x00, 0x00], nalUnitHeaderLength: 4))
        // Zero-length NAL unit.
        XCTAssertNil(AnnexB.convert(avcc: [0x00, 0x00, 0x00, 0x00], nalUnitHeaderLength: 4))
    }

    func testRejectsAnUnsupportedNALHeaderLength() {
        XCTAssertNil(AnnexB.convert(avcc: [0x00, 0x03, 0x65], nalUnitHeaderLength: 3))
        XCTAssertNil(AnnexB.convert(avcc: [0x00, 0x00, 0x00, 0x00, 0x03, 0x65], nalUnitHeaderLength: 8))
    }

    func testEmptyAccessUnitConvertsToNothing() {
        XCTAssertEqual(AnnexB.convert(avcc: [], nalUnitHeaderLength: 4), [])
    }

    func testCodecConfigIsAnnexBFramedSPSThenPPS() {
        let sps: [UInt8] = [0x67, 0x42, 0x00, 0x1E]
        let pps: [UInt8] = [0x68, 0xCE, 0x38, 0x80]
        XCTAssertEqual(
            AnnexB.codecConfig(sps: sps, pps: pps),
            [0x00, 0x00, 0x00, 0x01, 0x67, 0x42, 0x00, 0x1E,
             0x00, 0x00, 0x00, 0x01, 0x68, 0xCE, 0x38, 0x80]
        )
    }

    func testNALTypesAreReadInStreamOrder() {
        let stream: [UInt8] = [
            0x00, 0x00, 0x00, 0x01, 0x67, 0xAA,      // SPS
            0x00, 0x00, 0x00, 0x01, 0x68, 0xBB,      // PPS
            0x00, 0x00, 0x00, 0x01, 0x65, 0xCC,      // IDR
            0x00, 0x00, 0x00, 0x01, 0x41, 0xDD,      // non-IDR
        ]
        XCTAssertEqual(AnnexB.nalTypes(in: stream), [7, 8, 5, 1])
    }

    func testNALTypesAcceptsThreeByteStartCodes() {
        let stream: [UInt8] = [0x00, 0x00, 0x01, 0x65, 0xAA, 0x00, 0x00, 0x01, 0x41, 0xBB]
        XCTAssertEqual(AnnexB.nalTypes(in: stream), [5, 1])
    }

    func testDetectsIDR() {
        // The PC treats flags & 1 as "keyframe": it prepends the stored SPS/PPS
        // and uses the keyframe to resynchronise after a decode queue overflow.
        let withIDR: [UInt8] = [0x00, 0x00, 0x00, 0x01, 0x65, 0xAA, 0x00, 0x00, 0x00, 0x01, 0x41, 0xBB]
        XCTAssertTrue(AnnexB.containsIDR(annexB: withIDR))

        let withoutIDR: [UInt8] = [0x00, 0x00, 0x00, 0x01, 0x41, 0xBB]
        XCTAssertFalse(AnnexB.containsIDR(annexB: withoutIDR))

        // An SEI-only prefix followed by a non-IDR slice is still not a keyframe.
        let seiThenSlice: [UInt8] = [0x00, 0x00, 0x00, 0x01, 0x06, 0x05, 0x00, 0x00, 0x00, 0x01, 0x41, 0xBB]
        XCTAssertFalse(AnnexB.containsIDR(annexB: seiThenSlice))
    }

    func testNALHeaderTypeMasksTheLowFiveBits() {
        // forbidden_zero_bit | nal_ref_idc | type
        XCTAssertEqual(AnnexB.NALType(headerByte: 0x65), .idr)      // ref_idc 3, type 5
        XCTAssertEqual(AnnexB.NALType(headerByte: 0x25), .idr)      // ref_idc 1, type 5
        XCTAssertEqual(AnnexB.NALType(headerByte: 0x67), .sps)
        XCTAssertEqual(AnnexB.NALType(headerByte: 0x68), .pps)
        XCTAssertEqual(AnnexB.NALType(headerByte: 0x41), .nonIDR)
        XCTAssertNil(AnnexB.NALType(headerByte: 0x00))              // type 0 is unspecified
    }

    func testRoundTripOfARealisticKeyframeShape() {
        // SPS + PPS + IDR as VideoToolbox would hand it over after conversion.
        let avcc: [UInt8] = [
            0x00, 0x00, 0x00, 0x04, 0x65, 0x88, 0x84, 0x00,
            0x00, 0x00, 0x00, 0x02, 0x41, 0x9A,
        ]
        guard let annexB = AnnexB.convert(avcc: avcc, nalUnitHeaderLength: 4) else {
            return XCTFail("conversion failed")
        }
        XCTAssertTrue(AnnexB.containsIDR(annexB: annexB))
        XCTAssertEqual(AnnexB.nalTypes(in: annexB), [5, 1])
        // Every NAL unit must be reachable through a start code.
        XCTAssertEqual(AnnexB.nalTypes(in: annexB).count, 2)
    }
}
