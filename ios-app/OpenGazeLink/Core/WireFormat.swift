import Foundation

/// Single source of truth for every byte the PC provider parses.
///
/// Mirrors, field for field:
///   pc-provider/opengazelink_pc/camera.py          (UDP frame + intrinsics)
///   pc-provider/opengazelink_pc/h264_stream.py     (TCP AVC)
///   pc-provider/opengazelink_pc/transport_clock.py (clock probe)
///   pc-provider/opengazelink_pc/pairing.py         (discovery)
///   phone-app/.../UdpYuvSender.kt, AvcTcpSender.kt  (Android sender)
///
/// Nothing here may drift without a matching PC change; every constant below
/// is cross-checked by OpenGazeLinkTests/WireFormatTests.swift.
enum WireFormat {

    // MARK: - UDP frame packet: struct '<IHHIHHHHBBQQI', 42 bytes

    /// "EYUV" little-endian. camera.py: MAGIC
    static let frameMagic: UInt32 = 0x5655_5945
    static let frameVersion: UInt16 = 1
    static let frameHeaderSize = 42
    /// Android UdpYuvSender uses a 1400 byte chunk payload.
    static let frameChunkPayloadBytes = 1400

    /// camera.py: FORMAT_NV21 / FORMAT_JPEG
    static let formatNV21: UInt8 = 1
    static let formatJPEG: UInt8 = 2
    /// iOS delivers biplanar NV12 (UV order), which is not byte-compatible with
    /// the PC's NV21 reader, so the JPEG path is the only supported UDP format.
    static let supportedUDPFormats: Set<UInt8> = [formatJPEG]

    // MARK: - Intrinsics envelope: struct '<IHHI', 12 bytes

    /// "EYCI" little-endian. camera.py: INTRINSICS_MAGIC
    static let intrinsicsMagic: UInt32 = 0x4943_5945
    static let intrinsicsVersion: UInt16 = 1
    static let intrinsicsHeaderSize = 12

    // MARK: - Clock probe: struct '<IHHQQQ', 32 bytes

    /// "EYCT" little-endian. transport_clock.py: CLOCK_MAGIC
    static let clockMagic: UInt32 = 0x5443_5945
    static let clockVersion: UInt16 = 1
    static let clockRequestKind: UInt16 = 1
    static let clockReplyKind: UInt16 = 2
    static let clockPacketSize = 32

    // MARK: - TCP AVC packet: struct '<4sIIHHQQQI', 44 bytes

    /// h264_stream.py: AVC_HEADER magic b'AVC1'
    static let avcMagic: [UInt8] = Array("AVC1".utf8)
    static let avcHeaderSize = 44
    /// h264_stream.py: MAX_PACKET_BYTES
    static let maxAvcPacketBytes = 4 * 1024 * 1024

    /// h264_stream.py reads these exact bit values out of the header.
    /// They are also Android MediaCodec's BufferInfo flag values.
    static let avcFlagKeyframe: UInt32 = 1
    static let avcFlagCodecConfig: UInt32 = 2
    static let avcFlagPartialFrame: UInt32 = 8

    // MARK: - Discovery

    static let discoveryMagic = "EYETRACING_DISCOVERY_V1"
    static let discoveryVersion = 1
    static let discoveryPort: UInt16 = 5006
    /// Fallback only. The authoritative value is `data_port` in the PC offer.
    static let defaultDataPort: UInt16 = 5007
    static let discoveryPayloadType = "discover"
    static let discoveryOfferType = "offer"

    // MARK: - H.264 geometry gate

    /// The PC's H.264 receiver accepts exactly one capture geometry.
    ///
    /// h264_stream.py:205 rejects any packet whose header width/height is not
    /// (1280, 720) *before* decoding, and `_serve` verifies that the decoded
    /// frame size matches the header. Widening this set without editing the PC
    /// provider produces a stream that the PC silently refuses.
    ///
    /// The gaze model consumes a 64x36 px eye patch covering a 4.2 x 2.4 cm
    /// plane, i.e. 15.2 px/cm. A 720p frame already reaches that density for a
    /// face 40-60 cm away, so 1080p would only add oversampling while forcing
    /// 60 FPS. See ios-app/README.md for the full argument.
    static let acceptedH264Width = 1280
    static let acceptedH264Height = 720

    static func acceptsH264(width: Int, height: Int) -> Bool {
        width == acceptedH264Width && height == acceptedH264Height
    }
}

// MARK: - Little-endian serialisation

/// Explicit little-endian writer. The PC unpacks every header with
/// `struct.Struct("<...")`, so byte order must never depend on the host.
struct ByteWriter {
    private(set) var bytes: [UInt8]

    init(capacity: Int = 0) {
        bytes = []
        if capacity > 0 { bytes.reserveCapacity(capacity) }
    }

    var count: Int { bytes.count }

    mutating func putUInt8(_ value: UInt8) {
        bytes.append(value)
    }

    mutating func putUInt16(_ value: UInt16) {
        bytes.append(UInt8(truncatingIfNeeded: value))
        bytes.append(UInt8(truncatingIfNeeded: value >> 8))
    }

    mutating func putUInt32(_ value: UInt32) {
        for shift in stride(from: 0, to: 32, by: 8) {
            bytes.append(UInt8(truncatingIfNeeded: value >> UInt32(shift)))
        }
    }

    mutating func putUInt64(_ value: UInt64) {
        for shift in stride(from: 0, to: 64, by: 8) {
            bytes.append(UInt8(truncatingIfNeeded: value >> UInt64(shift)))
        }
    }

    mutating func putBytes(_ value: [UInt8]) {
        bytes.append(contentsOf: value)
    }
}

/// Little-endian reader used by the tests and by the clock-probe parser.
struct ByteReader {
    private let bytes: [UInt8]
    private var offset: Int

    init(_ bytes: [UInt8], offset: Int = 0) {
        self.bytes = bytes
        self.offset = offset
    }

    var remaining: Int { bytes.count - offset }

    mutating func readUInt8() -> UInt8? {
        guard remaining >= 1 else { return nil }
        defer { offset += 1 }
        return bytes[offset]
    }

    mutating func readUInt16() -> UInt16? {
        guard remaining >= 2 else { return nil }
        defer { offset += 2 }
        return UInt16(bytes[offset]) | (UInt16(bytes[offset + 1]) << 8)
    }

    mutating func readUInt32() -> UInt32? {
        guard remaining >= 4 else { return nil }
        defer { offset += 4 }
        var value: UInt32 = 0
        for index in stride(from: 3, through: 0, by: -1) {
            value = (value << 8) | UInt32(bytes[offset + index])
        }
        return value
    }

    mutating func readUInt64() -> UInt64? {
        guard remaining >= 8 else { return nil }
        defer { offset += 8 }
        var value: UInt64 = 0
        for index in stride(from: 7, through: 0, by: -1) {
            value = (value << 8) | UInt64(bytes[offset + index])
        }
        return value
    }
}

// MARK: - Packet builders

extension WireFormat {

    /// `struct '<IHHIHHHHBBQQI'` — exactly what UdpYuvSender.sendFrame writes.
    static func frameHeader(
        sequence: UInt32,
        chunkIndex: UInt16,
        chunkCount: UInt16,
        width: UInt16,
        height: UInt16,
        format: UInt8,
        flags: UInt8 = 0,
        sensorTimeNs: UInt64,
        frameSendTimeNs: UInt64,
        payloadSize: UInt32
    ) -> [UInt8] {
        var writer = ByteWriter(capacity: frameHeaderSize)
        writer.putUInt32(frameMagic)
        writer.putUInt16(frameVersion)
        writer.putUInt16(UInt16(frameHeaderSize))
        writer.putUInt32(sequence)
        writer.putUInt16(chunkIndex)
        writer.putUInt16(chunkCount)
        writer.putUInt16(width)
        writer.putUInt16(height)
        writer.putUInt8(format)
        writer.putUInt8(flags)
        writer.putUInt64(sensorTimeNs)
        writer.putUInt64(frameSendTimeNs)
        writer.putUInt32(payloadSize)
        return writer.bytes
    }

    /// `struct '<IHHI'` — exactly what UdpYuvSender.sendIntrinsics writes.
    static func intrinsicsEnvelope(json: [UInt8]) -> [UInt8] {
        var writer = ByteWriter(capacity: intrinsicsHeaderSize + json.count)
        writer.putUInt32(intrinsicsMagic)
        writer.putUInt16(intrinsicsVersion)
        writer.putUInt16(UInt16(intrinsicsHeaderSize))
        writer.putUInt32(UInt32(json.count))
        writer.putBytes(json)
        return writer.bytes
    }

    /// `struct '<IHHQQQ'` — the phone half of the four-timestamp probe.
    ///
    /// `t1` is echoed from the PC request, `t2`/`t3` are the phone's monotonic
    /// receive/send times in nanoseconds.
    static func clockReply(t1: UInt64, t2: UInt64, t3: UInt64) -> [UInt8] {
        var writer = ByteWriter(capacity: clockPacketSize)
        writer.putUInt32(clockMagic)
        writer.putUInt16(clockVersion)
        writer.putUInt16(clockReplyKind)
        writer.putUInt64(t1)
        writer.putUInt64(t2)
        writer.putUInt64(t3)
        return writer.bytes
    }

    /// Parsed PC clock request. Returns nil for anything the PC would not accept
    /// as a request, so a stray datagram can never trigger a reply.
    struct ClockRequest {
        let t1: UInt64
    }

    static func parseClockRequest(_ packet: [UInt8]) -> ClockRequest? {
        guard packet.count == clockPacketSize else { return nil }
        var reader = ByteReader(packet)
        guard let magic = reader.readUInt32(), magic == clockMagic else { return nil }
        guard let version = reader.readUInt16(), version == clockVersion else { return nil }
        guard let kind = reader.readUInt16(), kind == clockRequestKind else { return nil }
        guard let t1 = reader.readUInt64(), t1 > 0 else { return nil }
        guard let t2 = reader.readUInt64(), t2 == 0 else { return nil }
        guard let t3 = reader.readUInt64(), t3 == 0 else { return nil }
        return ClockRequest(t1: t1)
    }

    /// `struct '<4sIIHHQQQI'` — exactly what AvcTcpSender writes.
    static func avcHeader(
        sequence: UInt32,
        flags: UInt32,
        width: UInt16,
        height: UInt16,
        sensorTimeNs: UInt64,
        encodedTimeNs: UInt64,
        phoneSendTimeNs: UInt64,
        payloadSize: UInt32
    ) -> [UInt8] {
        var writer = ByteWriter(capacity: avcHeaderSize)
        writer.putBytes(avcMagic)
        writer.putUInt32(sequence)
        writer.putUInt32(flags)
        writer.putUInt16(width)
        writer.putUInt16(height)
        writer.putUInt64(sensorTimeNs)
        writer.putUInt64(encodedTimeNs)
        writer.putUInt64(phoneSendTimeNs)
        writer.putUInt32(payloadSize)
        return writer.bytes
    }

    // MARK: Discovery payloads

    static func discoveryQuery(
        phoneID: String,
        phoneName: String,
        instanceID: String,
        nonce: String
    ) -> [String: Any] {
        [
            "magic": discoveryMagic,
            "version": discoveryVersion,
            "type": discoveryPayloadType,
            "nonce": nonce,
            "phone_id": phoneID,
            "phone_name": phoneName,
            "instance_id": instanceID,
        ]
    }

    struct DiscoveryOffer {
        let instanceID: String
        let pcName: String
        let dataPort: UInt16
        let accepted: Bool
    }

    /// Mirrors MainActivity.kt's offer validation: magic, version, type, nonce
    /// echo, an optional instance match, and a port inside 1...65535.
    static func parseDiscoveryOffer(_ data: Data, nonce: String, expectedInstanceID: String) -> DiscoveryOffer? {
        guard let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else { return nil }
        guard object["magic"] as? String == discoveryMagic else { return nil }
        guard (object["version"] as? NSNumber)?.intValue == discoveryVersion else { return nil }
        guard object["type"] as? String == discoveryOfferType else { return nil }
        guard object["nonce"] as? String == nonce else { return nil }
        guard let instanceID = object["instance_id"] as? String, !instanceID.isEmpty else { return nil }
        if !expectedInstanceID.isEmpty && expectedInstanceID != instanceID { return nil }
        guard let portNumber = (object["data_port"] as? NSNumber)?.intValue, (1...65535).contains(portNumber) else {
            return nil
        }
        return DiscoveryOffer(
            instanceID: instanceID,
            pcName: object["pc_name"] as? String ?? "OpenGazeLink",
            dataPort: UInt16(portNumber),
            accepted: (object["accepted"] as? NSNumber)?.boolValue ?? false
        )
    }
}
