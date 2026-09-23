import Foundation

/// H.264 byte-stream helpers.
///
/// VideoToolbox hands back **AVCC** access units: each NAL unit is prefixed by
/// a big-endian length field whose width comes from the format description,
/// and SPS/PPS live in the format description rather than in the sample data.
///
/// The PC provider decodes with `av.CodecContext.create('h264', 'r')` and no
/// extradata (h264_stream.py:31), so it parses the elementary stream as
/// **Annex-B** and requires SPS/PPS inline. Everything sent over TCP must
/// therefore be converted here first.
enum AnnexB {

    /// Four-byte start code, matching what Android MediaCodec emits.
    static let startCode: [UInt8] = [0x00, 0x00, 0x00, 0x01]

    /// NAL unit types that matter to the PC's decoder.
    enum NALType: UInt8 {
        case nonIDR = 1
        case idr = 5
        case sei = 6
        case sps = 7
        case pps = 8
        case aud = 9

        /// H.264 NAL header: forbidden_zero_bit | nal_ref_idc (2 bits) | type (5 bits).
        init?(headerByte: UInt8) {
            self.init(rawValue: headerByte & 0x1F)
        }
    }

    /// Converts one AVCC access unit to Annex-B.
    ///
    /// Returns nil when a declared length overruns the buffer, which is the only
    /// way VideoToolbox output can be malformed; emitting a truncated access unit
    /// would desynchronise the PC decoder instead of dropping a single frame.
    static func convert(avcc: [UInt8], nalUnitHeaderLength: Int) -> [UInt8]? {
        guard nalUnitHeaderLength == 1 || nalUnitHeaderLength == 2 || nalUnitHeaderLength == 4 else { return nil }
        var output: [UInt8] = []
        output.reserveCapacity(avcc.count + 16)
        var offset = 0
        while offset + nalUnitHeaderLength <= avcc.count {
            var length = 0
            for index in 0..<nalUnitHeaderLength {
                length = (length << 8) | Int(avcc[offset + index])
            }
            offset += nalUnitHeaderLength
            guard length > 0, offset + length <= avcc.count else { return nil }
            output.append(contentsOf: startCode)
            output.append(contentsOf: avcc[offset..<(offset + length)])
            offset += length
        }
        // A trailing partial length field means the access unit was truncated.
        guard offset == avcc.count else { return nil }
        return output
    }

    /// Builds the codec-configuration access unit: SPS then PPS, Annex-B framed.
    ///
    /// The PC stores this verbatim when `flags & 2` is set and prepends it to
    /// every packet that carries `flags & 1` (h264_stream.py:275), so it must be
    /// a valid Annex-B access unit and not a bare parameter-set blob.
    static func codecConfig(sps: [UInt8], pps: [UInt8]) -> [UInt8] {
        var output: [UInt8] = []
        output.reserveCapacity(sps.count + pps.count + 8)
        output.append(contentsOf: startCode)
        output.append(contentsOf: sps)
        output.append(contentsOf: startCode)
        output.append(contentsOf: pps)
        return output
    }

    /// NAL unit types present in an Annex-B buffer, in stream order.
    static func nalTypes(in annexB: [UInt8]) -> [UInt8] {
        var types: [UInt8] = []
        var index = 0
        while index + 4 <= annexB.count {
            guard annexB[index] == 0, annexB[index + 1] == 0 else {
                index += 1
                continue
            }
            let headerIndex: Int
            if annexB[index + 2] == 1 {
                headerIndex = index + 3
            } else if annexB[index + 2] == 0, index + 5 <= annexB.count, annexB[index + 3] == 1 {
                headerIndex = index + 4
            } else {
                index += 1
                continue
            }
            guard headerIndex < annexB.count else { break }
            if let type = NALType(headerByte: annexB[headerIndex]) {
                types.append(type.rawValue)
            }
            index = headerIndex + 1
        }
        return types
    }

    /// True when the access unit contains an IDR slice.
    ///
    /// The PC treats `flags & 1` as "keyframe": it prepends the stored SPS/PPS
    /// and uses it to resynchronise after a decode queue overflow. Detecting the
    /// IDR in the bitstream is exact, and unlike reading
    /// `kCMSampleAttachmentKey_NotSync` it needs no CoreMedia attachment
    /// bridging and is unit-testable without an encoder.
    static func containsIDR(annexB: [UInt8]) -> Bool {
        nalTypes(in: annexB).contains(NALType.idr.rawValue)
    }
}
