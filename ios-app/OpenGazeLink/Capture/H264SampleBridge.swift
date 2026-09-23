import CoreMedia
import Foundation

/// CoreMedia side of the H.264 conversion: pull the compressed bytes and the
/// parameter sets out of VideoToolbox's output.
///
/// Kept separate from `AnnexB` so the byte-level transformation stays testable
/// without a CMSampleBuffer.
enum H264SampleBridge {

    struct ParameterSets {
        let sps: [UInt8]
        let pps: [UInt8]
        let nalUnitHeaderLength: Int

        var annexBCodecConfig: [UInt8] {
            AnnexB.codecConfig(sps: sps, pps: pps)
        }
    }

    /// SPS/PPS out of the encoder's `CMVideoFormatDescription`.
    ///
    /// `parameterSetCountOut` is also the only place `nalUnitHeaderLength` is
    /// reported, and it is needed to parse the access unit itself.
    static func parameterSets(formatDescription: CMFormatDescription) -> ParameterSets? {
        var count = 0
        var headerLength: Int32 = 0
        let countStatus = CMVideoFormatDescriptionGetH264ParameterSetAtIndex(
            formatDescription,
            parameterSetIndex: 0,
            parameterSetPointerOut: nil,
            parameterSetSizeOut: nil,
            parameterSetCountOut: &count,
            nalUnitHeaderLengthOut: &headerLength
        )
        guard countStatus == noErr, count >= 2 else { return nil }

        func parameterSet(at index: Int) -> [UInt8]? {
            var pointer: UnsafePointer<UInt8>?
            var size = 0
            let status = CMVideoFormatDescriptionGetH264ParameterSetAtIndex(
                formatDescription,
                parameterSetIndex: index,
                parameterSetPointerOut: &pointer,
                parameterSetSizeOut: &size,
                parameterSetCountOut: nil,
                nalUnitHeaderLengthOut: nil
            )
            guard status == noErr, let pointer, size > 0 else { return nil }
            return Array(UnsafeBufferPointer(start: pointer, count: size))
        }

        guard let sps = parameterSet(at: 0), let pps = parameterSet(at: 1) else { return nil }
        return ParameterSets(sps: sps, pps: pps, nalUnitHeaderLength: Int(headerLength))
    }

    /// Raw AVCC bytes of one access unit.
    static func accessUnitData(from sampleBuffer: CMSampleBuffer) -> [UInt8]? {
        guard let blockBuffer = CMSampleBufferGetDataBuffer(sampleBuffer) else { return nil }
        let length = CMBlockBufferGetDataLength(blockBuffer)
        guard length > 0 else { return nil }
        var bytes = [UInt8](repeating: 0, count: length)
        let status: OSStatus = bytes.withUnsafeMutableBytes { raw in
            guard let base = raw.baseAddress else { return OSStatus(-1) }
            return CMBlockBufferCopyDataBytes(
                blockBuffer, atOffset: 0, dataLength: length, destination: base
            )
        }
        guard status == noErr else { return nil }
        return bytes
    }

    /// Annex-B access unit ready for the wire.
    static func annexBAccessUnit(from sampleBuffer: CMSampleBuffer, formatDescription: CMFormatDescription) -> [UInt8]? {
        guard let parameterSets = parameterSets(formatDescription: formatDescription),
              let avcc = accessUnitData(from: sampleBuffer)
        else { return nil }
        return AnnexB.convert(avcc: avcc, nalUnitHeaderLength: parameterSets.nalUnitHeaderLength)
    }
}
