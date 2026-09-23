import Darwin
import Foundation

/// UDP JPEG sender plus the clock-probe responder.
///
/// Byte-for-byte port of `UdpYuvSender.kt`:
/// - one connected UDP socket, `SO_SNDBUF = 128 KiB`
/// - 1400-byte chunk payloads, 42-byte little-endian header per chunk
/// - a dedicated thread that answers the PC's 32-byte `EYCT` probe
///
/// The socket is connected (not merely addressed) so that the PC's clock probe
/// is the only datagram the receive path can ever see, and so the source port
/// stays stable for the whole session — the PC replies to whichever address it
/// last received a frame from (camera.py:305).
final class UdpSender {

    struct FrameSendStats {
        let payloadBytes: Int
        let chunks: Int
        /// Header bytes included, matching Android's `bytes + chunks * 42`.
        var wireBytes: Int { payloadBytes + chunks * WireFormat.frameHeaderSize }
    }

    enum Failure: LocalizedError {
        case addressUnresolved(String)
        case socket(Int32)
        case clockReplySend(Int32)

        var errorDescription: String? {
            switch self {
            case .addressUnresolved(let host):
                return "Could not resolve PC address \(host)"
            case .socket(let code):
                return "UDP socket failed: \(String(cString: strerror(code))) (errno \(code))"
            case .clockReplySend(let code):
                return "Clock reply failed: \(String(cString: strerror(code))) (errno \(code))"
            }
        }
    }

    private let fileDescriptor: Int32
    private let chunkPayloadBytes: Int
    private let lock = NSLock()
    private var frameSequence: UInt32 = 0
    private var closed = false
    private var replyThread: Thread?
    private var clockReplyFailures = 0

    init(host: String, port: UInt16, chunkPayloadBytes: Int = WireFormat.frameChunkPayloadBytes) throws {
        guard let resolved = SocketSupport.resolveIPv4(host: host),
              let address = SocketSupport.ipv4Address(host: resolved, port: port)
        else {
            throw Failure.addressUnresolved(host)
        }
        self.chunkPayloadBytes = max(1, chunkPayloadBytes)

        let fd = socket(AF_INET, SOCK_DGRAM, 0)
        guard fd >= 0 else { throw Failure.socket(errno) }
        fileDescriptor = fd

        // Keep back-pressure close to the camera. A multi-megabyte UDP send
        // queue makes send() look fast while displaying frames hundreds of
        // milliseconds late.
        SocketSupport.setIntOption(fd, level: SOL_SOCKET, name: SO_SNDBUF, value: 128 * 1024)
        SocketSupport.setIntOption(fd, level: SOL_SOCKET, name: SO_NOSIGPIPE, value: 1)
        SocketSupport.setReceiveTimeout(fd, seconds: 0.2)

        guard SocketSupport.connect(fd, to: address) == 0 else {
            let code = errno
            close(fd)
            throw Failure.socket(code)
        }

        let thread = Thread { [weak self] in self?.runClockReplies() }
        thread.name = "udp-clock-reply"
        thread.stackSize = 256 * 1024
        replyThread = thread
        thread.start()
    }

    deinit {
        close()
    }

    /// Chunks and sends one complete frame.
    func sendFrame(
        payload: [UInt8],
        width: Int,
        height: Int,
        sensorTimeNs: UInt64,
        format: UInt8
    ) -> FrameSendStats? {
        guard WireFormat.supportedUDPFormats.contains(format) else { return nil }
        lock.lock()
        if closed {
            lock.unlock()
            return nil
        }
        let sequence = frameSequence
        frameSequence &+= 1
        lock.unlock()

        let chunkCount = max(1, (payload.count + chunkPayloadBytes - 1) / chunkPayloadBytes)
        guard chunkCount <= Int(UInt16.max) else { return nil }
        let frameSendTimeNs = MonotonicClock.nowNs()

        var offset = 0
        for chunkIndex in 0..<chunkCount {
            let payloadSize = min(chunkPayloadBytes, payload.count - offset)
            let header = WireFormat.frameHeader(
                sequence: sequence,
                chunkIndex: UInt16(chunkIndex),
                chunkCount: UInt16(chunkCount),
                width: UInt16(clamping: width),
                height: UInt16(clamping: height),
                format: format,
                sensorTimeNs: sensorTimeNs,
                frameSendTimeNs: frameSendTimeNs,
                payloadSize: UInt32(payloadSize)
            )
            var datagram = header
            datagram.append(contentsOf: payload[offset..<(offset + payloadSize)])
            let sent = datagram.withUnsafeBufferPointer { buffer in
                Darwin.send(fileDescriptor, buffer.baseAddress, buffer.count, 0)
            }
            if sent < 0 {
                return nil
            }
            offset += payloadSize
        }
        return FrameSendStats(payloadBytes: payload.count, chunks: chunkCount)
    }

    /// Sends the `EYCI` intrinsics envelope. Not part of the frame sequence.
    func sendIntrinsics(_ json: [UInt8]) -> Bool {
        let envelope = WireFormat.intrinsicsEnvelope(json: json)
        lock.lock()
        let isClosed = closed
        lock.unlock()
        guard !isClosed else { return false }
        return envelope.withUnsafeBufferPointer { buffer in
            Darwin.send(fileDescriptor, buffer.baseAddress, buffer.count, 0) >= 0
        }
    }

    func clockReplyFailureCount() -> Int {
        lock.lock()
        defer { lock.unlock() }
        return clockReplyFailures
    }

    func close() {
        lock.lock()
        if closed {
            lock.unlock()
            return
        }
        closed = true
        lock.unlock()
        // shutdown() releases the blocked recv() before the descriptor goes
        // away; closing first would leave the reply thread reading a recycled fd.
        shutdown(fileDescriptor, SHUT_RDWR)
        close(fileDescriptor)
        replyThread = nil
    }

    // MARK: - Clock probe responder

    /// Replies independently of JPEG encoding and frame sending, so the PC's
    /// transport-backlog estimate is not polluted by encode jitter.
    private func runClockReplies() {
        var buffer = [UInt8](repeating: 0, count: WireFormat.clockPacketSize)
        while true {
            lock.lock()
            let isClosed = closed
            lock.unlock()
            if isClosed { return }

            let received = buffer.withUnsafeMutableBufferPointer { pointer in
                Darwin.recv(fileDescriptor, pointer.baseAddress, pointer.count, 0)
            }
            if received < 0 {
                let code = errno
                if code == EAGAIN || code == EWOULDBLOCK || code == EINTR { continue }
                return
            }
            let receivedNs = MonotonicClock.nowNs()
            guard received == WireFormat.clockPacketSize else { continue }
            guard let request = WireFormat.parseClockRequest(buffer) else { continue }

            let reply = WireFormat.clockReply(
                t1: request.t1,
                t2: receivedNs,
                t3: MonotonicClock.nowNs()
            )
            let sent = reply.withUnsafeBufferPointer { pointer in
                Darwin.send(fileDescriptor, pointer.baseAddress, pointer.count, 0)
            }
            if sent < 0 {
                lock.lock()
                clockReplyFailures += 1
                lock.unlock()
            }
        }
    }
}
