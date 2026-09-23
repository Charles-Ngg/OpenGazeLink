import Darwin
import Foundation

/// Bounded TCP sender for Annex-B H.264 access units.
///
/// Port of `AvcTcpSender.kt`:
/// - `TCP_NODELAY`, `SO_SNDBUF = 32 KiB` (deliberately small: a large send
///   buffer lets `write` return immediately while the PC is still draining,
///   which shows up as hundreds of milliseconds of display latency)
/// - a 16-packet queue that drops the OLDEST entry when full, so a transient PC
///   stall degrades to skipped frames instead of unbounded memory
/// - sequence numbers assigned at write time, so a dropped frame leaves no gap
///   for the PC's `seq != last_seq + 1` warning
///
/// The PC accepts a single connection at a time (h264_stream.py:121), so
/// `close()` must fully tear the socket down before a new sender is created.
final class AvcTcpSender {

    struct Packet {
        let data: [UInt8]
        let sensorTimeNs: UInt64
        let encodedTimeNs: UInt64
        let flags: UInt32
    }

    enum Failure: LocalizedError {
        case addressUnresolved(String)
        case socket(Int32)
        case connect(Int32)
        case write(Int32)
        case oversizedPacket(Int)

        var errorDescription: String? {
            switch self {
            case .addressUnresolved(let host):
                return "Could not resolve PC address \(host)"
            case .socket(let code):
                return "TCP socket failed: \(String(cString: strerror(code))) (errno \(code))"
            case .connect(let code):
                return "TCP connect failed: \(String(cString: strerror(code))) (errno \(code))"
            case .write(let code):
                return "TCP write failed: \(String(cString: strerror(code))) (errno \(code))"
            case .oversizedPacket(let size):
                return "H.264 access unit is \(size) bytes; the PC rejects packets over \(WireFormat.maxAvcPacketBytes)"
            }
        }
    }

    static let queueDepth = 16

    /// Called for every packet that carries frame data. Codec-configuration
    /// packets are excluded, matching Android's `if (packet.flags and 2 == 0)`.
    var onSent: ((Int) -> Void)?
    var onError: ((Error) -> Void)?

    private let fileDescriptor: Int32
    private let width: Int
    private let height: Int
    private let condition = NSCondition()
    private var queue: [Packet] = []
    private var closed = false
    private var sequence: UInt32 = 0
    private var droppedPackets = 0
    private var sendThread: Thread?

    init(
        host: String,
        port: UInt16,
        width: Int,
        height: Int,
        connectTimeoutSeconds: TimeInterval = 3.0
    ) throws {
        guard let resolved = SocketSupport.resolveIPv4(host: host),
              let address = SocketSupport.ipv4Address(host: resolved, port: port)
        else {
            throw Failure.addressUnresolved(host)
        }
        self.width = width
        self.height = height

        let fd = socket(AF_INET, SOCK_STREAM, 0)
        guard fd >= 0 else { throw Failure.socket(errno) }
        fileDescriptor = fd

        SocketSupport.setIntOption(fd, level: SOL_SOCKET, name: SO_NOSIGPIPE, value: 1)
        SocketSupport.setIntOption(fd, level: IPPROTO_TCP, name: TCP_NODELAY, value: 1)
        SocketSupport.setIntOption(fd, level: SOL_SOCKET, name: SO_SNDBUF, value: 32 * 1024)

        guard AvcTcpSender.connect(fd, to: address, timeout: connectTimeoutSeconds) == 0 else {
            let code = errno
            Darwin.close(fd)
            throw Failure.connect(code)
        }

        let thread = Thread { [weak self] in self?.runSendLoop() }
        thread.name = "avc-tcp-send"
        thread.stackSize = 512 * 1024
        sendThread = thread
        thread.start()
    }

    deinit {
        close()
    }

    /// Enqueues a packet. Never blocks the capture callback: when the queue is
    /// full the oldest entry is discarded.
    func offer(data: [UInt8], sensorTimeNs: UInt64, encodedTimeNs: UInt64, flags: UInt32) {
        guard data.count <= WireFormat.maxAvcPacketBytes else {
            onError?(Failure.oversizedPacket(data.count))
            return
        }
        let packet = Packet(
            data: data, sensorTimeNs: sensorTimeNs, encodedTimeNs: encodedTimeNs, flags: flags
        )
        condition.lock()
        if closed {
            condition.unlock()
            return
        }
        if queue.count >= AvcTcpSender.queueDepth {
            queue.removeFirst()
            droppedPackets += 1
        }
        queue.append(packet)
        condition.broadcast()
        condition.unlock()
    }

    func droppedPacketCount() -> Int {
        condition.lock()
        defer { condition.unlock() }
        return droppedPackets
    }

    func close() {
        condition.lock()
        if closed {
            condition.unlock()
            return
        }
        closed = true
        queue.removeAll()
        condition.broadcast()
        condition.unlock()
        // shutdown() releases the blocked send() before the descriptor is
        // recycled, which is what lets close() be called from the capture
        // callback without racing the send thread. `Darwin.close` is spelled out
        // because this type has its own close().
        shutdown(fileDescriptor, SHUT_RDWR)
        Darwin.close(fileDescriptor)
        sendThread = nil
    }

    // MARK: - Send loop

    private func runSendLoop() {
        while true {
            condition.lock()
            while queue.isEmpty && !closed {
                condition.wait(until: Date().addingTimeInterval(0.1))
            }
            if closed {
                condition.unlock()
                return
            }
            let packet = queue.removeFirst()
            condition.unlock()

            let header = WireFormat.avcHeader(
                sequence: sequence,
                flags: packet.flags,
                width: UInt16(clamping: width),
                height: UInt16(clamping: height),
                sensorTimeNs: packet.sensorTimeNs,
                encodedTimeNs: packet.encodedTimeNs,
                phoneSendTimeNs: MonotonicClock.nowNs(),
                payloadSize: UInt32(packet.data.count)
            )
            // One syscall per packet. With TCP_NODELAY, writing the header and
            // the payload separately can emit two segments per frame.
            var datagram = header
            datagram.append(contentsOf: packet.data)

            guard writeAll(datagram) else {
                let code = errno
                condition.lock()
                let alreadyClosed = closed
                closed = true
                queue.removeAll()
                condition.unlock()
                if !alreadyClosed {
                    shutdown(fileDescriptor, SHUT_RDWR)
                    Darwin.close(fileDescriptor)
                    onError?(Failure.write(code))
                }
                return
            }

            sequence &+= 1
            if packet.flags & WireFormat.avcFlagCodecConfig == 0 {
                onSent?(datagram.count)
            }
        }
    }

    private func writeAll(_ bytes: [UInt8]) -> Bool {
        var offset = 0
        return bytes.withUnsafeBufferPointer { buffer -> Bool in
            guard let base = buffer.baseAddress else { return true }
            while offset < buffer.count {
                let written = Darwin.send(fileDescriptor, base + offset, buffer.count - offset, 0)
                if written < 0 {
                    if errno == EINTR { continue }
                    return false
                }
                if written == 0 { return false }
                offset += written
            }
            return true
        }
    }

    // MARK: - Connect with timeout

    /// Non-blocking connect driven by `poll`, so a powered-off PC surfaces as a
    /// clear error in three seconds instead of a hung capture queue.
    private static func connect(_ fd: Int32, to address: sockaddr_in, timeout: TimeInterval) -> Int32 {
        let originalFlags = fcntl(fd, F_GETFL, 0)
        _ = fcntl(fd, F_SETFL, originalFlags | O_NONBLOCK)

        var mutable = address
        let result = withUnsafePointer(to: &mutable) { pointer in
            pointer.withMemoryRebound(to: sockaddr.self, capacity: 1) { socketAddress in
                Darwin.connect(fd, socketAddress, socklen_t(MemoryLayout<sockaddr_in>.size))
            }
        }

        func restoreBlocking() {
            _ = fcntl(fd, F_SETFL, originalFlags)
        }

        if result == 0 {
            restoreBlocking()
            return 0
        }
        guard errno == EINPROGRESS else { return -1 }

        var descriptor = pollfd(fd: fd, events: Int16(POLLOUT), revents: 0)
        let selected = poll(&descriptor, 1, Int32(max(1.0, timeout) * 1000.0))
        guard selected > 0 else {
            errno = selected == 0 ? ETIMEDOUT : errno
            return -1
        }
        var socketError: Int32 = 0
        var length = socklen_t(MemoryLayout<Int32>.size)
        guard getsockopt(fd, SOL_SOCKET, SO_ERROR, &socketError, &length) == 0, socketError == 0 else {
            errno = socketError
            return -1
        }
        restoreBlocking()
        return 0
    }
}
