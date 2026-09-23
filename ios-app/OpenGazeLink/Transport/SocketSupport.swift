import Darwin
import Foundation

/// Thin BSD-socket helpers.
///
/// BSD sockets are used rather than Network.framework for three reasons:
/// `NWConnection` has no broadcast support, the Android senders set explicit
/// socket buffer sizes (`sendBufferSize = 32 * 1024` on TCP,
/// `128 * 1024` on UDP) that map directly onto `SO_SNDBUF`, and the same
/// connected UDP socket must both send frames and receive the PC's clock probe.
enum SocketSupport {

    struct SocketError: LocalizedError {
        let operation: String
        let code: Int32
        var errorDescription: String? {
            let text = String(cString: strerror(code))
            return "\(operation) failed: \(text) (errno \(code))"
        }
    }

    static func lastError(_ operation: String) -> SocketError {
        SocketError(operation: operation, code: errno)
    }

    /// Builds an IPv4 address. Accepts a literal address; host names are
    /// resolved by the caller so a DNS stall cannot block the capture queue.
    static func ipv4Address(host: String, port: UInt16) -> sockaddr_in? {
        var address = sockaddr_in()
        address.sin_len = UInt8(MemoryLayout<sockaddr_in>.size)
        address.sin_family = sa_family_t(AF_INET)
        address.sin_port = port.bigEndian
        guard inet_pton(AF_INET, host, &address.sin_addr) == 1 else { return nil }
        return address
    }

    /// `inet_pton` only accepts literals, so resolve names first and keep the
    /// first IPv4 result.
    static func resolveIPv4(host: String) -> String? {
        if ipv4Address(host: host, port: 0) != nil { return host }
        var hints = addrinfo(
            ai_flags: 0, ai_family: AF_INET, ai_socktype: SOCK_DGRAM,
            ai_protocol: 0, ai_addrlen: 0, ai_canonname: nil, ai_addr: nil, ai_next: nil
        )
        var result: UnsafeMutablePointer<addrinfo>?
        guard getaddrinfo(host, nil, &hints, &result) == 0, let first = result else { return nil }
        defer { freeaddrinfo(result) }
        var pointer: UnsafeMutablePointer<addrinfo>? = first
        while let current = pointer {
            if let address = current.pointee.ai_addr, address.pointee.sa_family == sa_family_t(AF_INET) {
                var storage = sockaddr_in()
                memcpy(&storage, address, MemoryLayout<sockaddr_in>.size)
                var buffer = [CChar](repeating: 0, count: Int(INET_ADDRSTRLEN))
                guard inet_ntop(AF_INET, &storage.sin_addr, &buffer, socklen_t(INET_ADDRSTRLEN)) != nil else {
                    return nil
                }
                return String(cString: buffer)
            }
            pointer = current.pointee.ai_next
        }
        return nil
    }

    static func connect(_ fd: Int32, to address: sockaddr_in) -> Int32 {
        var mutable = address
        return withUnsafePointer(to: &mutable) { pointer in
            pointer.withMemoryRebound(to: sockaddr.self, capacity: 1) { socketAddress in
                Darwin.connect(fd, socketAddress, socklen_t(MemoryLayout<sockaddr_in>.size))
            }
        }
    }

    static func bind(_ fd: Int32, to address: sockaddr_in) -> Int32 {
        var mutable = address
        return withUnsafePointer(to: &mutable) { pointer in
            pointer.withMemoryRebound(to: sockaddr.self, capacity: 1) { socketAddress in
                Darwin.bind(fd, socketAddress, socklen_t(MemoryLayout<sockaddr_in>.size))
            }
        }
    }

    static func sendTo(_ fd: Int32, bytes: [UInt8], address: sockaddr_in) -> Int {
        var mutable = address
        return bytes.withUnsafeBufferPointer { buffer in
            withUnsafePointer(to: &mutable) { pointer in
                pointer.withMemoryRebound(to: sockaddr.self, capacity: 1) { socketAddress in
                    let sent = Darwin.sendto(
                        fd, buffer.baseAddress, buffer.count, 0,
                        socketAddress, socklen_t(MemoryLayout<sockaddr_in>.size)
                    )
                    return Int(sent)
                }
            }
        }
    }

    @discardableResult
    static func setIntOption(_ fd: Int32, level: Int32, name: Int32, value: Int32) -> Bool {
        var mutable = value
        return setsockopt(fd, level, name, &mutable, socklen_t(MemoryLayout<Int32>.size)) == 0
    }

    /// Bounds a blocking `recv`/`accept` so a closed socket cannot leave a
    /// thread parked forever.
    @discardableResult
    static func setReceiveTimeout(_ fd: Int32, seconds: Double) -> Bool {
        var timeout = timeval(
            tv_sec: Int(seconds),
            tv_usec: suseconds_t((seconds - Double(Int(seconds))) * 1_000_000)
        )
        return setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, socklen_t(MemoryLayout<timeval>.size)) == 0
    }

    /// Loopback and link-local IPv4 addresses, which are never a PC provider.
    static func isUsableLANAddress(_ address: String) -> Bool {
        var parsed = in_addr()
        guard inet_pton(AF_INET, address, &parsed) == 1 else { return false }
        let value = UInt32(bigEndian: parsed.s_addr)
        let first = UInt8(truncatingIfNeeded: value >> 24)
        if first == 127 || first == 0 || first >= 224 { return false }
        if first == 169 && UInt8(truncatingIfNeeded: value >> 16) == 254 { return false }
        return true
    }
}
