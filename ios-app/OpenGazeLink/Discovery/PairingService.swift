import Darwin
import Foundation

/// PC discovery and pairing client.
///
/// Protocol (pairing.py:73):
/// - Phone → `255.255.255.255:5006` and the subnet broadcast address, JSON
///   `{"magic":"EYETRACING_DISCOVERY_V1","version":1,"type":"discover",
///     "nonce":…,"phone_id":…,"phone_name":…,"instance_id":…}`.
/// - PC → phone, `{"type":"offer", …, "nonce":<echo>, "instance_id":…,
///   "pc_name":…, "data_port":…, "accepted":…}`.
///
/// iOS 14+ refuses broadcast traffic without the Multicast Networking
/// entitlement, and Network.framework cannot broadcast at all. This service
/// therefore sends the identical payload three ways, in increasing cost:
///
/// 1. Unicast to the last paired PC address — the common case after the first
///    pairing, and the only step needed for a stable setup.
/// 2. Broadcast to `255.255.255.255` and to each interface's directed
///    broadcast. Free with the entitlement, refused with `EPERM` without it.
/// 3. A paced unicast sweep of the phone's own /24. This needs no entitlement
///    and is the reason discovery still works on a free-account sideload.
///
/// Manual IP entry remains available in every case and is the primary path on
/// a free Apple account.
final class PairingService {

    struct Offer {
        let instanceID: String
        let pcName: String
        let dataPort: UInt16
        let accepted: Bool
        /// Address the offer actually arrived from, which is the address frames
        /// must be sent to (never the broadcast address).
        let address: String
    }

    enum Event {
        case searching
        case sweeping(Int)
        case offer(Offer)
        case failed(String)
    }

    private let store: PairingStore
    private let lock = NSLock()
    private var thread: Thread?
    private var cancelled = false

    init(store: PairingStore) {
        self.store = store
    }

    var isRunning: Bool {
        lock.lock()
        defer { lock.unlock() }
        return thread != nil && !cancelled
    }

    func start(onEvent: @escaping (Event) -> Void) {
        stop()
        lock.lock()
        cancelled = false
        let newThread = Thread { [weak self] in self?.run(onEvent: onEvent) }
        newThread.name = "phone-pc-discovery"
        newThread.stackSize = 512 * 1024
        thread = newThread
        lock.unlock()
        newThread.start()
    }

    func stop() {
        lock.lock()
        cancelled = true
        let existing = thread
        thread = nil
        lock.unlock()
        // The run loop polls the flag through its socket timeouts, so there is
        // nothing to join: it exits within one receive window.
        _ = existing
    }

    // MARK: - Run loop

    private func run(onEvent: @escaping (Event) -> Void) {
        let fd = socket(AF_INET, SOCK_DGRAM, 0)
        guard fd >= 0 else {
            onEvent(.failed("UDP socket failed: \(String(cString: strerror(errno)))"))
            return
        }
        defer { close(fd) }

        SocketSupport.setIntOption(fd, level: SOL_SOCKET, name: SO_BROADCAST, value: 1)
        SocketSupport.setIntOption(fd, level: SOL_SOCKET, name: SO_NOSIGPIPE, value: 1)
        SocketSupport.setIntOption(fd, level: SOL_SOCKET, name: SO_REUSEADDR, value: 1)
        SocketSupport.setReceiveTimeout(fd, seconds: 0.2)
        // Bind to an ephemeral port so the PC's reply reaches this socket.
        if let local = SocketSupport.ipv4Address(host: "0.0.0.0", port: 0),
           SocketSupport.bind(fd, to: local) != 0 {
            onEvent(.failed("Could not bind discovery socket: \(String(cString: strerror(errno)))"))
            return
        }

        let phoneID = store.phoneID
        let phoneName = DeviceIdentity.phoneName
        // One nonce per discovery run, exactly as the Android client does.
        let nonce = UUID().uuidString

        while !isCancelled {
            onEvent(.searching)

            let instanceID = store.pairedInstanceID
            let payload = discoveryPayload(
                phoneID: phoneID, phoneName: phoneName, instanceID: instanceID, nonce: nonce
            )

            sendSavedHost(fd: fd, payload: payload)
            let broadcastError = sendBroadcasts(fd: fd, payload: payload)

            if let offer = receiveOffer(fd: fd, nonce: nonce, expectedInstanceID: instanceID, window: 1.2) {
                onEvent(.offer(offer))
                sleepUnlessCancelled(3.0)
                continue
            }

            let candidates = LocalNetwork.sweepCandidates(over: LocalNetwork.interfaces())
            if !candidates.isEmpty {
                onEvent(.sweeping(candidates.count))
                sendSweep(fd: fd, payload: payload, candidates: candidates)
                if let offer = receiveOffer(fd: fd, nonce: nonce, expectedInstanceID: instanceID, window: 3.0) {
                    onEvent(.offer(offer))
                    sleepUnlessCancelled(3.0)
                    continue
                }
            }

            if let broadcastError {
                // Only report the broadcast failure when the sweep also found
                // nothing; otherwise it is noise about a path that was not needed.
                onEvent(.failed(broadcastError))
            } else {
                onEvent(.failed("No OpenGazeLink PC answered on this network"))
            }
            sleepUnlessCancelled(3.0)
        }
    }

    private var isCancelled: Bool {
        lock.lock()
        defer { lock.unlock() }
        return cancelled
    }

    private func sleepUnlessCancelled(_ seconds: TimeInterval) {
        let deadline = Date().addingTimeInterval(seconds)
        while !isCancelled && Date() < deadline {
            Thread.sleep(forTimeInterval: 0.1)
        }
    }

    // MARK: - Sending

    private func discoveryPayload(phoneID: String, phoneName: String, instanceID: String, nonce: String) -> [UInt8] {
        let object = WireFormat.discoveryQuery(
            phoneID: phoneID, phoneName: phoneName, instanceID: instanceID, nonce: nonce
        )
        guard let data = try? JSONSerialization.data(withJSONObject: object, options: []) else { return [] }
        return [UInt8](data)
    }

    private func sendSavedHost(fd: Int32, payload: [UInt8]) {
        let host = store.host
        guard !host.isEmpty, host != "0.0.0.0" else { return }
        guard let resolved = SocketSupport.resolveIPv4(host: host),
              let address = SocketSupport.ipv4Address(host: resolved, port: WireFormat.discoveryPort)
        else { return }
        _ = SocketSupport.sendTo(fd, bytes: payload, address: address)
    }

    /// Returns a human-readable error when broadcast is refused, which on iOS
    /// without the multicast entitlement is `EPERM` or `ENETUNREACH`.
    private func sendBroadcasts(fd: Int32, payload: [UInt8]) -> String? {
        var firstError: String?
        var targets = ["255.255.255.255"]
        targets.append(contentsOf: LocalNetwork.broadcastAddresses(LocalNetwork.interfaces()))
        for target in targets {
            guard let address = SocketSupport.ipv4Address(host: target, port: WireFormat.discoveryPort) else { continue }
            let sent = SocketSupport.sendTo(fd, bytes: payload, address: address)
            if sent < 0 && firstError == nil {
                let code = errno
                firstError = "Broadcast to \(target) refused: \(String(cString: strerror(code))) (errno \(code)). "
                    + "iOS needs the Multicast Networking entitlement for broadcast; the unicast sweep and manual address entry still work."
            }
        }
        return firstError
    }

    /// Paced bursts so a 254-address sweep does not become a single ARP storm.
    private func sendSweep(fd: Int32, payload: [UInt8], candidates: [String]) {
        let burst = 32
        var index = 0
        while index < candidates.count && !isCancelled {
            for offset in 0..<burst {
                let position = index + offset
                guard position < candidates.count else { break }
                guard let address = SocketSupport.ipv4Address(
                    host: candidates[position], port: WireFormat.discoveryPort
                ) else { continue }
                _ = SocketSupport.sendTo(fd, bytes: payload, address: address)
            }
            index += burst
            Thread.sleep(forTimeInterval: 0.01)
        }
    }

    // MARK: - Receiving

    private func receiveOffer(
        fd: Int32,
        nonce: String,
        expectedInstanceID: String,
        window: TimeInterval
    ) -> Offer? {
        let deadline = Date().addingTimeInterval(window)
        var buffer = [UInt8](repeating: 0, count: 4096)
        while !isCancelled && Date() < deadline {
            var source = sockaddr_in()
            var sourceLength = socklen_t(MemoryLayout<sockaddr_in>.size)
            let received = buffer.withUnsafeMutableBytes { raw -> Int in
                guard let base = raw.baseAddress else { return -1 }
                return withUnsafeMutablePointer(to: &source) { pointer in
                    pointer.withMemoryRebound(to: sockaddr.self, capacity: 1) { socketAddress in
                        Darwin.recvfrom(fd, base, raw.count, 0, socketAddress, &sourceLength)
                    }
                }
            }
            if received <= 0 {
                let code = errno
                if code == EAGAIN || code == EWOULDBLOCK || code == EINTR { continue }
                return nil
            }
            let data = Data(buffer[0..<received])
            guard let parsed = WireFormat.parseDiscoveryOffer(
                data, nonce: nonce, expectedInstanceID: expectedInstanceID
            ) else { continue }
            guard let address = Self.addressString(source) else { continue }
            return Offer(
                instanceID: parsed.instanceID,
                pcName: parsed.pcName,
                dataPort: parsed.dataPort,
                accepted: parsed.accepted,
                address: address
            )
        }
        return nil
    }

    private static func addressString(_ address: sockaddr_in) -> String? {
        var mutable = address.sin_addr
        var text = [CChar](repeating: 0, count: Int(INET_ADDRSTRLEN))
        guard inet_ntop(AF_INET, &mutable, &text, socklen_t(INET_ADDRSTRLEN)) != nil else { return nil }
        return String(cString: text)
    }
}
