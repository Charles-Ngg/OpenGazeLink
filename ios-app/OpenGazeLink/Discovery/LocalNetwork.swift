import Darwin
import Foundation

/// Local IPv4 interface inventory and unicast sweep candidates.
///
/// This exists because iOS blocks UDP broadcast without Apple's Multicast
/// Networking entitlement. `Network.framework` cannot broadcast at all, so the
/// discovery socket is a BSD one, and when a broadcast send is refused the
/// pairing service falls back to sending the same JSON payload directly to
/// every address in the phone's own subnet.
enum LocalNetwork {

    struct Interface {
        let name: String
        let address: String
        let netmask: String
        let broadcast: String
    }

    /// Non-loopback IPv4 interfaces, ordered so Wi-Fi comes first.
    static func interfaces() -> [Interface] {
        var head: UnsafeMutablePointer<ifaddrs>?
        guard getifaddrs(&head) == 0, let first = head else { return [] }
        defer { freeifaddrs(head) }

        var result: [Interface] = []
        var pointer: UnsafeMutablePointer<ifaddrs>? = first
        while let current = pointer {
            defer { pointer = current.pointee.ifa_next }
            let flags = Int32(current.pointee.ifa_flags)
            guard flags & IFF_UP != 0, flags & IFF_LOOPBACK == 0 else { continue }
            guard let addressPointer = current.pointee.ifa_addr,
                  addressPointer.pointee.sa_family == sa_family_t(AF_INET)
            else { continue }

            var address = sockaddr_in()
            memcpy(&address, addressPointer, MemoryLayout<sockaddr_in>.size)
            guard let ip = string(from: address.sin_addr) else { continue }
            guard SocketSupport.isUsableLANAddress(ip) else { continue }

            var netmask = sockaddr_in()
            if let netmaskPointer = current.pointee.ifa_netmask,
               netmaskPointer.pointee.sa_family == sa_family_t(AF_INET) {
                memcpy(&netmask, netmaskPointer, MemoryLayout<sockaddr_in>.size)
            } else {
                netmask.sin_addr.s_addr = inet_addr("255.255.255.0")
            }
            guard let mask = string(from: netmask.sin_addr) else { continue }

            let ipValue = UInt32(bigEndian: address.sin_addr.s_addr)
            let maskValue = UInt32(bigEndian: netmask.sin_addr.s_addr)
            let broadcastValue = (ipValue & maskValue) | ~maskValue
            var broadcastAddress = in_addr(s_addr: broadcastValue.bigEndian)

            result.append(Interface(
                name: String(cString: current.pointee.ifa_name),
                address: ip,
                netmask: mask,
                broadcast: string(from: broadcastAddress) ?? "255.255.255.255"
            ))
        }
        // en0 (Wi-Fi) first: it is the interface the PC provider is normally on.
        return result.sorted { lhs, rhs in
            if (lhs.name == "en0") != (rhs.name == "en0") { return lhs.name == "en0" }
            return lhs.name < rhs.name
        }
    }

    /// Directed broadcast addresses, deduplicated and excluding the limited
    /// broadcast address which is added separately.
    static func broadcastAddresses(_ interfaces: [Interface]) -> [String] {
        var seen = Set<String>()
        return interfaces.compactMap { interface in
            guard interface.broadcast != "255.255.255.255" else { return nil }
            // A /32 or point-to-point interface has no broadcast address, so
            // probing it would just duplicate the unicast send.
            guard interface.broadcast != interface.address else { return nil }
            return seen.insert(interface.broadcast).inserted ? interface.broadcast : nil
        }
    }

    /// Unicast addresses to probe when broadcast is unavailable.
    ///
    /// The phone's own /24 is enumerated host-first, because the PC provider is
    /// usually a low host number and a short sweep is far less intrusive than a
    /// full one. The address of the phone itself is never probed, and the
    /// network and broadcast addresses are skipped.
    static func sweepCandidates(
        over candidates: [Interface],
        maximumPerInterface: Int = 254
    ) -> [String] {
        var result: [String] = []
        var seen = Set<String>()
        // Every address the phone itself holds. Another interface's sweep must
        // not probe them either, or the discovery socket would receive its own
        // query back.
        let ownAddresses = Set(candidates.map(\.address))
        for interface in candidates {
            guard let ip = ipv4Value(interface.address), let mask = ipv4Value(interface.netmask) else { continue }
            let network = ip & mask
            let broadcast = network | ~mask
            // Only ever sweep the single /24 slice containing this interface's
            // own address. A /16 would mean 65k datagrams, and iOS Personal
            // Hotspot (/28) and typical home LANs (/24) both fit inside it.
            let sliceBase = ip & 0xFFFF_FF00
            var produced = 0
            var offset: UInt32 = 1
            while offset <= 254 && produced < maximumPerInterface {
                let candidateValue = sliceBase | offset
                offset += 1
                guard candidateValue & mask == network else { continue }
                guard candidateValue != broadcast else { continue }
                var address = in_addr(s_addr: candidateValue.bigEndian)
                guard let text = string(from: address) else { continue }
                guard !ownAddresses.contains(text) else { continue }
                guard SocketSupport.isUsableLANAddress(text) else { continue }
                guard seen.insert(text).inserted else { continue }
                result.append(text)
                produced += 1
            }
        }
        return result
    }

    // MARK: - Helpers

    private static func string(from address: in_addr) -> String? {
        var mutable = address
        var buffer = [CChar](repeating: 0, count: Int(INET_ADDRSTRLEN))
        guard inet_ntop(AF_INET, &mutable, &buffer, socklen_t(INET_ADDRSTRLEN)) != nil else { return nil }
        return String(cString: buffer)
    }

    private static func ipv4Value(_ text: String) -> UInt32? {
        var parsed = in_addr()
        guard inet_pton(AF_INET, text, &parsed) == 1 else { return nil }
        return UInt32(bigEndian: parsed.s_addr)
    }
}

/// Persisted phone identity and last known PC address.
///
/// Mirrors the Android `SharedPreferences` keys `phone_id`,
/// `paired_instance_id`, `host`, `port`.
struct PairingStore {

    private enum Key {
        static let phoneID = "phone_id"
        static let pairedInstanceID = "paired_instance_id"
        static let host = "host"
        static let port = "port"
        static let language = "language"
        static let fovReference = "fov_reference"
    }

    private let defaults: UserDefaults

    init(defaults: UserDefaults = .standard) {
        self.defaults = defaults
    }

    /// Stable per-install identifier. Generated once, exactly like Android.
    var phoneID: String {
        if let existing = defaults.string(forKey: Key.phoneID), !existing.isEmpty { return existing }
        let generated = UUID().uuidString
        defaults.set(generated, forKey: Key.phoneID)
        return generated
    }

    var pairedInstanceID: String {
        get { defaults.string(forKey: Key.pairedInstanceID) ?? "" }
        nonmutating set { defaults.set(newValue, forKey: Key.pairedInstanceID) }
    }

    var host: String {
        get { defaults.string(forKey: Key.host) ?? "" }
        nonmutating set { defaults.set(newValue, forKey: Key.host) }
    }

    var port: UInt16? {
        get {
            let value = defaults.integer(forKey: Key.port)
            return (1...65535).contains(value) ? UInt16(value) : nil
        }
        nonmutating set {
            if let newValue { defaults.set(Int(newValue), forKey: Key.port) }
            else { defaults.removeObject(forKey: Key.port) }
        }
    }

    var language: AppLanguage {
        get {
            guard let raw = defaults.string(forKey: Key.language),
                  let language = AppLanguage(rawValue: raw) else { return .english }
            return language
        }
        nonmutating set { defaults.set(newValue.rawValue, forKey: Key.language) }
    }

    var fovReference: FovReference {
        get {
            guard let raw = defaults.string(forKey: Key.fovReference),
                  let reference = FovReference(rawValue: raw) else { return .horizontal }
            return reference
        }
        nonmutating set { defaults.set(newValue.rawValue, forKey: Key.fovReference) }
    }

    /// Clears the paired PC, matching Android's "Change paired PC".
    nonmutating func forgetPairedPC() {
        defaults.removeObject(forKey: Key.pairedInstanceID)
        defaults.removeObject(forKey: Key.host)
        defaults.removeObject(forKey: Key.port)
    }
}

/// Human-readable phone name, sent as `phone_name` in the discovery payload.
///
/// Android sends `Build.MANUFACTURER + " " + Build.MODEL`. The iOS equivalent
/// is the hardware identifier, mapped to a marketing name where it is known so
/// the PC control centre shows "iPhone 16 Pro Max" instead of "iPhone17,2".
enum DeviceIdentity {

    static var phoneName: String {
        let identifier = machineIdentifier
        if let marketing = marketingNames[identifier] { return marketing }
        return "iPhone (\(identifier))"
    }

    static var machineIdentifier: String {
        var info = utsname()
        guard uname(&info) == 0 else { return "iPhone" }
        let mirror = Mirror(reflecting: info.machine)
        let bytes = mirror.children.compactMap { $0.value as? Int8 }
        let characters = bytes.prefix { $0 != 0 }.map { Character(UnicodeScalar(UInt8(bitPattern: $0))) }
        let text = String(characters)
        return text.isEmpty ? "iPhone" : text
    }

    private static let marketingNames: [String: String] = [
        "iPhone17,1": "iPhone 16 Pro",
        "iPhone17,2": "iPhone 16 Pro Max",
        "iPhone17,3": "iPhone 16",
        "iPhone17,4": "iPhone 16 Plus",
        "iPhone17,5": "iPhone 16e",
        "iPhone16,1": "iPhone 15 Pro",
        "iPhone16,2": "iPhone 15 Pro Max",
        "iPhone15,4": "iPhone 15",
        "iPhone15,5": "iPhone 15 Plus",
        "iPhone15,2": "iPhone 14 Pro",
        "iPhone15,3": "iPhone 14 Pro Max",
        "iPhone14,7": "iPhone 14",
        "iPhone14,8": "iPhone 14 Plus",
        "iPhone14,6": "iPhone SE (3rd generation)",
        "iPhone14,5": "iPhone 13",
        "iPhone14,4": "iPhone 13 mini",
        "iPhone14,2": "iPhone 13 Pro",
        "iPhone14,3": "iPhone 13 Pro Max",
        "iPhone13,2": "iPhone 12",
        "iPhone13,3": "iPhone 12 Pro",
        "iPhone13,4": "iPhone 12 Pro Max",
        "iPhone13,1": "iPhone 12 mini",
        "iPhone12,1": "iPhone 11",
        "iPhone12,3": "iPhone 11 Pro",
        "iPhone12,5": "iPhone 11 Pro Max",
        "iPhone11,2": "iPhone XS",
        "iPhone11,4": "iPhone XS Max",
        "iPhone11,8": "iPhone XR",
    ]
}
