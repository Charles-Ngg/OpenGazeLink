"""Four-timestamp UDP clock probes; no video retransmission or frame buffering."""
import struct


CLOCK_MAGIC = 0x54435945  # EYCT
CLOCK_PACKET = struct.Struct("<IHHQQQ")


class TransportClock:
    def __init__(self):
        self.pending = {}
        self.samples = ()

    def request(self, now_ns):
        self.pending[now_ns] = now_ns
        self.pending = {key: value for key, value in self.pending.items() if now_ns-value < 5_000_000_000}
        return CLOCK_PACKET.pack(CLOCK_MAGIC, 1, 1, now_ns, 0, 0)

    def receive(self, packet, now_ns):
        if len(packet) != CLOCK_PACKET.size:
            return False
        magic, version, kind, t1, t2, t3 = CLOCK_PACKET.unpack(packet)
        if magic != CLOCK_MAGIC or version != 1 or kind != 2 or t1 not in self.pending:
            return False
        self.pending.pop(t1)
        rtt = (now_ns-t1) - (t3-t2)
        if not (0 < t2 <= t3 and 0 <= rtt <= 200_000_000 and now_ns >= t1):
            return False
        offset = ((t1-t2) + (now_ns-t3)) / 2.0  # PC minus phone
        self.samples = (*self.samples[-15:], (now_ns, rtt, offset))
        return True

    def latest(self, now_ns):
        # A lifetime minimum is invalid for clocks with different rates. Use
        # the lowest-RTT probe within five seconds and expose its age/error.
        recent = [s for s in self.samples if 0 <= now_ns-s[0] <= 5_000_000_000]
        if not recent:
            return {}
        received, rtt, offset = min(recent, key=lambda sample: sample[1])
        return {"phone_to_pc_offset_ns": offset, "clock_probe_rtt_ms": rtt/1e6,
                "clock_probe_uncertainty_ms": rtt/2e6,
                "clock_probe_age_ms": (now_ns-received)/1e6,
                "clock_probe_samples": len(recent)}
