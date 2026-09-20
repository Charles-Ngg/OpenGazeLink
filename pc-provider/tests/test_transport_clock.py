import unittest
from opengazelink_pc.transport_clock import TransportClock, CLOCK_MAGIC, CLOCK_PACKET
from opengazelink_pc.prediction_timing import PredictionClock


class TransportClockTests(unittest.TestCase):
    def exchange(self, clock, t1, offset, outgoing=2_000_000, returning=2_000_000, processing=500_000):
        clock.request(t1)
        t2 = t1 + outgoing - offset
        t3 = t2 + processing
        t4 = t1 + outgoing + returning + processing
        response = CLOCK_PACKET.pack(CLOCK_MAGIC, 1, 2, t1, t2, t3)
        self.assertTrue(clock.receive(response, t4))
        return t4, response

    def test_roundtrip_removes_phone_processing_and_bounds_asymmetry(self):
        clock = TransportClock()
        t4, response = self.exchange(clock, 10_000_000_000, -100_000_000, outgoing=1_000_000, returning=5_000_000)
        sample = clock.latest(t4)
        self.assertEqual(sample['clock_probe_rtt_ms'], 6)
        self.assertLessEqual(abs(sample['phone_to_pc_offset_ns']+100_000_000)/1e6, sample['clock_probe_uncertainty_ms'])
        self.assertFalse(clock.receive(response, t4))  # duplicate is not a fresh probe
        self.assertEqual(clock.latest(t4+5_000_000_001), {})

    def test_clock_drift_does_not_look_like_a_growing_video_queue(self):
        clock, prediction = TransportClock(), PredictionClock()
        measured = []
        for second in range(120):
            t1 = 10_000_000_000 + second*1_000_000_000
            offset = -100_000_000 + second*100_000  # 100 ppm relative clock drift
            t4, _ = self.exchange(clock, t1, offset)
            send = t4+20_000_000-offset
            delay = 20_000_000 if second == 119 else 3_000_000
            receive = send+offset+delay
            timing = {**clock.latest(receive), 'phone_sensor_time_ns':send-60_000_000,
                'phone_send_time_ns':send, 'pc_first_packet_monotonic_ns':receive}
            row = prediction.observe((send-60_000_000)/1e6, timing, (receive+8_000_000)/1e6)
            self.assertEqual(row['clock_basis'], 'phone_roundtrip_alignment')
            measured.append(row['transport_to_first_packet_ms'])
        self.assertLess(max(measured[:-1])-min(measured[:-1]), 0.6)
        self.assertGreater(measured[-1], 19)  # A real queue is still reported.
        self.assertAlmostEqual(row['transport_excess_ms_proxy'], 28.9, places=5)  # 17 ms queue increase + 11.9 ms drift.

    def test_expired_probe_falls_back_explicitly(self):
        clock = PredictionClock()
        row = clock.observe(1000, {'phone_sensor_time_ns':1_000_000_000,
            'phone_send_time_ns':1_060_000_000,'pc_first_packet_monotonic_ns':2_000_000_000}, 2020)
        self.assertEqual(row['clock_basis'], 'phone_minimum_transit_proxy')
        self.assertTrue(row['unknown_transport_floor'])
