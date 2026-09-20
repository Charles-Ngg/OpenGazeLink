"""An interrupted recording must not donate a partial or discarded trial."""
import unittest

from tools.evaluate_partial_events import completed_attempts, restore_capture_clocks


class PartialEventTests(unittest.TestCase):
    def test_clock_restore_requires_exact_frame_and_retains_packet_timing(self):
        def frame(sensor, send):
            return dict(timing=dict(phone_sensor_time_ns=sensor, phone_send_time_ns=send,
                                    pc_first_packet_monotonic_ns=999))
        saved = frame(100, 110)
        saved['timing'].update(phone_to_pc_offset_ns=200, clock_probe_uncertainty_ms=3,
                               clock_probe_age_ms=50, pc_first_packet_monotonic_ns=888)
        rows = [frame(100, 110), frame(100, 120), frame(101, 110), dict(timing={})]
        self.assertEqual(restore_capture_clocks(rows, [saved]), 1)
        self.assertEqual(rows[0]['timing']['clock_probe_uncertainty_ms'], 3)
        self.assertEqual(rows[0]['timing']['pc_first_packet_monotonic_ns'], 999)
        for row in rows[1:]:
            self.assertNotIn('phone_to_pc_offset_ns', row['timing'])

    def test_only_complete_attempts_survive_without_stitching_restarts(self):
        events = [
            dict(trial_id='train-a', capture_segment=1, trial_complete=False),
            dict(trial_id='train-a', capture_segment=2, trial_complete=True),
            dict(trial_id='validation-b', capture_segment=3, trial_complete=True),
            dict(trial_id='test-c', capture_segment=4, trial_complete=False),
            dict(trial_id='test-c', capture_segment=4, trial_complete=False),
        ]
        self.assertEqual(completed_attempts(events, discarded=[3]), {('train-a', 2)})
        self.assertEqual(completed_attempts([]), set())


if __name__ == '__main__':
    unittest.main()
