import time
import unittest
from opengazelink_pc import runtime_clock


class RuntimeClockTests(unittest.TestCase):
    def test_pipeline_clock_tracks_high_resolution_elapsed_time(self):
        before = runtime_clock.monotonic_ns()
        start = time.perf_counter_ns()
        while time.perf_counter_ns() - start < 2_000_000:
            pass
        elapsed = runtime_clock.monotonic_ns() - before
        reference = time.perf_counter_ns() - start
        self.assertGreater(elapsed, 0)
        self.assertLess(abs(elapsed - reference), 1_000_000)
        self.assertLess(runtime_clock.info()["resolution_ms"], 1.0)
        self.assertLess(abs(runtime_clock.monotonic() - time.monotonic()), 0.1)
