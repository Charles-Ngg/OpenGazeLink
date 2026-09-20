import gc
import unittest
import weakref
import sys
import threading
from unittest.mock import patch
from opengazelink_pc.runtime_diagnostics import GCPauseMonitor, FrameTimingProbe, TrackingGCPolicy
from opengazelink_pc.latency import PipelineLatencyTracker
from opengazelink_pc.runtime_scheduling import configure_tracking_process, restore_tracking_process, TrackingThreadSwitch


class RuntimeDiagnosticsTests(unittest.TestCase):
    def test_tracking_start_failure_restores_policies_and_monitor(self):
        from opengazelink_pc.engine import EyeTrackingEngine
        engine = EyeTrackingEngine.__new__(EyeTrackingEngine)
        engine._backend = None
        engine._scheduling = {}
        engine._tracking_stop = threading.Event()
        callbacks, interval, frozen = gc.callbacks[:], sys.getswitchinterval(), gc.get_freeze_count()
        with self.assertRaises(AssertionError):
            engine._tracking_loop(None, None)
        self.assertEqual(callbacks, gc.callbacks)
        self.assertEqual(interval, sys.getswitchinterval())
        self.assertEqual(frozen, gc.get_freeze_count())

    def test_interpreter_switch_interval_is_scoped_and_nested(self):
        before = sys.getswitchinterval()
        try:
            sys.setswitchinterval(.005)
            with self.assertRaisesRegex(RuntimeError, 'test failure'):
                with TrackingThreadSwitch():
                    self.assertAlmostEqual(.001, sys.getswitchinterval())
                    with TrackingThreadSwitch():
                        self.assertAlmostEqual(.001, sys.getswitchinterval())
                    self.assertAlmostEqual(.001, sys.getswitchinterval())
                    raise RuntimeError('test failure')
            self.assertAlmostEqual(.005, sys.getswitchinterval())
            sys.setswitchinterval(.0005)
            with TrackingThreadSwitch():
                self.assertAlmostEqual(.0005, sys.getswitchinterval())
            self.assertAlmostEqual(.0005, sys.getswitchinterval())
            with TrackingThreadSwitch():
                sys.setswitchinterval(.002)
            self.assertAlmostEqual(.002, sys.getswitchinterval())
        finally:
            sys.setswitchinterval(before)

    def test_tracking_gc_keeps_new_cycles_collectable_and_restores_on_exception(self):
        class Node:
            pass
        enabled, thresholds, frozen = gc.isenabled(), gc.get_threshold(), gc.get_freeze_count()
        with self.assertRaisesRegex(RuntimeError, 'test failure'):
            with TrackingGCPolicy() as policy:
                self.assertEqual(enabled, gc.isenabled())
                self.assertEqual(thresholds, gc.get_threshold())
                node = Node()
                node.cycle = node
                reference = weakref.ref(node)
                del node
                gc.collect()
                self.assertIsNone(reference())
                with TrackingGCPolicy():
                    self.assertTrue(policy.active)
                if not frozen and enabled:
                    self.assertGreater(gc.get_freeze_count(), 0)
                raise RuntimeError('test failure')
        self.assertEqual(frozen, gc.get_freeze_count())
        self.assertEqual(enabled, gc.isenabled())
        self.assertEqual(thresholds, gc.get_threshold())

    def test_tracking_gc_respects_external_policy_and_never_collects_at_start(self):
        with patch('opengazelink_pc.runtime_diagnostics.gc') as collector:
            collector.isenabled.return_value = True
            collector.get_freeze_count.return_value = 12
            with TrackingGCPolicy() as policy:
                self.assertEqual('external_policy', policy.status['mode'])
            collector.freeze.assert_not_called()
            collector.unfreeze.assert_not_called()
            collector.collect.assert_not_called()
            collector.get_freeze_count.return_value = 0
            collector.isenabled.return_value = False
            with TrackingGCPolicy():
                pass
            collector.freeze.assert_not_called()
            collector.enable.assert_not_called()
            collector.disable.assert_not_called()

    def test_gc_generation_breakdown(self):
        monitor = GCPauseMonitor()
        with patch('opengazelink_pc.runtime_diagnostics.time.perf_counter_ns', side_effect=[1_000_000, 3_000_000]):
            monitor._callback('start', {'generation': 2})
            monitor._callback('stop', {'generation': 2})
        self.assertEqual([0., 0., 2.], monitor.generation_ms)
        self.assertEqual([0, 0, 1], monitor.collections)

    def test_gc_monitor_does_not_disable_gc_and_unregisters(self):
        enabled = gc.isenabled()
        monitor = GCPauseMonitor()
        monitor.start()
        try:
            gc.collect()
            self.assertGreaterEqual(monitor.total_ms, 0)
            self.assertEqual(enabled, gc.isenabled())
        finally:
            monitor.close()
        self.assertNotIn(monitor._callback, gc.callbacks)

    def test_frame_probe_separates_wall_cpu_and_output_interval(self):
        monitor = GCPauseMonitor()
        probe = FrameTimingProbe(monitor)
        prepared = dict(process_start_ns=100_000_000, thread_cpu_start_ns=10_000_000,
                        face_geometry=dict(source_age_ms=20, detection_ms=12, thread_cpu_ms=2))
        with patch("opengazelink_pc.runtime_diagnostics.time.thread_time_ns", return_value=14_000_000):
            first = probe.sample(prepared, 110_000_000)
            self.assertEqual(4, first["inference_thread_cpu_ms"])
            self.assertEqual(6, first["inference_wait_ms_proxy"])
            self.assertIsNone(first["output_interval_ms"])
            monitor.total_ms = 3
            second = probe.sample(prepared, 140_000_000)
            self.assertEqual(30, second["output_interval_ms"])
            self.assertEqual(3, second["gc_pause_ms"])

    def test_logging_is_background_and_rate_limited(self):
        tracker = PipelineLatencyTracker(auto_start=False)
        with patch("opengazelink_pc.latency.logging.getLogger") as logger:
            tracker.add(dict(output_interval_ms=100, pc_processing_ms=50), 1000)
            logger.assert_not_called()
            tracker.refresh(1000)
            self.assertEqual(1, logger.return_value.info.call_count)
            tracker.refresh(2000)
            self.assertEqual(1, logger.return_value.info.call_count)

    def test_tracking_does_not_set_affinity_and_restores_priority(self):
        with patch("opengazelink_pc.runtime_scheduling._kernel") as kernel, \
             patch("opengazelink_pc.runtime_scheduling.os.name", "nt"):
            api = kernel.return_value
            api.GetPriorityClass.return_value = 32
            state = configure_tracking_process()
            self.assertFalse(state["affinity"])
            api.SetProcessAffinityMask.assert_not_called()
            restore_tracking_process(state)
            self.assertEqual(32, api.SetPriorityClass.call_args.args[1])
