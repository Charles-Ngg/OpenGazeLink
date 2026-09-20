from __future__ import annotations

import time
import unittest

from opengazelink_pc.latency import PipelineLatencyTracker, finite_milliseconds
from opengazelink_pc.engine import EyeTrackingEngine


class PipelineLatencyTrackerTests(unittest.TestCase):
    def test_window_statistics_and_current_values(self) -> None:
        tracker = PipelineLatencyTracker(window_seconds=2.0, max_samples=10, auto_start=False)
        tracker.add({"pc_processing_ms": 10.0, "clock_basis": "a"}, observed_at_ms=1000.0)
        tracker.add({"pc_processing_ms": 30.0, "clock_basis": "b"}, observed_at_ms=2000.0)
        result = tracker.refresh(now_ms=2000.0)
        self.assertEqual(2, result["sample_count"])
        self.assertEqual(30.0, result["current"]["pc_processing_ms"])
        self.assertEqual("b", result["current"]["clock_basis"])
        self.assertEqual(20.0, result["metrics"]["pc_processing_ms"]["mean"])
        result = tracker.summary(now_ms=4001.0)
        self.assertEqual(0, result["sample_count"])

    def test_invalid_numeric_values_are_not_reported_as_latency(self) -> None:
        self.assertIsNone(finite_milliseconds(float("nan")))
        self.assertIsNone(finite_milliseconds(-1))
        tracker = PipelineLatencyTracker(auto_start=False)
        tracker.add({"pc_processing_ms": float("inf")}, observed_at_ms=0.0)
        result = tracker.refresh(now_ms=0.0)
        self.assertIsNone(result["current"]["pc_processing_ms"])
        self.assertEqual(0, result["metrics"]["pc_processing_ms"]["samples"])

    def test_add_keeps_current_immediate_without_synchronous_aggregation(self) -> None:
        tracker = PipelineLatencyTracker(auto_start=False)
        first = tracker.add({"pc_processing_ms": 10.0}, observed_at_ms=1000.0)
        self.assertEqual(10.0, first["current"]["pc_processing_ms"])
        self.assertEqual(0, first["metrics"]["pc_processing_ms"]["samples"])
        tracker.refresh(now_ms=1000.0)
        second = tracker.add({"pc_processing_ms": 20.0}, observed_at_ms=1010.0)
        self.assertEqual(20.0, second["current"]["pc_processing_ms"])
        self.assertEqual(1, second["metrics"]["pc_processing_ms"]["samples"])
        self.assertEqual("background", second["aggregation"]["mode"])

    def test_background_worker_publishes_aggregates_and_closes(self) -> None:
        tracker = PipelineLatencyTracker(aggregation_interval_s=0.1)
        try:
            # Production starts the service before tracking. An empty first
            # aggregation must not terminate the worker.
            time.sleep(0.15)
            self.assertTrue(tracker._thread.is_alive())
            self.assertEqual(0, tracker.summary()["aggregation"]["sample_count"])
            tracker.add({"pc_processing_ms": 12.0})
            deadline = time.perf_counter() + 1.0
            while tracker.summary()["metrics"]["pc_processing_ms"]["samples"] == 0:
                self.assertLess(time.perf_counter(), deadline)
                time.sleep(0.01)
            self.assertEqual(12.0, tracker.summary()["metrics"]["pc_processing_ms"]["mean"])
        finally:
            tracker.close()
        self.assertIsNone(tracker._thread)

    def test_separate_stage_sum_closes_against_source_to_shared_memory(self) -> None:
        engine = EyeTrackingEngine.__new__(EyeTrackingEngine)
        sample = engine._latency_sample(
            {
                "pc_first_packet_monotonic_ns": 1_040_000_000,
                "pc_last_packet_monotonic_ns": 1_052_000_000,
                "pc_decode_start_monotonic_ns": 1_055_000_000,
                "pc_decode_done_monotonic_ns": 1_060_000_000,
                "h264_reference_decode_ms": 2.,
                "h264_image_queue_ms": 1.,
                "h264_image_conversion_ms": 2.,
            },
            {
                "clock_basis": "phone_minimum_transit_proxy",
                "source_pc_ms_proxy": 1000.0,
                "phone_capture_to_send_ms": 10.0,
                "transport_excess_ms_proxy": 30.0,
                "display_delay_ms_assumed": 16.0,
                "unknown_transport_floor": True,
            },
            {"horizon_ms": 116.0},
            1_080_000_000, 1_100_000_000,
            backend_ms=6.0, detection_ms=2.0, normalization_ms=3.0,
            gaze_model_ms=5.0, projection_fusion_ms=2.0,
            prediction_postprocess_ms=3.0, shared_memory_write_ms=1.0,
        )
        self.assertAlmostEqual(100.0, sample["source_to_shared_memory_ms_proxy"])
        self.assertAlmostEqual(100.0, sample["stage_sum_ms"])
        self.assertAlmostEqual(0.0, sample["sum_error_ms"])
        self.assertAlmostEqual(12.0, sample["packet_assembly_ms"])
        self.assertAlmostEqual(3.0, sample["decode_queue_ms"])
        self.assertAlmostEqual(5.0, sample["jpeg_decode_ms"])
        self.assertEqual(2., sample['h264_reference_decode_ms'])
        self.assertEqual(1., sample['h264_image_queue_ms'])
        self.assertEqual(2., sample['h264_image_conversion_ms'])
        tracker = PipelineLatencyTracker(auto_start=False)
        tracker.add(sample, observed_at_ms=1100.)
        summary = tracker.refresh(now_ms=1100.)
        self.assertEqual(2., summary['metrics']['h264_reference_decode_ms']['mean'])
        self.assertAlmostEqual(sample["receive_decode_ms"], sum(sample[key] for key in
            ("packet_assembly_ms", "decode_queue_ms", "jpeg_decode_ms")))
        self.assertAlmostEqual(116.0, sample["source_to_display_ms_estimate"])

    def test_local_camera_uses_read_completion_to_measure_queue(self) -> None:
        engine = EyeTrackingEngine.__new__(EyeTrackingEngine)
        sample = engine._latency_sample(
            {"source_capture_monotonic_ns": 2_000_000_000},
            {"clock_basis": "camera_read_completion_proxy", "source_pc_ms_proxy": 2000.0,
             "display_delay_ms_assumed": 16.0},
            {"horizon_ms": 31.0}, 2_015_000_000, 2_030_000_000,
            backend_ms=5.0, detection_ms=2.0, normalization_ms=2.0,
            gaze_model_ms=3.0, projection_fusion_ms=1.0,
            prediction_postprocess_ms=2.0, shared_memory_write_ms=1.0,
        )
        self.assertAlmostEqual(15.0, sample["pc_queue_ms"])
        self.assertAlmostEqual(30.0, sample["stage_sum_ms"])
        self.assertAlmostEqual(30.0, sample["source_to_shared_memory_ms_proxy"])


if __name__ == "__main__":
    unittest.main()
