"""Exercise fresh-pixel consumption with independently computed face geometry."""
import threading
import time
from types import SimpleNamespace
import unittest

from opengazelink_pc.engine import EyeTrackingEngine
from opengazelink_pc.latest_preprocessor import LatestEyePreprocessor
from opengazelink_pc.latency import PipelineLatencyTracker
from opengazelink_pc import runtime_clock


class H264PacingTest(unittest.TestCase):
    def test_geometry_completed_during_pixel_wait_is_used_unless_from_future(self):
        for new_timestamp, expected_timestamp in ((1000., 1000.), (1010., 990.)):
            with self.subTest(new_timestamp=new_timestamp):
                old = SimpleNamespace(t_ms=990., detection_ms=10.)
                new = SimpleNamespace(t_ms=new_timestamp, detection_ms=10.)
                prepared = []
                def read_latest(after_seq, timeout_s):
                    # Simulate MediaPipe finishing while get waits for pixels.
                    pipeline._snapshot = (new, 2, runtime_clock.monotonic_ns(), 0, '', 1.)
                    return True, object(), 1000., 2
                camera = SimpleNamespace(read_latest=read_latest,
                    latest_frame_timing=lambda seq: {}, camera_model=lambda: {})
                backend = SimpleNamespace(prepare_with_geometry=lambda frame, t, model, geometry: prepared.append(geometry.t_ms))
                pipeline = LatestEyePreprocessor(camera, backend, threading.Event(), None)
                pipeline._snapshot = (old, 1, runtime_clock.monotonic_ns(), 0, '', 1.)
                item = pipeline.get()
                self.assertEqual(item['error'], '')
                self.assertEqual(prepared, [expected_timestamp])

    def test_tracking_reads_fresh_pixels_and_never_builds_a_queue(self):
        engine = EyeTrackingEngine.__new__(EyeTrackingEngine)
        engine._tracking_stop = threading.Event()
        engine._lock = threading.Lock()
        engine._writer = None
        engine._motion_diagnostics = SimpleNamespace(record=lambda value: None)
        engine._publish_gaze = lambda value: None
        engine._inference_times = []
        engine._latency = PipelineLatencyTracker(auto_start=False)
        engine._event_temporal = SimpleNamespace(reset=lambda: None)
        samples = []
        origin = time.perf_counter()

        def read_latest(after_seq, timeout_s):
            now = time.perf_counter()
            seq = int((now-origin)*120)
            if seq == after_seq:
                engine._tracking_stop.wait(min(timeout_s, .001))
                return False, None, seq*1000/120, seq
            return True, seq, seq*1000/120, seq

        def detect_geometry(frame, timestamp, camera_model):
            time.sleep(.010)
            return SimpleNamespace(t_ms=timestamp, detection_ms=10., source_frame=frame)

        def prepare_with_geometry(frame, timestamp, camera_model, geometry):
            now = time.perf_counter()
            samples.append((now, frame, geometry.source_frame))
            time.sleep(.010)
            if len(samples) >= 24:
                engine._tracking_stop.set()
            return None  # Exercise the same pacing on failed face detection.

        engine._camera = SimpleNamespace(inference_max_fps=60.0, read_latest=read_latest,
            latest_frame_timing=lambda sequence: {}, camera_model=lambda: {'source': 'camera2'})
        engine._backend = SimpleNamespace(detect_geometry=detect_geometry, prepare_with_geometry=prepare_with_geometry)
        # A fixture/API mismatch must fail promptly instead of keeping the
        # real tracking loop alive forever with no valid observations.
        watchdog = threading.Timer(3., engine._tracking_stop.set)
        watchdog.start()
        try:
            engine._tracking_loop(SimpleNamespace(temporal=False), None)
        finally:
            watchdog.cancel()
        self.assertEqual(len(samples), 24)
        gaps = [b[0]-a[0] for a, b in zip(samples, samples[1:])]
        self.assertGreaterEqual(min(gaps), .008)  # Allow native Sleep timer rounding.
        self.assertTrue(any(b[1]-a[1] > 1 for a, b in zip(samples, samples[1:])))
        self.assertTrue(all(geometry <= frame for _, frame, geometry in samples))
        self.assertTrue(any(geometry < frame for _, frame, geometry in samples))
        ages = [(now-origin)-seq/120 for now, seq, _ in samples]
        self.assertLess(max(ages), .020, 'Consumer reused old pixels while geometry was computing')

    def test_preprocessing_overlaps_downstream_work(self):
        stop=threading.Event();sequence=[0]
        class Camera:
            inference_max_fps=0
            def read_latest(self,after_seq,timeout_s):
                sequence[0]+=1
                return True,object(),sequence[0],sequence[0]
            def latest_frame_timing(self,sequence):return {}
            def camera_model(self):return {"source":"camera2"}
        class Backend:
            def detect_geometry(self,frame,timestamp,camera_model):
                time.sleep(.012)
                return SimpleNamespace(t_ms=timestamp,detection_ms=12.)
            def prepare_with_geometry(self,*args):return object()
        class Pacer:
            def wait_until(self,*args):return False
        pipeline=LatestEyePreprocessor(Camera(),Backend(),stop,Pacer());pipeline.start()
        started=time.perf_counter();received=0
        try:
            while received<25 and time.perf_counter()-started<3.:
                item=pipeline.get(.5)
                if isinstance(item,dict) and item['observation'] is not None:
                    received+=1
                    time.sleep(.008)  # Simulated model/projection stage.
                else:
                    time.sleep(.001)
        finally:
            pipeline.close()
        elapsed=time.perf_counter()-started
        serial_started=time.perf_counter()
        for _ in range(5):
            time.sleep(.012);time.sleep(.008)
        serial_estimate=(time.perf_counter()-serial_started)*5
        self.assertEqual(25,received)
        self.assertLess(elapsed,serial_estimate*.82,"preprocessing and model work ran serially")


if __name__ == '__main__':
    unittest.main()
