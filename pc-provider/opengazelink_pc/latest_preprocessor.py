"""Independent latest-frame face geometry and gaze preprocessing for live use."""
from __future__ import annotations

import threading
import time

from . import runtime_clock

_END = object()


class LatestEyePreprocessor:
    """MediaPipe publishes geometry; the gaze consumer reads fresh camera pixels.

    Neither stage queues frames nor waits for the other stage's inference.
    Geometry is held, not extrapolated. Offline replay still uses backend.predict.
    """

    MAX_GEOMETRY_AGE_MS = 150.0

    def __init__(self, camera, backend, stop: threading.Event, pacer) -> None:
        self.camera, self.backend, self.stop, self.pacer = camera, backend, stop, pacer
        self._read_latest = getattr(camera, 'read_latest_shared', camera.read_latest)
        self.dropped = 0
        self._lock = threading.Lock()
        self._snapshot = None
        self._epoch = 0
        self._consumer_epoch = -1
        self._sequence = -1
        self._timestamp = None
        self._thread = threading.Thread(target=self._run, name="gaze-mediapipe", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def get(self, timeout_s: float = .1):
        if self.stop.is_set():
            return _END
        # Snapshot before pixels: geometry must never come from a future frame.
        with self._lock:
            snapshot = self._snapshot
        ok, frame, t_ms, sequence = self._read_latest(self._sequence, timeout_s=timeout_s)
        if not ok or frame is None or sequence == self._sequence:
            return None
        # read_latest can wait across a stream stall. Use geometry completed
        # during that wait when it belongs to these pixels (never future ones).
        with self._lock:
            candidate = self._snapshot
        if candidate is not None and candidate[0] is not None and candidate[0].t_ms <= t_ms:
            snapshot = candidate
        if self._sequence >= 0:
            self.dropped += max(0, sequence - self._sequence - 1)
        self._sequence = sequence
        frame_timing = self.camera.latest_frame_timing(sequence)
        process_start_ns = runtime_clock.monotonic_ns()
        thread_cpu_start_ns = time.thread_time_ns()
        started, wall_time_ns = time.perf_counter(), time.time_ns()
        observation, error = None, ""
        source_age, completed_age = None, None
        geometry_sequence, geometry_timestamp, detector_ms = None, None, None
        discontinuity = self._timestamp is not None and not 0 < t_ms - self._timestamp <= 250.
        self._timestamp = t_ms
        try:
            if snapshot is None:
                raise RuntimeError("waiting for face geometry")
            geometry, geometry_sequence, completed_ns, epoch, face_error, face_cpu_ms = snapshot
            discontinuity = discontinuity or epoch != self._consumer_epoch
            self._consumer_epoch = epoch
            if face_error or geometry is None:
                raise RuntimeError(face_error or "landmarker produced no face")
            source_age = float(t_ms) - geometry.t_ms
            completed_age = (process_start_ns - completed_ns) / 1_000_000.
            geometry_timestamp, detector_ms = geometry.t_ms, geometry.detection_ms
            if not 0 <= source_age <= self.MAX_GEOMETRY_AGE_MS or completed_age > self.MAX_GEOMETRY_AGE_MS:
                raise RuntimeError("face geometry is stale; waiting for MediaPipe")
            observation = self.backend.prepare_with_geometry(frame, t_ms, self.camera.camera_model(), geometry)
        except Exception as caught:
            error = str(caught)
        done_ns = runtime_clock.monotonic_ns()
        return dict(sequence=sequence, t_ms=t_ms, frame_timing=frame_timing,
                    thread_cpu_start_ns=thread_cpu_start_ns,
                    process_start_ns=process_start_ns, wall_time_ns=wall_time_ns, started=started,
                    backend_start_ns=process_start_ns, backend_done_ns=done_ns,
                    observation=observation, error=error, pipeline_discontinuity=discontinuity,
                    face_geometry=dict(source_sequence=geometry_sequence, source_t_ms=geometry_timestamp,
                                       source_age_ms=source_age, completed_age_ms=completed_age,
                                       detection_ms=detector_ms, thread_cpu_ms=snapshot[-1] if snapshot else None,
                                       scheduling="independent_latest_geometry"))

    def _run(self) -> None:
        from .runtime_scheduling import set_realtime_thread_priority
        set_realtime_thread_priority()
        sequence, timestamp = -1, None
        while not self.stop.is_set():
            ok, frame, t_ms, next_sequence = self._read_latest(sequence, timeout_s=.05)
            if not ok or frame is None or next_sequence == sequence:
                continue
            discontinuity = timestamp is not None and not 0 < t_ms - timestamp <= 250.
            sequence, timestamp = next_sequence, t_ms
            geometry, error = None, ""
            cpu_started = time.thread_time_ns()
            try:
                camera_model = self.camera.camera_model()
                if camera_model.get("source") == "estimated_frame_center":
                    raise RuntimeError("waiting for Camera2 intrinsics")
                geometry = self.backend.detect_geometry(frame, t_ms, camera_model)
                if geometry is None:
                    raise RuntimeError("landmarker produced no face")
                prepare_geometry = getattr(self.backend, 'prepare_live_geometry', None)
                if prepare_geometry is not None:
                    geometry = prepare_geometry(frame, geometry)
            except Exception as caught:
                error = str(caught)
            completed_ns = runtime_clock.monotonic_ns()
            with self._lock:
                if discontinuity or error or getattr(self.backend, "state_discontinuity", False):
                    self._epoch += 1
                self._snapshot = (geometry, sequence, completed_ns, self._epoch, error,
                                  (time.thread_time_ns() - cpu_started) / 1e6)

    def close(self) -> None:
        self.stop.set()
        if self._thread.ident is not None:
            self._thread.join()

    @property
    def ended(self) -> bool:
        return not self._thread.is_alive()


def is_pipeline_end(item) -> bool:
    return item is _END
