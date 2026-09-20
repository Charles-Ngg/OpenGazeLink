"""Cheap frame timing probes; aggregation and log I/O stay off tracking threads."""
import gc
import time
import threading


class TrackingGCPolicy:
    """Keep loaded library/model graphs out of cyclic GC during tracking.

    Automatic collection stays enabled for newly allocated objects, including
    frame/session cycles. The startup graph is unfrozen when the last tracking
    scope exits (also on exceptions). Never take ownership of an external freeze.
    This is process-wide, so overlapping tracking scopes share one lease.
    """
    _lock = threading.Lock()
    _users = 0
    _owned = False

    def __init__(self):
        self.active = False
        self.status = {}

    def __enter__(self):
        cls = type(self)
        with cls._lock:
            if self.active:
                raise RuntimeError('GC tracking scope already entered')
            if cls._users == 0:
                cls._owned = False
                if gc.isenabled() and gc.get_freeze_count() == 0:
                    # Do not force a full collection here: camera reception is
                    # already live, and a 60ms+ pause could starve its decoder.
                    # Any pre-existing unreachable cycles can wait until exit.
                    gc.freeze()
                    cls._owned = True
            cls._users += 1
            self.active = True
            self.status = dict(mode='freeze_startup_graph' if cls._owned else 'external_policy',
                               automatic_gc=gc.isenabled(), frozen_objects=gc.get_freeze_count())
        return self

    def __exit__(self, *args):
        cls = type(self)
        with cls._lock:
            if not self.active:
                return
            self.active = False
            cls._users -= 1
            if cls._users == 0 and cls._owned:
                gc.unfreeze()
                cls._owned = False


class GCPauseMonitor:
    def __init__(self):
        self.total_ms = 0.
        self._started = None
        self.generation_ms = [0., 0., 0.]
        self.collections = [0, 0, 0]

    def _callback(self, phase, info):
        if phase == "start":
            self._started = time.perf_counter_ns()
        elif self._started is not None:
            elapsed = (time.perf_counter_ns() - self._started) / 1e6
            self.total_ms += elapsed
            generation = int(info.get('generation', 0))
            self.generation_ms[generation] += elapsed
            self.collections[generation] += 1
            self._started = None

    def start(self):
        gc.callbacks.append(self._callback)

    def close(self):
        if self._callback in gc.callbacks:
            gc.callbacks.remove(self._callback)


class FrameTimingProbe:
    def __init__(self, gc_monitor):
        self.gc_monitor = gc_monitor
        self._previous_done = None
        self._previous_gc = gc_monitor.total_ms
        self._previous_generation_ms = list(gc_monitor.generation_ms)

    def sample(self, prepared, done_ns):
        wall = max(0., (done_ns - prepared["process_start_ns"]) / 1e6)
        cpu_start = prepared.get("thread_cpu_start_ns")
        cpu = (time.thread_time_ns() - cpu_start) / 1e6 if cpu_start is not None else None
        interval = (done_ns - self._previous_done) / 1e6 if self._previous_done is not None else None
        self._previous_done = done_ns
        collected = self.gc_monitor.total_ms
        gc_ms = max(0., collected - self._previous_gc)
        self._previous_gc = collected
        generations = self.gc_monitor.generation_ms[:]
        generation_pauses = [max(0., a-b) for a, b in zip(generations, self._previous_generation_ms)]
        self._previous_generation_ms = generations
        face = prepared.get("face_geometry", {})
        return dict(output_interval_ms=interval, inference_thread_cpu_ms=cpu,
                    inference_wait_ms_proxy=max(0., wall-cpu) if cpu is not None else None,
                    gc_pause_ms=gc_ms, gc_gen0_ms=generation_pauses[0], gc_gen1_ms=generation_pauses[1],
                    gc_gen2_ms=generation_pauses[2], face_geometry_age_ms=face.get("source_age_ms"),
                    face_detection_ms=face.get("detection_ms"), face_thread_cpu_ms=face.get("thread_cpu_ms"),
                    frame_sequence=prepared.get("sequence"))
