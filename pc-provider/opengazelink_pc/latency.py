"""Rolling, JSON-safe latency statistics for the live gaze pipeline."""
from __future__ import annotations

from collections import deque
import math
import logging
import json
import threading
import time
from . import runtime_clock

import numpy as np


ADDITIVE_FIELDS = (
    "phone_capture_to_send_ms",
    "transport_to_first_packet_ms",
    "receive_decode_ms",
    "pc_queue_ms",
    "face_landmarks_ms",
    "eye_normalization_ms",
    "backend_overhead_ms",
    "gaze_model_ms",
    "projection_fusion_ms",
    "prediction_postprocess_ms",
    "shared_memory_write_ms",
    "other_processing_ms",
)

SUMMARY_FIELDS = ADDITIVE_FIELDS + (
    "output_interval_ms", "inference_thread_cpu_ms", "inference_wait_ms_proxy",
    "gc_pause_ms", "gc_gen0_ms", "gc_gen1_ms", "gc_gen2_ms",
    "face_geometry_age_ms", "face_detection_ms", "face_thread_cpu_ms",
    "transport_excess_ms_proxy",
    "clock_probe_rtt_ms",
    "clock_probe_uncertainty_ms",
    "clock_probe_age_ms",
    "packet_assembly_ms",
    "decode_queue_ms",
    "jpeg_decode_ms",
    "h264_reference_decode_ms", "h264_image_queue_ms", "h264_image_conversion_ms",
    "stage_sum_ms",
    "source_to_shared_memory_ms_proxy",
    "pc_processing_ms",
    "display_delay_ms_assumed",
    "source_to_display_ms_estimate",
    "prediction_horizon_ms",
)


def finite_milliseconds(value, maximum: float = 10_000.0) -> float | None:
    """Return a usable non-negative millisecond value, otherwise ``None``."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number) or number < 0.0 or number > maximum:
        return None
    return number


class PipelineLatencyTracker:
    """Collect on the tracking thread and aggregate on a low-rate worker.

    Adding a frame must stay bounded and cheap.  In particular, it must not
    calculate percentiles across the rolling window: on the production Python
    3.8 runtime that work took several milliseconds and delayed the next frame.
    """

    def __init__(
        self, window_seconds: float = 30.0, max_samples: int = 1800,
        aggregation_interval_s: float = 1.0, auto_start: bool = True,
    ) -> None:
        self.window_seconds = max(1.0, float(window_seconds))
        self.samples: deque[tuple[float, dict, tuple[float, ...]]] = deque(
            maxlen=max(1, int(max_samples)),
        )
        self._lock = threading.RLock()
        self._aggregation_interval_s = max(0.1, float(aggregation_interval_s))
        self._metrics = self._empty_metrics()
        self._aggregated_at_ms: float | None = None
        self._aggregated_sample_count = 0
        self._revision = 0
        self._last_log_ms = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        if auto_start:
            self._thread = threading.Thread(
                target=self._aggregation_loop,
                name="latency-aggregation",
                daemon=True,
            )
            self._thread.start()

    def reset(self) -> None:
        with self._lock:
            self.samples.clear()
            self._metrics = self._empty_metrics()
            self._aggregated_at_ms = None
            self._aggregated_sample_count = 0
            self._revision += 1
            self._last_log_ms = None

    def close(self) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=max(1.0, self._aggregation_interval_s * 2.0))
        self._thread = None

    def add(self, sample: dict, observed_at_ms: float | None = None) -> dict:
        now_ms = runtime_clock.monotonic() * 1000.0 if observed_at_ms is None else float(observed_at_ms)
        cleaned = dict(sample)
        numeric = []
        for field in SUMMARY_FIELDS:
            cleaned[field] = finite_milliseconds(cleaned.get(field))
            numeric.append(np.nan if cleaned[field] is None else cleaned[field])
        with self._lock:
            self.samples.append((now_ms, cleaned, tuple(numeric)))
            self._prune(now_ms)
            self._revision += 1
            return self._result(cleaned, now_ms)

    def summary(self, now_ms: float | None = None) -> dict:
        current_ms = runtime_clock.monotonic() * 1000.0 if now_ms is None else float(now_ms)
        with self._lock:
            if self._prune(current_ms):
                self._revision += 1
            current = dict(self.samples[-1][1]) if self.samples else {}
            return self._result(current, current_ms)

    def refresh(self, now_ms: float | None = None) -> dict:
        """Synchronously refresh aggregates; production calls this on the worker."""
        current_ms = runtime_clock.monotonic() * 1000.0 if now_ms is None else float(now_ms)
        with self._lock:
            if self._prune(current_ms):
                self._revision += 1
            rows = [sample for _, sample, _ in self.samples]
            numeric_rows = [numeric for _, _, numeric in self.samples]
            revision = self._revision
        current = dict(rows[-1]) if rows else {}
        matrix = np.asarray(numeric_rows, dtype=np.float64) if numeric_rows else None
        metrics = self._empty_metrics()
        if matrix is not None:
            for index, field in enumerate(SUMMARY_FIELDS):
                values = matrix[:, index]
                values = values[np.isfinite(values)]
                latest = current.get(field)
                metrics[field] = {
                    "current": latest if latest is not None else None,
                    "mean": float(np.mean(values)) if values.size else None,
                    "p95": float(np.percentile(values, 95.0)) if values.size else None,
                    "max": float(np.max(values)) if values.size else None,
                    "samples": int(values.size),
                }
        with self._lock:
            # A few new frames may have arrived while numpy worked. Publishing
            # this immutable snapshot is still useful; metadata exposes its age
            # and sample count, and the next current frame remains immediate.
            self._metrics = metrics
            self._aggregated_at_ms = current_ms
            self._aggregated_sample_count = len(rows)
            latest = dict(self.samples[-1][1]) if self.samples else {}
            result = self._result(latest, current_ms)
            result["aggregation"]["source_revision"] = revision
        self._log_diagnostics(current_ms, metrics, rows)
        return result

    def _log_diagnostics(self, now_ms, metrics, rows):
        # Only this background aggregation worker performs JSON encoding/disk
        # logging. Keep samples bounded and retain stages for spike attribution.
        if not rows or (self._last_log_ms is not None and now_ms-self._last_log_ms < 10_000):
            return
        since = self._last_log_ms if self._last_log_ms is not None else now_ms-self.window_seconds*1000
        with self._lock:
            recent = [(stamp, sample) for stamp, sample, _ in self.samples if stamp > since]
        self._last_log_ms = now_ms
        if not recent:
            return
        keys = ("pc_processing_ms", "source_to_shared_memory_ms_proxy", "output_interval_ms")
        thresholds = {key: max(25., (metrics[key]["mean"] or 0.) * 2.) for key in keys}
        spikes = [(stamp, sample) for stamp, sample in recent
                  if any((sample.get(key) or 0.) > thresholds[key] for key in keys)]
        worst = sorted(spikes, key=lambda pair: max(pair[1].get(k) or 0. for k in keys), reverse=True)[:8]
        payload = {"monotonic_ms": now_ms, "frames": len(recent), "spikes": len(spikes),
                   "thresholds_ms": thresholds, "metrics": metrics,
                   "worst_frames": [{"monotonic_ms": stamp, **sample} for stamp, sample in sorted(worst)],
                   "note": "wall minus thread CPU is a wait proxy, including native worker waits; not pure OS scheduling delay"}
        logging.getLogger("eyetracing.latency").info("latency_snapshot %s", json.dumps(payload, separators=(",", ":")))

    def _result(self, current: dict, now_ms: float) -> dict:
        aggregate_age = (
            max(0.0, now_ms - self._aggregated_at_ms)
            if self._aggregated_at_ms is not None else None
        )
        return {
            "schema": "opengazelink-live-latency-v1",
            "window_seconds": self.window_seconds,
            "sample_count": len(self.samples),
            "additive_fields": list(ADDITIVE_FIELDS),
            "current": current,
            "metrics": self._metrics,
            "aggregation": {
                "mode": "background",
                "interval_ms": self._aggregation_interval_s * 1000.0,
                "age_ms": aggregate_age,
                "sample_count": self._aggregated_sample_count,
            },
        }

    def _aggregation_loop(self) -> None:
        while not self._stop.wait(self._aggregation_interval_s):
            self.refresh()

    @staticmethod
    def _empty_metrics() -> dict:
        return {
            field: {"current": None, "mean": None, "p95": None, "max": None, "samples": 0}
            for field in SUMMARY_FIELDS
        }

    def _prune(self, now_ms: float) -> bool:
        cutoff = now_ms - self.window_seconds * 1000.0
        changed = False
        while self.samples and self.samples[0][0] < cutoff:
            self.samples.popleft()
            changed = True
        return changed
