from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import queue
import threading
from typing import Any

from .paths import MOTION_DIAGNOSTICS_DIR


_STOP = object()


class MotionDiagnosticsRecorder:
    """Write per-frame gaze diagnostics without blocking the inference thread."""

    def __init__(self, output_dir: Path = MOTION_DIAGNOSTICS_DIR) -> None:
        self.output_dir = Path(output_dir)
        self._lock = threading.Lock()
        self._queue: queue.Queue[Any] | None = None
        self._thread: threading.Thread | None = None
        self._active_path: Path | None = None
        self._latest_path: Path | None = self._find_latest_path()
        self._started_at = ""
        self._sample_count = 0
        self._dropped_records = 0

    def _find_latest_path(self) -> Path | None:
        if not self.output_dir.is_dir():
            return None
        paths = sorted(self.output_dir.glob("motion-*.jsonl"))
        return paths[-1] if paths else None

    def configure(self, enabled: bool, metadata: dict | None = None) -> None:
        if enabled:
            self.start(metadata or {})
        else:
            self.stop()

    def start(self, metadata: dict) -> Path:
        with self._lock:
            if self._active_path is not None:
                return self._active_path
            self.output_dir.mkdir(parents=True, exist_ok=True)
            now = datetime.now(timezone.utc)
            stamp = now.strftime("%Y%m%dT%H%M%S-%fZ")
            path = self.output_dir / f"motion-{stamp}.jsonl"
            records: queue.Queue[Any] = queue.Queue(maxsize=4096)
            thread = threading.Thread(
                target=self._write_loop,
                args=(path, records),
                name="motion-diagnostics-writer",
                daemon=True,
            )
            self._queue = records
            self._thread = thread
            self._active_path = path
            self._latest_path = path
            self._started_at = now.isoformat()
            self._sample_count = 0
            self._dropped_records = 0
            thread.start()
            records.put({
                "type": "session",
                "schema": "opengazelink-motion-v1",
                "started_at_utc": self._started_at,
                "metadata": metadata,
            })
            return path

    def record(self, payload: dict) -> None:
        with self._lock:
            records = self._queue
            active = self._active_path is not None
        if not active or records is None:
            return
        try:
            records.put_nowait(payload)
        except queue.Full:
            with self._lock:
                self._dropped_records += 1
        else:
            if payload.get("type") == "frame":
                with self._lock:
                    self._sample_count += 1

    def stop(self) -> Path | None:
        with self._lock:
            path = self._active_path
            records = self._queue
            thread = self._thread
            sample_count = self._sample_count
            dropped_records = self._dropped_records
            started_at = self._started_at
            self._active_path = None
            self._queue = None
            self._thread = None
        if path is None or records is None:
            return self._latest_path
        records.put({
            "type": "summary",
            "started_at_utc": started_at,
            "stopped_at_utc": datetime.now(timezone.utc).isoformat(),
            "sample_count": sample_count,
            "dropped_records": dropped_records,
        })
        records.put(_STOP)
        if thread is not None:
            thread.join(timeout=5.0)
        return path

    def status(self) -> dict:
        with self._lock:
            path = self._active_path or self._latest_path
            return {
                "active": self._active_path is not None,
                "path": str(path) if path is not None else "",
                "sample_count": self._sample_count,
                "dropped_records": self._dropped_records,
                "started_at_utc": self._started_at,
            }

    @staticmethod
    def _write_loop(path: Path, records: queue.Queue[Any]) -> None:
        with path.open("w", encoding="utf-8", buffering=1024 * 1024) as stream:
            while True:
                payload = records.get()
                if payload is _STOP:
                    stream.flush()
                    return
                stream.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
                stream.write("\n")
