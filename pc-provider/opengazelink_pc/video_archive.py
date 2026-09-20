"""Append-only camera archive, independent of inference and training filters."""
from __future__ import annotations

import json
import queue
import threading
import time
from . import runtime_clock


class RawVideoArchive:
    """Preserve received UDP packets / original OpenCV BGR bytes before decoding.

    RAM is bounded; overload fails the recording explicitly instead of silently
    dropping archival data. The camera itself remains usable after a disk error.
    Each flushed index entry refers to bytes already written to the data file.
    """
    def __init__(self, path, max_pending_bytes=128 * 1024 * 1024):
        self.path = path
        path.mkdir()
        self._queue = queue.Queue()
        self._lock = threading.Lock()
        self._pending = 0
        self._limit = max_pending_bytes
        self.error = ""
        self.records = 0
        self.bytes = 0
        self._closed = False
        self._worker = threading.Thread(target=self._write, name="video-raw-archive", daemon=True)
        self._worker.start()

    def submit(self, kind, metadata, payload):
        with self._lock:
            if self._closed or self.error:
                return
            if self._pending + len(payload) > self._limit:
                self.error = "原始数据写入跟不上采集，已停止本次采集；此前数据保留"
                return
            self._pending += len(payload)
            self._queue.put((kind, dict(metadata), bytes(payload)))

    def _write(self):
        try:
            with (self.path / "data.bin").open("wb") as data, (self.path / "index.jsonl").open("w", encoding="utf-8", buffering=1) as index:
                while True:
                    item = self._queue.get()
                    if item is None:
                        break
                    kind, metadata, payload = item
                    offset = data.tell()
                    data.write(payload)
                    data.flush()
                    index.write(json.dumps(dict(kind=kind, offset=offset, length=len(payload), **metadata), allow_nan=False) + "\n")
                    with self._lock:
                        self._pending -= len(payload)
                        self.records += 1
                        self.bytes += len(payload)
        except Exception as error:
            self.error = f"原始数据保存失败：{error}"
        finally:
            # Release any queued buffers after a disk failure.
            while not self._queue.empty():
                self._queue.get_nowait()

    def close(self):
        with self._lock:
            if not self._closed:
                self._closed = True
                self._queue.put(None)
        self._worker.join(timeout=10)
        if self._worker.is_alive():
            raise RuntimeError("原始数据仍在写入，请等待保存完成")
        (self.path / "archive.json").write_text(json.dumps({
            "schema": "opengazelink-raw-camera-v1", "records": self.records,
            "bytes": self.bytes, "error": self.error, "closed_pc_ms": runtime_clock.monotonic() * 1000,
            "policy": "original received packets or pre-transform BGR; no quality filtering or lossy recompression",
        }, ensure_ascii=False, indent=2), encoding="utf-8")
