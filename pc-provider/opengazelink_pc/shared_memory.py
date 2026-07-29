from __future__ import annotations

import mmap
import struct
import threading
import time
from dataclasses import dataclass
from typing import Optional

from .types import GazeSample


SLOT_SIZE = 64
MAGIC = 0x315A4745
VERSION = 1
DEFAULT_MAP_NAME = "Local\\EyeTracingGazeV1"


@dataclass(frozen=True)
class SharedMemorySnapshot:
    magic: int
    version: int
    seq: int
    flags: int
    timestamp_ms: float
    x: float
    y: float
    width: float
    height: float
    confidence: float
    source_timestamp_ms: float


class GazeSharedMemoryWriter:
    def __init__(self, name: str = DEFAULT_MAP_NAME) -> None:
        self.name = name
        self._lock = threading.Lock()
        self._map: Optional[mmap.mmap] = None
        self._seq = 0

    def open(self) -> None:
        if self._map is not None:
            return
        self._map = mmap.mmap(-1, SLOT_SIZE, tagname=self.name, access=mmap.ACCESS_WRITE)
        self._write_header(valid=False)

    def close(self) -> None:
        if self._map is not None:
            self._map.close()
            self._map = None

    def write_invalid(self, width: float = 0.0, height: float = 0.0) -> None:
        self.write(
            GazeSample(
                t_ms=time.monotonic() * 1000.0,
                x=0.0,
                y=0.0,
                raw_x=0.0,
                raw_y=0.0,
                confidence=0.0,
                valid=False,
                status="LOST",
            ),
            width=width,
            height=height,
        )

    def write(self, gaze: GazeSample, width: float, height: float) -> None:
        if self._map is None:
            self.open()
        assert self._map is not None
        with self._lock:
            next_seq = self._seq + 1
            if next_seq % 2 == 0:
                next_seq += 1
            self._seq = next_seq
            buffer = bytearray(SLOT_SIZE)
            struct.pack_into("<IIII", buffer, 0, MAGIC, VERSION, next_seq, 1 if gaze.valid else 0)
            struct.pack_into("<d", buffer, 16, time.monotonic() * 1000.0)
            struct.pack_into("<ffff", buffer, 24, float(gaze.x), float(gaze.y), float(width), float(height))
            struct.pack_into("<f", buffer, 40, float(gaze.confidence))
            struct.pack_into("<f", buffer, 44, 0.0)
            struct.pack_into("<d", buffer, 48, float(gaze.t_ms))
            struct.pack_into("<Q", buffer, 56, 0)

            self._map.seek(0)
            self._map.write(buffer)
            self._map.seek(8)
            self._map.write(struct.pack("<I", next_seq + 1))
            self._map.flush()
            self._seq = next_seq + 1

    def _write_header(self, valid: bool) -> None:
        if self._map is None:
            return
        buffer = bytearray(SLOT_SIZE)
        struct.pack_into("<IIII", buffer, 0, MAGIC, VERSION, self._seq, 1 if valid else 0)
        self._map.seek(0)
        self._map.write(buffer)
        self._map.flush()


def read_shared_memory(name: str = DEFAULT_MAP_NAME) -> SharedMemorySnapshot:
    view = mmap.mmap(-1, SLOT_SIZE, tagname=name, access=mmap.ACCESS_READ)
    try:
        data = view[:SLOT_SIZE]
    finally:
        view.close()
    magic, version, seq, flags = struct.unpack_from("<IIII", data, 0)
    timestamp_ms = struct.unpack_from("<d", data, 16)[0]
    x, y, width, height = struct.unpack_from("<ffff", data, 24)
    confidence = struct.unpack_from("<f", data, 40)[0]
    source_timestamp_ms = struct.unpack_from("<d", data, 48)[0]
    return SharedMemorySnapshot(
        magic=magic,
        version=version,
        seq=seq,
        flags=flags,
        timestamp_ms=timestamp_ms,
        x=x,
        y=y,
        width=width,
        height=height,
        confidence=confidence,
        source_timestamp_ms=source_timestamp_ms,
    )
