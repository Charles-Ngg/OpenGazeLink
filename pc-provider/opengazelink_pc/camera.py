from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import socket
import struct
import threading
import time

import cv2
import numpy as np

from .paths import DATA_DIR


MAGIC = 0x56555945
INTRINSICS_MAGIC = 0x49435945
FORMAT_NV21 = 1
FORMAT_JPEG = 2
SUPPORTED_FRAME_FORMATS = {FORMAT_NV21, FORMAT_JPEG}
HEADER = struct.Struct("<IHHIHHHHBBQQI")
INTRINSICS_HEADER = struct.Struct("<IHHI")
DEFAULT_INTRINSICS_CACHE_PATH = DATA_DIR / "camera_intrinsics.json"


@dataclass
class UdpYuvConfig:
    bind: str = "0.0.0.0"
    port: int = 5007
    rotate: str | int = "auto"
    mirror: bool = False
    assembly_timeout_s: float = 0.15
    frame_stale_after_s: float = 0.75
    sequence_reset_after_s: float = 0.5
    intrinsics_cache_path: str = str(DEFAULT_INTRINSICS_CACHE_PATH)


@dataclass
class FrameAssembly:
    seq: int
    width: int
    height: int
    fmt: int
    chunk_count: int
    sensor_time_ns: int
    send_time_ns: int
    started_at: float
    chunks: dict[int, bytes]

    @property
    def complete(self) -> bool:
        return len(self.chunks) == self.chunk_count

    def payload(self) -> bytes:
        return b"".join(self.chunks[index] for index in range(self.chunk_count))


def decode_udp_frame(frame_bytes: bytes, width: int, height: int, fmt: int) -> np.ndarray:
    if fmt == FORMAT_NV21:
        expected = width * height * 3 // 2
        if len(frame_bytes) != expected:
            raise ValueError(f"NV21 byte count is {len(frame_bytes)}, expected {expected}")
        nv21 = np.frombuffer(frame_bytes, dtype=np.uint8).reshape((height * 3 // 2, width))
        return cv2.cvtColor(nv21, cv2.COLOR_YUV2BGR_NV21)
    if fmt == FORMAT_JPEG:
        frame = cv2.imdecode(np.frombuffer(frame_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("JPEG decode failed")
        if frame.shape[:2] != (height, width):
            raise ValueError(
                f"JPEG is {frame.shape[1]}x{frame.shape[0]}, packet says {width}x{height}"
            )
        return frame
    raise ValueError(f"unsupported frame format {fmt}")


class UdpYuvCamera:
    def __init__(self, config: UdpYuvConfig) -> None:
        self.config = config
        self._lock = threading.Lock()
        self._frame_condition = threading.Condition(self._lock)
        self._stop = threading.Event()
        self._latest_frame = None
        self._latest_t_ms = 0.0
        self._latest_seq = 0
        self._latest_frame_timings: dict[int, dict] = {}
        self._last_frame_at = 0.0
        self._last_error = ""
        self._source_camera_model: dict = {}
        self._packet_rotation = 270
        self._times: list[float] = []
        self._sensor_samples: list[tuple[int, int]] = []
        self._raw_width = 0
        self._raw_height = 0
        self._last_frame_seq: int | None = None
        self._last_format = FORMAT_NV21
        self._received_chunks = 0
        self._dropped_chunks = 0
        self._dropped_frames = 0
        self._sequence_resets = 0
        self._assemblies: dict[int, FrameAssembly] = {}
        self._pending_decode: FrameAssembly | None = None
        self._decode_drops = 0
        self._decode_ms = 0.0
        self._phone_pipeline_ms = 0.0
        self._transport_delta_min_ns: int | None = None
        self._transport_queue_ms = 0.0
        self._intrinsics_cache_error = ""
        self._allowed_source_ip: str | None = None
        self._last_source_ip = ""
        self._load_cached_intrinsics()
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 256 * 1024)
        self._receive_buffer_bytes = self._socket.getsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF)
        self._socket.bind((config.bind, config.port))
        self._socket.settimeout(0.05)
        self._thread = threading.Thread(target=self._receive_loop, name="udp-camera", daemon=True)
        self._decode_thread = threading.Thread(
            target=self._decode_loop, name="udp-camera-decode", daemon=True,
        )
        self._thread.start()
        self._decode_thread.start()

    def read_latest(self, after_seq: int = -1, timeout_s: float = 0.25):
        deadline = time.monotonic() + timeout_s
        with self._frame_condition:
            while True:
                now = time.monotonic()
                live = (
                    self._latest_frame is not None
                    and now - self._last_frame_at <= self.config.frame_stale_after_s
                )
                if live and self._latest_seq != after_seq:
                    return True, self._latest_frame.copy(), self._latest_t_ms, self._latest_seq
                remaining = deadline - now
                if self._last_error or self._stop.is_set() or remaining <= 0.0:
                    return False, None, 0.0, after_seq
                self._frame_condition.wait(remaining)

    def latest_frame_timing(self, sequence: int | None = None) -> dict:
        with self._lock:
            requested = self._latest_seq if sequence is None else sequence
            return dict(self._latest_frame_timings.get(requested) or {})

    def camera_model(self) -> dict:
        with self._lock:
            raw_width = self._raw_width
            raw_height = self._raw_height
            source = dict(self._source_camera_model)
            packet_rotation = self._packet_rotation
        if raw_width <= 0 or raw_height <= 0:
            return {}
        source_matches = (
            int(source.get("width", 0)) == raw_width
            and int(source.get("height", 0)) == raw_height
            and float(source.get("fx", 0.0)) > 0.0
            and float(source.get("fy", 0.0)) > 0.0
        )
        if source_matches:
            fx, fy = float(source["fx"]), float(source["fy"])
            cx, cy = float(source["cx"]), float(source["cy"])
            model_source = str(source.get("source") or "android_camera2")
            distortion = list(source.get("distortion") or [])
        else:
            fx = fy = float(max(raw_width, raw_height))
            cx, cy = (raw_width - 1) * 0.5, (raw_height - 1) * 0.5
            model_source = "estimated_frame_center"
            distortion = []
        width, height = raw_width, raw_height
        rotate = self._effective_rotation(packet_rotation)
        if rotate == 90:
            cx, cy = (height - 1) - cy, cx
            fx, fy = fy, fx
            width, height = height, width
        elif rotate == 180:
            cx, cy = (width - 1) - cx, (height - 1) - cy
        elif rotate == 270:
            cx, cy = cy, (width - 1) - cx
            fx, fy = fy, fx
            width, height = height, width
        if self.config.mirror:
            cx = (width - 1) - cx
        return {
            "rawWidth": raw_width, "rawHeight": raw_height,
            "width": width, "height": height,
            "fx": fx, "fy": fy, "cx": cx, "cy": cy,
            "rotate": rotate, "mirror": self.config.mirror,
            "source": model_source,
            "sourceOrigin": str(source.get("loadedFrom") or "estimated"),
            "distortion": distortion,
            "sourceMetadata": dict(source.get("metadata") or {}),
        }

    def _effective_rotation(self, packet_rotation: int | None = None) -> int:
        configured = str(self.config.rotate).lower()
        if configured != "auto":
            return int(configured) % 360
        rotation = self._packet_rotation if packet_rotation is None else packet_rotation
        return rotation if rotation in (0, 90, 180, 270) else 270

    def reported_mode(self) -> dict:
        now = time.monotonic()
        with self._lock:
            live = (
                self._latest_frame is not None
                and now - self._last_frame_at <= self.config.frame_stale_after_s
            )
            times = [value for value in self._times if value >= now - 2.0] if live else []
            sensor = list(self._sensor_samples)
            fps = 0.0
            if len(times) >= 2:
                fps = (len(times) - 1) / max(times[-1] - times[0], 1e-9)
            sensor_fps = 0.0
            if live and len(sensor) >= 2:
                sensor_fps = (sensor[-1][0] - sensor[0][0]) / max(
                    (sensor[-1][1] - sensor[0][1]) / 1_000_000_000.0, 1e-9,
                )
            return {
                "source": "udp_yuv",
                "width": self._raw_width if live else 0,
                "height": self._raw_height if live else 0,
                "fps": fps, "sensorFps": sensor_fps,
                "fourcc": "JPEG" if self._last_format == FORMAT_JPEG else "NV21",
                "backend": f"udp://{self.config.bind}:{self.config.port}",
                "chunks": self._received_chunks, "dropChunks": self._dropped_chunks,
                "dropFrames": self._dropped_frames, "buffered": len(self._assemblies),
                "decodeDrops": self._decode_drops,
                "decodeMs": self._decode_ms,
                "phonePipelineMs": self._phone_pipeline_ms,
                "transportQueueMs": self._transport_queue_ms,
                "receiveBufferBytes": self._receive_buffer_bytes,
                "sequenceResets": self._sequence_resets,
                "frameAgeMs": max(0.0, (now - self._last_frame_at) * 1000.0) if self._last_frame_at else None,
                "intrinsicsSource": self._source_camera_model.get("source", "estimated_frame_center"),
                "intrinsicsOrigin": self._source_camera_model.get("loadedFrom", "estimated"),
                "intrinsicsCacheError": self._intrinsics_cache_error,
                "rotation": self._effective_rotation(),
                "rotationSource": "phone" if str(self.config.rotate).lower() == "auto" else "manual",
                "error": self._last_error,
            }

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        rotate = self._effective_rotation()
        if rotate == 90:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        elif rotate == 180:
            frame = cv2.rotate(frame, cv2.ROTATE_180)
        elif rotate == 270:
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        if self.config.mirror:
            frame = cv2.flip(frame, 1)
        return frame

    def _receive_loop(self) -> None:
        while not self._stop.is_set():
            now = time.monotonic()
            self._drop_stale(now)
            try:
                packet, address = self._socket.recvfrom(65535)
            except socket.timeout:
                continue
            except OSError:
                break
            try:
                if self._allowed_source_ip and address[0] != self._allowed_source_ip:
                    continue
                self._last_source_ip = address[0]
                self._handle_packet(packet, now)
            except Exception as error:
                with self._frame_condition:
                    self._last_error = str(error)
                    self._frame_condition.notify_all()

    def set_allowed_source_ip(self, address: str | None) -> None:
        with self._lock:
            self._allowed_source_ip = str(address or "") or None

    def source_status(self) -> dict:
        with self._lock:
            return {
                "last_source_ip": self._last_source_ip,
                "allowed_source_ip": self._allowed_source_ip or "",
            }

    def _handle_packet(self, packet: bytes, now: float) -> None:
        if len(packet) >= INTRINSICS_HEADER.size:
            if struct.unpack_from("<I", packet)[0] == INTRINSICS_MAGIC:
                self._handle_intrinsics(packet)
                return
        if len(packet) < HEADER.size:
            return
        (
            magic, version, header_size, frame_seq, chunk_index, chunk_count,
            width, height, fmt, _flags, sensor_time_ns, send_time_ns, payload_size,
        ) = HEADER.unpack_from(packet)
        if magic != MAGIC or version != 1 or header_size != HEADER.size:
            return
        if fmt not in SUPPORTED_FRAME_FORMATS or chunk_index >= chunk_count or chunk_count < 1:
            return
        payload = packet[HEADER.size:HEADER.size + payload_size]
        if len(payload) != payload_size:
            return
        with self._frame_condition:
            if self._last_frame_seq is not None and frame_seq <= self._last_frame_seq:
                if now - self._last_frame_at < self.config.sequence_reset_after_s:
                    return
                self._last_frame_seq = None
                self._assemblies.clear()
                self._pending_decode = None
                self._times.clear()
                self._sensor_samples.clear()
                self._transport_delta_min_ns = None
                self._transport_queue_ms = 0.0
                self._sequence_resets += 1
            assembly = self._assemblies.get(frame_seq)
            if assembly is None:
                assembly = FrameAssembly(
                    frame_seq, width, height, fmt, chunk_count,
                    sensor_time_ns, send_time_ns, now, {},
                )
                self._assemblies[frame_seq] = assembly
                if send_time_ns:
                    delta_ns = time.monotonic_ns() - send_time_ns
                    if self._transport_delta_min_ns is None or delta_ns < self._transport_delta_min_ns:
                        self._transport_delta_min_ns = delta_ns
                    self._transport_queue_ms = max(
                        0.0, (delta_ns - self._transport_delta_min_ns) / 1_000_000.0,
                    )
                if sensor_time_ns and send_time_ns >= sensor_time_ns:
                    self._phone_pipeline_ms = (send_time_ns - sensor_time_ns) / 1_000_000.0
            if (
                assembly.width != width or assembly.height != height
                or assembly.fmt != fmt or assembly.chunk_count != chunk_count
            ):
                self._assemblies.pop(frame_seq, None)
                return
            if chunk_index not in assembly.chunks:
                assembly.chunks[chunk_index] = payload
                self._received_chunks += 1
            complete = assembly.complete
        if not complete:
            return
        with self._frame_condition:
            self._assemblies.pop(frame_seq, None)
            if self._last_frame_seq is not None and frame_seq > self._last_frame_seq + 1:
                self._dropped_frames += frame_seq - self._last_frame_seq - 1
            self._last_frame_seq = frame_seq
            if self._pending_decode is not None:
                self._decode_drops += 1
            self._pending_decode = assembly
            for old_seq in [value for value in self._assemblies if value < frame_seq]:
                old = self._assemblies.pop(old_seq)
                self._dropped_chunks += max(0, old.chunk_count - len(old.chunks))
            self._frame_condition.notify_all()

    def _decode_loop(self) -> None:
        while not self._stop.is_set():
            with self._frame_condition:
                while self._pending_decode is None and not self._stop.is_set():
                    self._frame_condition.wait(0.1)
                if self._stop.is_set():
                    return
                assembly = self._pending_decode
                self._pending_decode = None
            assert assembly is not None
            started = time.perf_counter()
            try:
                frame = self._preprocess(decode_udp_frame(
                    assembly.payload(), assembly.width, assembly.height, assembly.fmt,
                ))
            except Exception as error:
                with self._frame_condition:
                    self._last_error = str(error)
                    self._frame_condition.notify_all()
                continue
            decode_ms = (time.perf_counter() - started) * 1000.0
            monotonic_now = time.monotonic()
            with self._frame_condition:
                if self._pending_decode is not None and self._pending_decode.seq > assembly.seq:
                    self._decode_drops += 1
                    continue
                self._raw_width, self._raw_height = assembly.width, assembly.height
                self._last_format = assembly.fmt
                self._latest_frame = frame
                self._latest_t_ms = (
                    assembly.sensor_time_ns / 1_000_000.0
                    if assembly.sensor_time_ns else monotonic_now * 1000.0
                )
                self._latest_seq += 1
                self._last_frame_at = monotonic_now
                self._latest_frame_timings[self._latest_seq] = {
                    "read_sequence": self._latest_seq,
                    "phone_frame_sequence": assembly.seq,
                    "phone_sensor_time_ns": assembly.sensor_time_ns,
                    "phone_send_time_ns": assembly.send_time_ns,
                    "pc_first_packet_monotonic_ns": int(assembly.started_at * 1_000_000_000.0),
                    "pc_decode_done_monotonic_ns": int(monotonic_now * 1_000_000_000.0),
                    "decode_ms": decode_ms,
                }
                for old_sequence in [
                    value for value in self._latest_frame_timings
                    if value < self._latest_seq - 8
                ]:
                    self._latest_frame_timings.pop(old_sequence, None)
                self._last_error = ""
                self._decode_ms = decode_ms
                self._times.append(monotonic_now)
                self._times = [value for value in self._times if value >= monotonic_now - 2.0]
                if assembly.sensor_time_ns:
                    self._sensor_samples.append((assembly.seq, assembly.sensor_time_ns))
                    cutoff = assembly.sensor_time_ns - 2_000_000_000
                    self._sensor_samples = [
                        value for value in self._sensor_samples if value[1] >= cutoff
                    ]
                self._frame_condition.notify_all()

    def _drop_stale(self, now: float) -> None:
        with self._lock:
            stale = [
                seq for seq, value in self._assemblies.items()
                if now - value.started_at > self.config.assembly_timeout_s
            ]
            for seq in stale:
                value = self._assemblies.pop(seq)
                self._dropped_chunks += max(0, value.chunk_count - len(value.chunks))

    def _handle_intrinsics(self, packet: bytes) -> None:
        magic, version, header_size, payload_size = INTRINSICS_HEADER.unpack_from(packet)
        if magic != INTRINSICS_MAGIC or version != 1 or header_size != INTRINSICS_HEADER.size:
            return
        raw = packet[header_size:header_size + payload_size]
        if len(raw) != payload_size:
            return
        message = json.loads(raw.decode("utf-8"))
        stream = dict(message.get("streamIntrinsics") or {})
        if any(key not in stream for key in ("width", "height", "fx", "fy", "cx", "cy")):
            raise ValueError("Camera2 intrinsics packet is incomplete")
        model = {
            "width": int(stream["width"]), "height": int(stream["height"]),
            "fx": float(stream["fx"]), "fy": float(stream["fy"]),
            "cx": float(stream["cx"]), "cy": float(stream["cy"]),
            "source": str(message.get("source") or "android_camera2"),
            "distortion": list(message.get("distortion") or []),
            "metadata": message, "loadedFrom": "udp",
        }
        packet_rotation = int(message.get("frameRotation", 270))
        if packet_rotation not in (0, 90, 180, 270):
            packet_rotation = 270
        maximum = max(model["width"], model["height"])
        if not (
            0.1 * maximum <= model["fx"] <= 10.0 * maximum
            and 0.1 * maximum <= model["fy"] <= 10.0 * maximum
            and -model["width"] <= model["cx"] <= 2.0 * model["width"]
            and -model["height"] <= model["cy"] <= 2.0 * model["height"]
        ):
            raise ValueError("Camera2 intrinsics failed sanity checks")
        with self._lock:
            self._source_camera_model = model
            self._packet_rotation = packet_rotation
            self._intrinsics_cache_error = ""
            if self._raw_width <= 0 or self._raw_height <= 0:
                self._raw_width, self._raw_height = model["width"], model["height"]
        self._save_cached_intrinsics(model)

    def _cache_path(self) -> Path | None:
        return Path(self.config.intrinsics_cache_path) if self.config.intrinsics_cache_path else None

    def _load_cached_intrinsics(self) -> None:
        path = self._cache_path()
        if path is None or not path.exists():
            return
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            model = dict(payload.get("model") or payload)
            for key in ("width", "height", "fx", "fy", "cx", "cy", "source"):
                if key not in model:
                    raise ValueError(f"cached intrinsics miss {key}")
            model["loadedFrom"] = "cache"
            self._source_camera_model = model
            metadata = dict(model.get("metadata") or {})
            packet_rotation = int(metadata.get("frameRotation", 270))
            self._packet_rotation = packet_rotation if packet_rotation in (0, 90, 180, 270) else 270
        except Exception as error:
            self._intrinsics_cache_error = str(error)

    def _save_cached_intrinsics(self, model: dict) -> None:
        path = self._cache_path()
        if path is None:
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(path.suffix + ".tmp")
            temporary.write_text(json.dumps({
                "schema": "eyetracing-camera2-intrinsics-cache-v1",
                "savedAt": time.time(), "model": {**model, "loadedFrom": "cache"},
            }, ensure_ascii=False, indent=2), encoding="utf-8")
            temporary.replace(path)
        except Exception as error:
            self._intrinsics_cache_error = str(error)

    def release(self) -> None:
        self._stop.set()
        with self._frame_condition:
            self._frame_condition.notify_all()
        try:
            self._socket.close()
        except OSError:
            pass
        self._thread.join(timeout=1.0)
        self._decode_thread.join(timeout=1.0)
