"""Deferred MediaPipe VIDEO replay for raw camera archives."""
from __future__ import annotations

import json
import os
import time
import uuid
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from fractions import Fraction

import cv2
import numpy as np

from .camera import HEADER, MAGIC, FrameAssembly, FORMAT_H264, decode_udp_frame
from .h264_stream import AVC_HEADER, AvcDecoder
from .normalized_eye import NormalizedEyeBackend
from .video_session import audit_value, write_json


def processing_rate_limit(metadata: dict) -> float | None:
    """Return a replay limit only for the line-based spatial stage."""
    plan = metadata.get("plan") or []
    if not plan or not all(step.get("calibration_stage") == "spatial_v1" for step in plan):
        return None
    rates = {float(step["sample_rate_hz"]) for step in plan if step.get("sample_rate_hz") is not None}
    return rates.pop() if len(rates) == 1 else None


def _transform(image: np.ndarray, camera_model: dict) -> np.ndarray:
    rotation = int(camera_model.get("rotate", 0) or 0) % 360
    if rotation == 90:
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    elif rotation == 180:
        image = cv2.rotate(image, cv2.ROTATE_180)
    elif rotation == 270:
        image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if camera_model.get("mirror"):
        image = cv2.flip(image, 1)
    return image


def _read_payload(stream, record: dict) -> bytes:
    stream.seek(int(record["offset"]))
    payload = stream.read(int(record["length"]))
    if len(payload) != int(record["length"]):
        raise ValueError("raw camera archive is truncated")
    return payload


def _reconstruct_baseline_config(width: int, height: int, fps: int = 120) -> bytes:
    """Recover SPS/PPS for legacy archives that began after codec setup."""
    import av
    codec = av.CodecContext.create("libx264", "w")
    codec.width, codec.height = int(width), int(height)
    codec.pix_fmt = "yuv420p"
    codec.time_base = Fraction(1, fps)
    codec.framerate = Fraction(fps, 1)
    codec.bit_rate = width * height * fps
    codec.options = {"profile": "baseline", "preset": "ultrafast", "tune": "zerolatency",
                     "x264-params": f"keyint={fps}:min-keyint={fps}:scenecut=0:repeat-headers=1"}
    codec.open()
    frame = av.VideoFrame.from_ndarray(np.zeros((height, width, 3), np.uint8), format="bgr24")
    frame.pts = 0
    encoded = b"".join(bytes(packet) for packet in codec.encode(frame))
    boundary = encoded.find(b"\x00\x00\x01\x06")
    if boundary <= 0:
        raise ValueError("could not reconstruct H.264 SPS/PPS")
    return encoded[:boundary]


def iter_archived_frames(archive: Path, camera_model: dict, diagnostics=None, *, records=None, cancelled=lambda: False):
    """Yield every archived decoded frame, including frames skipped live."""
    archive = Path(archive)
    if records is None:
        records = [json.loads(line) for line in (archive / "index.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    assemblies: dict[tuple[int, int, int], FrameAssembly] = {}
    decoder = None
    diagnostics = diagnostics if diagnostics is not None else {}
    diagnostics.update(codec_config_records=0, codec_config_reconstructed=False,
                       skipped_before_keyframe=0, decode_errors=0)
    with (archive / "data.bin").open("rb") as stream:
        for record in records:
            if cancelled():
                raise RuntimeError("VIDEO replay cancelled; raw capture retained")
            kind = record.get("kind")
            payload = _read_payload(stream, record)
            if kind in ("windows_bgr", "processed_bgr"):
                image = np.frombuffer(payload, dtype=np.dtype(record["dtype"])).reshape(record["shape"]).copy()
                yield {"sequence": int(record.get("sequence", record.get("pc_capture_ms", 0))),
                       "source_ms": float(record.get("source_ms", record.get("pc_capture_ms", 0))),
                       "pc_read_ms": float(record.get("pc_capture_ms", record.get("pc_receive_ms", 0))),
                       "timing": {"source_capture_monotonic_ns": int(float(record.get("pc_capture_ms", 0)) * 1e6)},
                       "image": _transform(image, camera_model)}
                continue
            if kind == "h264_access_unit":
                if len(payload) < AVC_HEADER.size:
                    continue
                magic, seq, flags, width, height, sensor, encoded, sent, size = AVC_HEADER.unpack_from(payload)
                if magic != b"AVC1" or size != len(payload) - AVC_HEADER.size:
                    continue
                if decoder is None:
                    decoder = AvcDecoder()
                if flags & 2:
                    diagnostics["codec_config_records"] += 1
                if not decoder.config and not (flags & (1 | 2)):
                    diagnostics["skipped_before_keyframe"] += 1
                    continue
                if not decoder.config and flags & 1:
                    decoder.config = _reconstruct_baseline_config(width, height)
                    diagnostics["codec_config_reconstructed"] = True
                try:
                    images = decoder.decode(payload[AVC_HEADER.size:], flags, sensor)
                except Exception:
                    diagnostics["decode_errors"] += 1
                    decoder = AvcDecoder()
                    continue
                for image in images:
                    yield {"sequence": int(seq), "source_ms": float(sensor) / 1e6,
                           "pc_read_ms": float(record.get("pc_receive_ms", 0)),
                           "timing": {"phone_sensor_time_ns": int(sensor),
                                      "phone_send_time_ns": int(sent),
                                      "pc_first_packet_monotonic_ns": int(float(record.get("pc_receive_ms", 0)) * 1e6),
                                      "phone_encoded_time_ns": int(encoded)},
                           "image": _transform(image, camera_model)}
                continue
            if kind != "udp_packet" or len(payload) < HEADER.size:
                continue
            magic, version, header_size, seq, chunk, count, width, height, fmt, flags, sensor, send, size = HEADER.unpack_from(payload)
            if magic != MAGIC or version != 1 or header_size != HEADER.size or chunk >= count:
                continue
            key = (int(seq), int(sensor), int(send))
            item = assemblies.setdefault(key, FrameAssembly(seq, width, height, fmt, count, sensor, send,
                                                            float(record.get("pc_receive_ms", 0)) / 1000.0, {}))
            item.chunks[int(chunk)] = payload[header_size:header_size + size]
            if not item.complete:
                continue
            image = _transform(decode_udp_frame(item.payload(), width, height, fmt), camera_model)
            yield {"sequence": int(seq), "source_ms": float(sensor) / 1e6,
                   "pc_read_ms": float(record.get("pc_receive_ms", 0)),
                   "timing": {"phone_sensor_time_ns": int(sensor), "phone_send_time_ns": int(send),
                              "pc_first_packet_monotonic_ns": int(item.started_at * 1e9)},
                   "image": image}
            del assemblies[key]


def _prepare_replay_part(session_path, replay_dir, records, lower, upper, *, cancelled=lambda: False) -> dict:
    """Run Tasks Face Landmarker VIDEO after capture and write trainable inputs."""
    session_path = Path(session_path)
    metadata = json.loads((session_path / "session.json").read_text(encoding="utf-8"))
    camera_model = metadata.get("camera_model") or {}
    rate_limit_hz = processing_rate_limit(metadata)
    sample_interval_ms = 1000.0 / rate_limit_hz if rate_limit_hz else None
    sample_tolerance_ms = sample_interval_ms * .25 if sample_interval_ms else 0.0
    replay_dir.mkdir(parents=True)
    manifest = replay_dir / "frames.jsonl"
    backend = NormalizedEyeBackend("tasks", conditioned=True)
    backend.record_diagnostics = True
    frames = 0
    archived_frames = 0
    valid_frames = 0
    source_times = []
    archive_source_times = []
    last_archive_source_ms = None
    next_sample_ms = None
    archive_diagnostics = {}
    try:
        with manifest.open("w", encoding="utf-8", buffering=1) as output:
            for item in iter_archived_frames(session_path / "raw-camera", camera_model, archive_diagnostics,
                                             records=records, cancelled=cancelled):
                if cancelled():
                    raise RuntimeError("VIDEO replay cancelled; raw capture retained")
                # Earlier codec/keyframe records only warm up the H.264 decoder.
                # Each real frame belongs to exactly one half-open time range.
                if not lower <= item["pc_read_ms"] < upper:
                    continue
                source_ms = float(item["source_ms"])
                archived_frames += 1
                archive_source_times.append(source_ms)
                if last_archive_source_ms is not None and source_ms <= last_archive_source_ms:
                    backend.close()
                    backend = NormalizedEyeBackend("tasks", conditioned=True)
                    backend.record_diagnostics = True
                    next_sample_ms = None
                last_archive_source_ms = source_ms
                if sample_interval_ms is not None:
                    if next_sample_ms is None:
                        next_sample_ms = source_ms + sample_interval_ms
                    elif source_ms + sample_tolerance_ms < next_sample_ms:
                        continue
                    else:
                        while next_sample_ms <= source_ms + sample_tolerance_ms:
                            next_sample_ms += sample_interval_ms
                source_times.append(source_ms)
                record = {"index": frames, "sequence": item["sequence"], "source_ms": source_ms,
                          "pc_read_ms": item["pc_read_ms"], "timing": item["timing"], "valid": False}
                try:
                    observation = backend.predict(item["image"], source_ms, camera_model)
                except Exception as error:
                    observation = None
                    record["processing_error"] = str(error)
                diagnostics = dict(getattr(backend, "last_diagnostics", {}))
                landmark_faces = diagnostics.pop("landmarks", None)
                record["diagnostics"] = audit_value(diagnostics)
                record["rejection_reasons"] = ["no_conditioned_observation"]
                if observation is not None and observation.conditioned_inputs is not None:
                    arrays = {key: np.stack([value[key] for value in observation.conditioned_inputs])
                              for key in ("images", "head", "points", "crop", "rotation", "center")}
                    if landmark_faces:
                        arrays["mediapipe_landmarks"] = np.asarray(landmark_faces[0], dtype=np.float32)
                    aperture = min(observation.right.aperture_ratio, observation.left.aperture_ratio)
                    record.update(aperture=float(aperture), valid=bool(aperture >= .18))
                    record["aperture_per_eye"] = [float(observation.right.aperture_ratio), float(observation.left.aperture_ratio)]
                    record["rejection_reasons"] = [] if record["valid"] else ["aperture_below_0.18"]
                    path = replay_dir / f"{frames:07d}.npz"
                    temporary = path.with_suffix(".tmp")
                    with temporary.open("wb") as stream:
                        np.savez_compressed(stream, **arrays)
                    temporary.replace(path)
                    record["input"] = str(path.relative_to(session_path)).replace("\\", "/")
                    valid_frames += int(record["valid"])
                record["pc_done_ms"] = item["pc_read_ms"]
                output.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                frames += 1
    finally:
        backend.close()
    intervals = np.diff(np.asarray(source_times, dtype=np.float64)) if len(source_times) > 1 else np.asarray([])
    intervals = intervals[(intervals > 0.0) & (intervals < 250.0)]
    median_interval = float(np.median(intervals)) if len(intervals) else None
    processed_fps = 1000.0 / median_interval if median_interval else None
    archive_intervals = np.diff(np.asarray(archive_source_times, dtype=np.float64)) if len(archive_source_times) > 1 else np.asarray([])
    archive_intervals = archive_intervals[(archive_intervals > 0.0) & (archive_intervals < 250.0)]
    archive_median_interval = float(np.median(archive_intervals)) if len(archive_intervals) else None
    source_fps = 1000.0 / archive_median_interval if archive_median_interval else None
    result = {"frames": frames, "valid_frames": valid_frames, "manifest": str(manifest.resolve()),
              "mediapipe_running_mode": "VIDEO", "raw_archive": str((session_path / "raw-camera").resolve()),
              "archived_frames": archived_frames,
              "median_source_interval_ms": archive_median_interval, "source_fps_estimate": source_fps,
              "processed_median_interval_ms": median_interval, "processed_fps_estimate": processed_fps,
              "processing_rate_limit_hz": rate_limit_hz,
              "expected_source_fps": 120.0,
              "source_fps_warning": bool(source_fps is not None and source_fps < 100.0),
              "archive_decode": archive_diagnostics}
    result["_source_times"] = source_times
    result["_archive_source_times"] = archive_source_times
    return result


def replay_jobs(session_path, records):
    """Partition at capture attempts, with codec warm-up but no shared VIDEO state."""
    from bisect import bisect_left
    stimulus = session_path / "stimulus.jsonl"
    boundaries, previous = [], None
    if stimulus.exists():
        for line in stimulus.read_text(encoding="utf-8").splitlines():
            event = json.loads(line)
            segment = (event.get("capture_segment"), event.get("trial_id", event.get("block")))
            if segment != previous:
                boundaries.append(float(event["pc_ms"]))
                previous = segment
    times = [float(r.get("pc_capture_ms", r.get("pc_receive_ms", 0))) for r in records]
    if any(b < a for a, b in zip(times, times[1:])):
        # Legacy archives can contain an out-of-order cached codec record.
        # A single chronological replay remains safe for those archives.
        return [(records, -float("inf"), float("inf"))]
    # The first segment also owns pre-stimulus frames; the last owns the tail.
    cuts = sorted(set(boundaries[1:]))
    limits = [-float("inf"), *cuts, float("inf")]
    jobs = []
    with (session_path / "raw-camera" / "data.bin").open("rb") as stream:
        flags = {}
        for i, record in enumerate(records):
            if record.get("kind") == "h264_access_unit":
                stream.seek(int(record["offset"]))
                header = stream.read(AVC_HEADER.size)
                if len(header) == AVC_HEADER.size:
                    flags[i] = AVC_HEADER.unpack(header)[2]
    keyframes = [i for i, flag in flags.items() if flag & 1]
    configs = [i for i, flag in flags.items() if flag & 2]
    for lower, upper in zip(limits, limits[1:]):
        start, stop = bisect_left(times, lower), bisect_left(times, upper)
        if start >= stop:
            continue
        warm = start
        if flags:
            keys = [i for i in keyframes if i <= start]
            warm = keys[-1] if keys else 0
        elif start and records[start].get("kind") == "udp_packet":
            # Include the immediately preceding packet stream so the boundary
            # frame's first chunks are not lost. UDP assemblies expire at .25s.
            warm = bisect_left(times, times[start] - 250.)
        prefix = [i for i in configs if i < warm]
        selected = ([records[prefix[-1]]] if prefix else []) + records[warm:stop]
        jobs.append((selected, lower, upper))
    return jobs


def prepare_replay(session_path, *, progress=print, cancelled=lambda: False, workers=None) -> dict:
    """Parallel independent sequences, sequential frames within each sequence.

    Workers read compressed archive ranges themselves. No decoded video is
    queued in RAM, and the live receiver is never stopped/reconfigured.
    """
    session_path = Path(session_path)
    started = time.perf_counter()
    archive = session_path / "raw-camera"
    records = [json.loads(line) for line in (archive / "index.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    jobs = replay_jobs(session_path, records)
    if not jobs:
        raise RuntimeError("原始归档中没有可重放的视频帧")
    if workers is None:
        workers = int(os.environ.get("OPENGAZELINK_REPLAY_WORKERS", "0") or 0)
    workers = min(len(jobs), max(1, min(16, workers or min(8, max(1, (os.cpu_count() or 4) // 4)))))
    output = session_path / "replay-inputs" / uuid.uuid4().hex
    stop = threading.Event()
    parts = [None] * len(jobs)
    def aborted():
        return stop.is_set() or cancelled()
    progress({"stage": "replay", "completed": 0, "total": len(jobs), "workers": workers})
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="video-replay") as pool:
        futures = {pool.submit(_prepare_replay_part, session_path, output / str(i), *job,
                               cancelled=aborted): i for i, job in enumerate(jobs)}
        try:
            for done, future in enumerate(as_completed(futures), 1):
                parts[futures[future]] = future.result()
                progress({"stage": "replay", "completed": done, "total": len(jobs), "workers": workers})
        except BaseException:
            stop.set()
            for future in futures:
                future.cancel()
            raise
    if cancelled():
        raise RuntimeError("VIDEO replay cancelled; raw capture retained")
    manifest = session_path / "replay-frames.jsonl"
    temporary = output / "manifest.jsonl"
    count = 0
    with temporary.open("w", encoding="utf-8") as target:
        for part in parts:
            with Path(part["manifest"]).open(encoding="utf-8") as source:
                for line in source:
                    record = json.loads(line)
                    record["index"] = count
                    target.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                    count += 1
    if not count:
        raise RuntimeError("原始归档中没有可重放的视频帧")
    temporary.replace(manifest)
    result = {key: value for key, value in parts[0].items() if not key.startswith("_")}
    result.update(frames=count, valid_frames=sum(p["valid_frames"] for p in parts),
                  archived_frames=sum(p["archived_frames"] for p in parts), manifest=str(manifest.resolve()),
                  workers=workers, segments=len(parts), elapsed_s=time.perf_counter()-started,
                  parallel_policy="independent VIDEO state per capture segment; chronological merge",
                  archive_decode={"segments": [p["archive_decode"] for p in parts]})
    for field, median_key, fps_key in (("_source_times", "processed_median_interval_ms", "processed_fps_estimate"),
                                      ("_archive_source_times", "median_source_interval_ms", "source_fps_estimate")):
        intervals = np.concatenate([np.diff(p[field]) for p in parts])
        intervals = intervals[(intervals > 0) & (intervals < 250)]
        median = float(np.median(intervals)) if len(intervals) else None
        result[median_key], result[fps_key] = median, 1000. / median if median else None
    result["source_fps_warning"] = bool(result["source_fps_estimate"] is not None and result["source_fps_estimate"] < 100)
    write_json(session_path / "replay-report.json", result)
    return result
