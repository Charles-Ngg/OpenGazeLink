"""Continuous VIDEO capture. Stimulus telemetry is kept separate from eye inputs."""
from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import json
import math
import shutil
import hashlib
from pathlib import Path
import threading
import time
from . import runtime_clock

import numpy as np

from .paths import DATA_DIR
from .prediction_timing import PredictionClock
from .video_archive import RawVideoArchive

SCHEMA = "opengazelink-video-session-v1"
FRAME_STALL_TIMEOUT_S = 1.5


def audit_value(value):
    """Keep diagnostics JSON-readable, including rejected non-finite values."""
    if isinstance(value, np.ndarray):
        return audit_value(value.tolist())
    if isinstance(value, np.generic):
        return audit_value(value.item())
    if isinstance(value, dict):
        return {str(key): audit_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [audit_value(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return {"nonfinite": repr(value)}
    return value


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


class VideoSession:
    def __init__(self, camera, config, registry, *, root=None, backend=None, purpose="current_gaze", plan=None):
        if purpose not in ("current_gaze", "prediction", "unified"):
            raise ValueError("unknown capture purpose")
        self.purpose = purpose
        from .unified_capture import validate_plan
        self.plan = validate_plan(plan) if purpose == "unified" else []
        self.discarded_segments = set()
        self._resume_backend = False
        self.camera, self.config, self.registry = camera, config, registry
        self.path = (root or DATA_DIR / "video-sessions") / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S-%fZ")
        self.path.mkdir(parents=True)
        (self.path / "inputs").mkdir()
        # Keep the exact incoming camera stream for later replay/debugging.
        # Processed eye tensors remain alongside it for training convenience.
        self._raw_archive = None
        if getattr(camera, "supports_video_archive", True):
            self._raw_archive = RawVideoArchive(self.path / "raw-camera")
            camera.video_archive = self._raw_archive
            cached_config = getattr(camera, "_h264_codec_config_archive", None)
            if cached_config is not None:
                self._raw_archive.submit(*cached_config)
        if purpose == "prediction" or (self.plan and self.plan[0].get("calibration_stage") == "events_v1"):
            # Bind to the exact current estimator at capture time, including its geometry.
            source = DATA_DIR / "conditioned-video-model.json"
            meta = json.loads(source.read_text(encoding="utf-8"))
            item = meta["variants"]["conditioned_video"]
            if item.get("feature_dim") != 404:
                raise ValueError("预测校准需要已训练的 CNN VIDEO 模型")
            if (meta["screen"] != {"width":config.screen_width,"height":config.screen_height}
                or meta["screen_diagonal_inches"] != config.screen_diagonal_inches
                or meta["camera_position_screen_cm"] != list(config.camera_position_screen_cm)):
                raise ValueError("预测校准的屏幕与相机位置需要与当前 VIDEO 模型一致")
            module = source.with_name(item["module_file"])
            if hashlib.sha256(module.read_bytes()).hexdigest() != item["module_sha256"]:
                raise ValueError("VIDEO model checksum mismatch")
            (self.path / "base-model").mkdir()
            shutil.copy2(source, self.path / "base-model" / source.name)
            shutil.copy2(module, self.path / "base-model" / module.name)
        # MediaPipe is deferred until the post-capture VIDEO replay.  Keeping
        # the capture thread raw-only is required to retain the 120 FPS source.
        self._backend = backend
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self.state, self.error, self.phase = "collecting", "", "prediction_capture" if purpose == "prediction" else "unified_capture" if purpose == "unified" else "video"
        self.frames, self.valid_frames, self.events = 0, 0, 0
        self.training_result = None
        self.training_progress = {}
        self._last_event_ms = -math.inf
        self._heartbeat = runtime_clock.monotonic()
        self._accepted_batches = set()
        self.metadata = {"schema": SCHEMA, "config": asdict(config), "state": self.state,
                         "clock": "PC monotonic milliseconds; browser midpoint synchronization",
                         "label_policy": "stimulus positions are weak labels, never measured gaze",
                         "mediapipe_running_mode": "deferred_VIDEO", "created_at": datetime.now(timezone.utc).isoformat(),
                         "capture_revision": 6, "purpose": purpose, "plan": self.plan, "discarded_segments": [], "camera_model": camera.camera_model(),
                         "capture_mode": "raw_only",
                         "stimulus_schema": "opengazelink-stimulus-v3",
                         "timing_policy": "display times are requestAnimationFrame submission estimates, not measured photon times; phone offset is a minimum-transit proxy",
                         "retention": "exact full-frame camera archive plus lightweight frame timing and validated stimulus labels; MediaPipe VIDEO tensors are generated during replay"}
        write_json(self.path / "session.json", self.metadata)
        self._thread = threading.Thread(target=self._capture, name="video-capture", daemon=True)
        self._thread.start()

    def status(self):
        with self._lock:
            return {"active": self.state in ("collecting", "paused", "stopping", "training"), "state": self.state,
                    "phase": self.phase, "purpose": self.purpose, "frames": self.frames, "valid_frames": self.valid_frames,
                    "calibration_stage": self.plan[0].get("calibration_stage", "unified") if self.plan else None,
                    "events": self.events, "error": self.error, "directory": str(self.path),
                    "progress": dict(getattr(self, "training_progress", {})),
                    "evaluation": self._evaluation_summary()}

    def _evaluation_summary(self):
        result = self.training_result or {}
        if result.get("mode") == "event_evaluation":
            test = result.get("by_split", {}).get("test", {})
            return {"kind": "events", "test": test,
                    "stability_calibration": result.get("stability_calibration"),
                    "stability_published": result.get("stability_published", False),
                    "report": str(Path(result["training_directory"]) / "report.json")}
        spatial = result.get("spatial", {})
        test = spatial.get("independent_test", {})
        return {"kind": "spatial", "test": test.get("candidate"),
                "validation": spatial.get("candidate"),
                "published": result.get("published", False)} if spatial else None

    def add_events(self, body):
        with self._lock:
            attempt = {"received_pc_ms": runtime_clock.monotonic() * 1000}
            batch = str(body.get("batch_id", ""))
            try:
                if batch and batch in self._accepted_batches:
                    result = self.status()
                    decision = "duplicate_already_saved"
                else:
                    result = self._add_validated_events(body)
                    if batch:
                        self._accepted_batches.add(batch)
                    decision = "accepted"
            except Exception as error:
                decision = f"rejected: {error}"
                raise
            finally:
                # Retain one compact audit row, not a second copy of every event
                # plus the complete plan on every browser flush.
                with (self.path / "event-batches.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps({"received_pc_ms": attempt["received_pc_ms"], "batch_id": batch,
                                             "event_count": len(body.get("events", [])) if isinstance(body.get("events"), list) else None,
                                             "decision": decision}, ensure_ascii=False) + "\n")
            return result

    def _add_validated_events(self, body):
        events = body.get("events", [])
        if not isinstance(events, list) or not 1 <= len(events) <= 240:
            raise ValueError("expected 1–240 stimulus events")
        with self._lock:
            if self.state != "collecting":
                raise RuntimeError("VIDEO capture is not active")
            previous = self._last_event_ms
            validated = []
            for event in events:
                numeric = {key: float(event[key]) for key in ("pc_ms", "browser_ms", "x", "y", "sync_rtt_ms")}
                if not all(math.isfinite(value) for value in numeric.values()):
                    raise ValueError("non-finite stimulus telemetry")
                item = dict(event, **numeric)
                if item["pc_ms"] <= previous or abs(item["pc_ms"] - runtime_clock.monotonic() * 1000) > 15000:
                    raise ValueError("stimulus clock is out of order or stale")
                if not (0 <= item["x"] <= 1 and 0 <= item["y"] <= 1 and 0 <= item["sync_rtt_ms"] <= 200):
                    raise ValueError("invalid stimulus position or clock uncertainty")
                phase = str(event["phase"])
                if phase not in ("anchor", "pursuit", "jump", "pause"):
                    raise ValueError("unknown VIDEO phase")
                item.update(phase=phase, block=str(event["block"])[:80], visible=bool(event.get("visible", True)))
                if self.purpose in ("prediction", "unified"):
                    split = event.get("split")
                    if split not in ("train", "validation", "test") or not str(event.get("trial_id", "")):
                        raise ValueError("prediction stimulus requires trial_id and split")
                    if not item["block"].startswith(split + "-") or item["block"] != event["trial_id"]:
                        raise ValueError("trial identity and split do not match")
                    if not math.isfinite(float(event.get("trial_age_ms", -1))) or float(event.get("trial_age_ms", -1)) < 0:
                        raise ValueError("invalid trial age")
                if self.purpose == "unified":
                    if event["trial_id"] not in {s["trial_id"] for s in self.plan}:
                        raise ValueError("unknown unified trial")
                    segment = event.get("capture_segment")
                    if not isinstance(segment, int) or segment < 1:
                        raise ValueError("invalid capture segment")
                previous = item["pc_ms"]
                validated.append(item)
            with (self.path / "stimulus.jsonl").open("a", encoding="utf-8") as stream:
                for item in validated:
                    stream.write(json.dumps(item, allow_nan=False) + "\n")
            self._last_event_ms = previous
            self.events += len(validated)
            self._heartbeat = runtime_clock.monotonic()
        return self.status()

    def _capture(self):
        sequence, last_ms = -1, None
        last_frame_at = runtime_clock.monotonic()
        clock=PredictionClock()
        try:
            with (self.path / "frames.jsonl").open("a", encoding="utf-8", buffering=1) as stream:
                while not self._stop.is_set():
                    if self._raw_archive is not None and self._raw_archive.error:
                        raise RuntimeError(self._raw_archive.error)
                    if self.state == "paused":
                        self._stop.wait(.1)
                        continue
                    if self._resume_backend:
                        self._resume_backend = False
                        last_ms = None
                        last_frame_at = runtime_clock.monotonic()
                        clock.reset()
                    if runtime_clock.monotonic() - self._heartbeat > 15:
                        raise RuntimeError("VIDEO page disconnected; capture stopped, partial data retained")
                    ok, frame, source_ms, next_sequence = self.camera.read_latest(sequence, timeout_s=.1)
                    if not ok or frame is None:
                        if runtime_clock.monotonic() - last_frame_at > FRAME_STALL_TIMEOUT_S:
                            camera_status = self.camera.reported_mode() if hasattr(self.camera, "reported_mode") else {}
                            detail = str(camera_status.get("error") or f"超过 {FRAME_STALL_TIMEOUT_S:g} 秒没有收到新画面")
                            raise RuntimeError(f"采集画面流已中断：{detail}")
                        continue
                    last_frame_at = runtime_clock.monotonic()
                    sequence = next_sequence
                    record = {"index": self.frames, "sequence": sequence, "source_ms": float(source_ms),
                              "pc_read_ms": runtime_clock.monotonic() * 1000,
                              "timing": self.camera.latest_frame_timing(sequence), "valid": True,
                              "input": "raw-camera", "capture_only": True}
                    last_ms = source_ms
                    # Do not invoke MediaPipe or write derived tensors here.
                    # RawVideoArchive receives the complete source stream in
                    # parallel, including frames skipped by read_latest().
                    self.valid_frames += 1
                    record["pc_done_ms"] = runtime_clock.monotonic() * 1000
                    # Compact numeric timing is retained because prediction
                    # training/auditing needs the effective forecast horizon.
                    record["prediction_timing"]=clock.observe(source_ms,record["timing"],record["pc_done_ms"],self.config.prediction_display_delay_ms)
                    with self._lock:
                        stream.write(json.dumps(record, allow_nan=False) + "\n")
                        self.frames += 1
        except Exception as error:
            self.error, self.state = str(error), "failed"
        finally:
            if self._backend is not None:
                self._backend.close()
            if self._raw_archive is not None:
                self._raw_archive.close()
                if getattr(self.camera, "video_archive", None) is self._raw_archive:
                    self.camera.video_archive = None
            self._persist()

    def _persist(self):
        self.metadata.update(self.status())
        self.metadata["discarded_segments"] = sorted(self.discarded_segments)
        write_json(self.path / "session.json", self.metadata)

    def pause(self, discard_segment=None):
        with self._lock:
            if self.state not in ("collecting", "paused"):
                raise RuntimeError("当前采集无法暂停")
            if discard_segment is not None:
                if not isinstance(discard_segment, int) or discard_segment < 1:
                    raise ValueError("invalid discarded segment")
                self.discarded_segments.add(discard_segment)
            self.state = "paused"
            self._persist()
            return self.status()

    def resume(self):
        with self._lock:
            if self.state != "paused":
                raise RuntimeError("当前采集并未暂停")
            self._heartbeat = runtime_clock.monotonic()
            self._resume_backend = True
            self.state = "collecting"
            return self.status()

    def review(self):
        from .video_dataset import align_frames, read_jsonl
        from .unified_capture import coverage_report
        with self._lock:
            if self.state != "paused" or not self.plan:
                raise RuntimeError("请先暂停统一采集，再检查覆盖情况")
            frames = read_jsonl(self.path / "frames.jsonl")
            events = read_jsonl(self.path / "stimulus.jsonl") if (self.path / "stimulus.jsonl").exists() else []
            rows = align_frames(frames, events, discarded_segments=self.discarded_segments) if frames and len(events)>1 else []
            report = coverage_report(rows, self.plan, events=events)
            write_json(self.path / "coverage-report.json", report)
            return report

    def _stop_capture(self):
        self._stop.set()
        self._thread.join(timeout=10)
        if self._thread.is_alive():
            raise RuntimeError("VIDEO capture is still stopping")

    def finish(self):
        with self._lock:
            if self.purpose == "unified":
                report = self.review()
                if not report["ready"]:
                    raise ValueError(f"仍有 {len(report['missing_trials'])} 个小段需要补采")
                from .video_dataset import read_jsonl
                observed = {e["capture_segment"] for e in read_jsonl(self.path / "stimulus.jsonl")}
                self.discarded_segments.update(observed - set(report["accepted_segments"]))
                self._persist()
            if self.state not in ("collecting", "paused"):
                raise RuntimeError("VIDEO session cannot be finished in its current state")
            self.state = "stopping"
        self._stop_capture()
        if self.error:
            raise RuntimeError(self.error)
        self.state, self.phase = "training", "prediction_training" if self.purpose == "prediction" else "unified_training"
        self._persist()
        self._training_thread = threading.Thread(target=self._train, name="video-training", daemon=True)
        self._training_thread.start()
        return self.status()

    def _train(self):
        try:
            from .video_replay import prepare_replay
            self.phase = "unified_replay" if self.purpose == "unified" else "video_replay"
            replay = prepare_replay(self.path, progress=self._training_progress, cancelled=self._stop_training)
            self.metadata["replay"] = replay
            self.frames = int(replay["frames"])
            self.valid_frames = int(replay["valid_frames"])
            capture_manifest = self.path / "capture-frames.jsonl"
            frames_manifest = self.path / "frames.jsonl"
            if frames_manifest.exists() and not capture_manifest.exists():
                frames_manifest.replace(capture_manifest)
            shutil.copy2(replay["manifest"], frames_manifest)
            self.metadata["capture_manifest"] = capture_manifest.name
            self._persist()
            if getattr(self, "plan", None) and self.plan[0].get("calibration_stage") == "events_v1":
                from .event_evaluation import evaluate_session
                self.training_result = evaluate_session(self.path, progress=self._training_progress,
                                                        cancelled=self._stop_training)
                with self._lock:
                    if self.state != "cancelled":
                        from .stability_profile import publish_profile
                        self.training_result["stability_published"] = publish_profile(
                            self.training_result, self.config, self.metadata["camera_model"])
                        write_json(Path(self.training_result["training_directory"]) / "report.json",
                                   self.training_result)
                        self.state, self.phase = "complete", "event_evaluation_complete"
                return
            if self.purpose == "prediction":
                from .unified_prediction_training import train_unified
                self.training_result = train_unified(
                    self.path, self.path / "base-model" / "conditioned-video-model.json",
                    publish_model_path=DATA_DIR / "conditioned-video-model.json",
                    progress=self._training_progress, cancelled=self._stop_training)
                self.registry.clear()
                if self.state != "cancelled":
                    self.state = "complete"
                    self.phase = "prediction_complete" if self.training_result["published"] else "prediction_not_accepted"
                return
            from .unified_calibration_training import train_calibration
            self.training_result = train_calibration(self.path, progress=self._training_progress,
                                                     cancelled=self._stop_training)
            self.registry.clear()
            if self.state != "cancelled":
                self.state = "complete"
                self.phase = "unified_complete" if self.training_result["published"] else "unified_kept_existing"
        except Exception as error:
            if self.state != "cancelled":
                self.error, self.state = str(error), "failed"
        finally:
            self._persist()

    def _stop_training(self):
        return self.state == "cancelled"

    def _training_progress(self, phase):
        with self._lock:
            if isinstance(phase, dict):
                self.training_progress = dict(phase)
                self.phase = "unified_replay" if phase.get("stage") == "replay" else "unified_spatial"
            elif phase.startswith("personal_epoch:"):
                completed, total = map(int, phase.split(":", 1)[1].split("/"))
                self.training_progress = {"stage": "spatial", "completed": completed, "total": total}
                self.phase = "unified_spatial"
            else:
                self.phase = phase
                self.training_progress = {}

    def cancel(self):
        self.state = "cancelled"
        self._stop_capture()
        worker = getattr(self, "_training_thread", None)
        if worker is not None:
            worker.join(timeout=10)
        self._persist()
