from __future__ import annotations

from dataclasses import asdict
import copy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
from pathlib import Path
import socket
import threading
import time
from . import runtime_clock
import webbrowser

from .calibration_session import (
    BUILTIN_LIGHTING_PROFILES, CALIBRATION_SCHEMA,
    CalibrationSession, LightingAdaptationSession, delete_lighting_profile,
    calibration_targets, light_anchor_targets, pose_targets,
    lighting_profile_library_status,
    retrain_lighting_profiles,
)
from .config import DEFAULT_CONFIG_PATH, ProviderConfig, save_config
from .camera import enumerate_windows_cameras
from .engine import EyeTrackingEngine
from .model_registry import ModelRegistry
from .normalized_eye import screen_camera_origin
from .paths import WEB_DIR
from .video_session import VideoSession


def _geometry_matches_dataset_status(dataset: dict, config: ProviderConfig) -> bool:
    if dataset.get("input_source", "phone_udp") != config.input_source:
        return False
    if config.input_source == "windows_camera":
        camera = dataset.get("windows_camera") or {}
        if (
            camera.get("device_index") != config.windows_camera_index
            or camera.get("width") != config.windows_camera_width
            or camera.get("height") != config.windows_camera_height
            or camera.get("rotate") != config.rotate
            or camera.get("mirror") != config.mirror
            or camera.get("fov_x_degrees") is None
            or abs(float(camera["fov_x_degrees"]) - config.windows_camera_fov_x_degrees) > 0.01
        ):
            return False
    if dataset.get("screen") != {
        "width": config.screen_width, "height": config.screen_height,
    }:
        return False
    if abs(float(dataset.get("screen_diagonal_inches") or 0.0) - config.screen_diagonal_inches) > 0.01:
        return False
    origin = dataset.get("screen_camera_origin_cm")
    expected = screen_camera_origin(
        config.screen_width, config.screen_height,
        config.screen_diagonal_inches, config.camera_position_screen_cm,
    )
    return isinstance(origin, list) and len(origin) == 3 and all(
        abs(float(actual) - float(wanted)) <= 0.05
        for actual, wanted in zip(origin, expected)
    )


def _lighting_profile_status(artifacts: dict, config: ProviderConfig) -> list[dict]:
    models = artifacts.get("models") or {}
    datasets = artifacts.get("datasets") or {}
    library = lighting_profile_library_status(config)
    names = {"reference"}
    for key in ("legacy_cnn", "tasks_cnn"):
        names.update((models.get(key) or {}).get("lighting_profiles") or [])
    for backend in ("legacy", "tasks"):
        names.update(
            ((datasets.get(backend) or {}).get("lighting_profile_samples") or {}).keys()
        )
    names.update(library)
    profiles = []
    for name in sorted(names):
        sample_counts = {
            backend: int(
                (datasets.get(backend) or {}).get("lighting_profile_samples", {}).get(name, 0)
            )
            for backend in ("legacy", "tasks")
        }
        if name in library:
            sample_counts = dict(library[name]["sample_frames"])
        model_present = any(
            name in ((models.get(key) or {}).get("lighting_profiles") or [])
            for key in ("legacy_cnn", "tasks_cnn")
        )
        geometry_matches = (
            bool(library[name]["geometry_matches"])
            if name in library
            else all(
                _geometry_matches_dataset_status(datasets.get(backend) or {}, config)
                for backend in ("legacy", "tasks")
            )
        )
        reusable = bool(library.get(name, {}).get("reusable", False))
        profiles.append({
            "name": name,
            "builtin": name in BUILTIN_LIGHTING_PROFILES,
            "deletable": name != "reference",
            "reusable": reusable,
            "model_present": model_present,
            "sample_frames": sample_counts,
            "geometry_matches": geometry_matches,
        })
    return profiles


class ControlApplication:
    def __init__(
        self,
        config: ProviderConfig,
        config_path: Path = DEFAULT_CONFIG_PATH,
        registry: ModelRegistry | None = None,
        engine: EyeTrackingEngine | None = None,
        host_status=None,
        enter_background=None,
        accept_pairing=None,
        forget_pairing=None,
        config_updated=None,
        shutdown_application=None,
    ) -> None:
        self.config = config
        self.config_path = config_path
        self.registry = registry or ModelRegistry()
        self.engine = engine or EyeTrackingEngine(config, self.registry)
        self._host_status = host_status
        self._enter_background = enter_background
        self._accept_pairing = accept_pairing
        self._forget_pairing = forget_pairing
        self._config_updated = config_updated
        self._shutdown_application = shutdown_application
        self.calibration: CalibrationSession | None = None
        self._lock = threading.RLock()

    def status(self) -> dict:
        calibration = self.calibration.status() if self.calibration is not None else {
            "active": False, "state": "idle", "phase": "idle",
            "index": 0,
            "total": len(calibration_targets(self.config.screen_width, self.config.screen_height)),
            "light_index": 0,
            "light_total": len(light_anchor_targets(self.config.screen_width, self.config.screen_height)),
            "pose_index": 0,
            "pose_total": len(pose_targets(self.config.screen_width, self.config.screen_height)),
        }
        artifacts = copy.deepcopy(self.registry.status())
        expected_origin = screen_camera_origin(
            self.config.screen_width,
            self.config.screen_height,
            self.config.screen_diagonal_inches,
            self.config.camera_position_screen_cm,
        ).tolist()
        for item in artifacts.get("models", {}).values():
            actual_origin = item.get("screen_camera_origin_cm")
            origin_matches = (
                isinstance(actual_origin, list)
                and len(actual_origin) == 3
                and all(
                    abs(float(a) - float(b)) <= 0.05
                    for a, b in zip(actual_origin, expected_origin)
                )
            )
            screen_matches = item.get("screen") == {
                "width": self.config.screen_width,
                "height": self.config.screen_height,
            }
            diagonal_matches = abs(
                float(item.get("screen_diagonal_inches") or 0.0)
                - self.config.screen_diagonal_inches
            ) <= 0.01
            geometry_compatible = bool(
                self.config.geometry_configured
                and origin_matches and screen_matches and diagonal_matches
            )
            input_source_compatible = item.get("input_source", "phone_udp") == self.config.input_source
            if input_source_compatible and self.config.input_source == "windows_camera":
                calibrated_camera = item.get("windows_camera") or {}
                current_camera = {
                    "device_index": self.config.windows_camera_index,
                    "width": self.config.windows_camera_width,
                    "height": self.config.windows_camera_height,
                    "fov_x_degrees": self.config.windows_camera_fov_x_degrees,
                    "rotate": self.config.rotate,
                    "mirror": self.config.mirror,
                }
                input_source_compatible = all(
                    abs(float(calibrated_camera.get(key)) - value) <= 0.01
                    if isinstance(value, float) and calibrated_camera.get(key) is not None
                    else calibrated_camera.get(key) == value
                    for key, value in current_camera.items()
                )
            item["geometry_compatible"] = geometry_compatible
            item["input_source_compatible"] = input_source_compatible
            if item.get("ready"):
                item["compatible"] = bool(
                    item.get("compatible", True)
                    and geometry_compatible and input_source_compatible
                )
        result = {
            "calibration_schema": CALIBRATION_SCHEMA,
            "config": asdict(self.config),
            "geometry": {
                "configured": self.config.geometry_configured,
                "screen_resolution": [self.config.screen_width, self.config.screen_height],
                "screen_diagonal_inches": self.config.screen_diagonal_inches,
                "camera_position_screen_cm": list(self.config.camera_position_screen_cm),
            },
            "engine": self.engine.status(),
            "artifacts": artifacts,
            "lighting_profiles": _lighting_profile_status(artifacts, self.config),
            "calibration": calibration,
        }
        from .stability_profile import read_profile
        profile = read_profile(self.config, self.engine.camera.camera_model())
        result["stability"] = {"automatic": True, "calibrated": profile is not None,
                               "created_at": profile.get("created_at") if profile else None}
        if self._host_status is not None:
            result["application"] = self._host_status()
        return result

    def windows_cameras(self) -> list[dict]:
        cameras = enumerate_windows_cameras()
        mode = self.engine.camera.reported_mode()
        if mode.get("source") == "windows_camera":
            current_index = int(mode.get("deviceIndex", self.config.windows_camera_index))
            if not any(int(item.get("index", -1)) == current_index for item in cameras):
                cameras.insert(0, {
                    "index": current_index,
                    "name": f"Windows camera {current_index} (active)",
                    "width": int(mode.get("width") or 0),
                    "height": int(mode.get("height") or 0),
                    "fps": float(mode.get("fps") or 0.0),
                    "backend": str(mode.get("backend") or self.config.windows_camera_backend),
                })
        return cameras

    def update_config(self, values: dict) -> dict:
        with self._lock:
            if self.calibration is not None and self.calibration.status()["active"]:
                raise RuntimeError("cannot change configuration during calibration")
            updated = ProviderConfig(**asdict(self.config))
            updated.update(values)
            save_config(updated, self.config_path)
            self.engine.reconfigure(updated)
            self.config = updated
            if self._config_updated is not None:
                self._config_updated(updated)
            return asdict(updated)

    def start_tracking(self) -> None:
        if self.calibration is not None and self.calibration.status()["active"]:
            raise RuntimeError("calibration is active")
        self.config.require_geometry()
        input_status = self.engine.input_status()
        if not input_status["ready"]:
            raise RuntimeError(input_status["error"])
        self.engine.start_tracking()

    def start_calibration(self) -> dict:
        with self._lock:
            if self.calibration is not None and self.calibration.status()["active"]:
                raise RuntimeError("calibration is already active")
            self.config.require_geometry()
            input_status = self.engine.input_status()
            if not input_status["ready"]:
                raise RuntimeError(input_status["error"])
            self.engine.stop_tracking()
            self.calibration = CalibrationSession(
                self.engine.camera, self.config, self.registry,
            )
            return {
                "calibration_schema": CALIBRATION_SCHEMA,
                "targets": self.calibration.targets,
                "light_targets": self.calibration.light_targets,
                "pose_targets": self.calibration.pose_targets,
                **self.calibration.status(),
            }

    def start_video(self, purpose="current_gaze", plan=None) -> dict:
        with self._lock:
            if self.calibration is not None and self.calibration.status()["active"]:
                raise RuntimeError("calibration is already active")
            self.config.require_geometry()
            input_status = self.engine.input_status()
            if not input_status["ready"]:
                raise RuntimeError(input_status["error"])
            self.engine.stop_tracking()
            self.calibration = VideoSession(self.engine.camera, self.config, self.registry, purpose=purpose, plan=plan)
            return self.calibration.status()

    def complete_calibration_lighting_profiles(self, profile_names: list[str]) -> dict:
        with self._lock:
            if self.calibration is None:
                raise RuntimeError("calibration has not started")
            return self.calibration.complete_lighting_profiles(profile_names)

    def delete_lighting_profile(self, profile_name: str) -> dict:
        with self._lock:
            if self.calibration is not None and self.calibration.status()["active"]:
                raise RuntimeError("cannot delete a lighting profile during calibration")
            name = str(profile_name or "").strip()
            if name == self.config.lighting_profile and name != "reference":
                updated = ProviderConfig(**asdict(self.config))
                updated.lighting_profile = "reference"
                save_config(updated, self.config_path)
                self.engine.reconfigure(updated)
                self.config = updated
            result = delete_lighting_profile(name)
            self.registry.clear()
            return result

    def retrain_lighting_profiles(self, profile_names: list[str]) -> dict:
        with self._lock:
            if self.calibration is not None and self.calibration.status()["active"]:
                raise RuntimeError("cannot retrain lighting profiles during calibration")
            self.engine.stop_tracking()
            result = retrain_lighting_profiles(self.config, profile_names)
            self.registry.clear()
            return result

    def start_lighting_adaptation(self, profile_name: str, screen_level: float) -> dict:
        with self._lock:
            if self.calibration is not None and self.calibration.status()["active"]:
                raise RuntimeError("calibration is already active")
            self.config.require_geometry()
            input_status = self.engine.input_status()
            if not input_status["ready"]:
                raise RuntimeError(input_status["error"])
            self.engine.stop_tracking()
            self.calibration = LightingAdaptationSession(
                self.engine.camera, self.config, self.registry,
                profile_name, screen_level,
            )
            return {
                "targets": self.calibration.targets,
                **self.calibration.status(),
            }

    def close(self) -> None:
        if self.calibration is not None and self.calibration.status()["active"]:
            self.calibration.cancel()
        self.engine.close()

    def enter_background(self) -> None:
        if self._enter_background is None:
            raise RuntimeError("background mode is unavailable")
        if self.calibration is not None and self.calibration.status()["active"]:
            raise RuntimeError("cannot enter background mode during calibration")
        self.engine.start_tracking()
        self._enter_background()

    def accept_pairing(self, phone_id: str) -> dict:
        if self._accept_pairing is None:
            raise RuntimeError("phone pairing is unavailable")
        return self._accept_pairing(phone_id)

    def forget_pairing(self) -> dict:
        if self._forget_pairing is None:
            raise RuntimeError("phone pairing is unavailable")
        return self._forget_pairing()

    def shutdown_application(self) -> None:
        if self._shutdown_application is None:
            raise RuntimeError("application shutdown is unavailable")
        self._shutdown_application()


def _handler(application: ControlApplication):
    class Handler(BaseHTTPRequestHandler):
        server_version = "OpenGazeLinkControl/1.0"
        protocol_version = "HTTP/1.1"

        def log_message(self, format: str, *args) -> None:
            return

        def _json(self, payload: object, status: int = 200) -> None:
            body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _error(self, error: Exception, status: int = 400) -> None:
            self._json({"ok": False, "error": str(error), "type": type(error).__name__}, status)

        def _body(self) -> dict:
            size = int(self.headers.get("Content-Length", "0") or 0)
            if size <= 0:
                return {}
            payload = json.loads(self.rfile.read(size).decode("utf-8"))
            return payload if isinstance(payload, dict) else {}

        def _gaze_stream(self, include_performance: bool = False) -> None:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Connection", "keep-alive")
            self.end_headers()
            last_sequence = -1
            last_sent_at = 0.0
            try:
                while True:
                    gaze = application.engine.wait_for_gaze(last_sequence, timeout_s=1.0)
                    sequence = int(gaze.get("output_seq", -1))
                    if sequence == last_sequence:
                        self.wfile.write(b": keepalive\n\n")
                    else:
                        last_sequence = sequence
                        now = runtime_clock.monotonic()
                        # The performance page displays one-second aggregates;
                        # serializing and rendering them at camera FPS adds CPU
                        # work without adding information. Preview remains full
                        # rate and carries no performance payload.
                        if include_performance and now - last_sent_at < 0.2:
                            continue
                        streamed_gaze = gaze
                        if not include_performance and "performance" in gaze:
                            streamed_gaze = dict(gaze)
                            streamed_gaze.pop("performance", None)
                        body = json.dumps(
                            streamed_gaze, ensure_ascii=False, separators=(",", ":"),
                        ).encode("utf-8")
                        self.wfile.write(b"data: " + body + b"\n\n")
                        last_sent_at = now
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, OSError):
                return

        def _frame_stream(self) -> None:
            boundary = b"frame"
            # Bound the TCP queue so a slow browser skips frames instead of
            # displaying an increasingly old MJPEG stream.
            self.connection.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 128 * 1024)
            self.connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            self.send_response(200)
            self.send_header(
                "Content-Type", "multipart/x-mixed-replace; boundary=frame",
            )
            self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
            self.send_header("Pragma", "no-cache")
            self.send_header("Connection", "close")
            self.end_headers()
            last_sequence = -1
            try:
                while True:
                    payload, sequence = application.engine.preview_frame_jpeg(
                        last_sequence, timeout_s=0.25,
                    )
                    if payload is None:
                        mode = application.engine.camera.reported_mode()
                        if int(mode.get("width") or 0) <= 0:
                            return
                        continue
                    header = (
                        b"--" + boundary + b"\r\n"
                        b"Content-Type: image/jpeg\r\n"
                        + f"Content-Length: {len(payload)}\r\n\r\n".encode("ascii")
                    )
                    self.wfile.write(header)
                    self.wfile.write(payload)
                    self.wfile.write(b"\r\n")
                    self.wfile.flush()
                    last_sequence = sequence
            except (BrokenPipeError, ConnectionResetError, OSError):
                return

        def do_GET(self) -> None:
            path = self.path.split("?", 1)[0]
            if path == "/api/status":
                # Always return a JSON error for status failures.  Letting an
                # exception escape closes the HTTP socket, which browsers
                # report only as the misleading generic "failed to fetch".
                try:
                    self._json(application.status())
                except Exception as error:
                    self._error(error, status=500)
                return
            if path == "/api/windows-cameras":
                self._json({"cameras": application.windows_cameras()})
                return
            if path == "/api/gaze":
                self._json(application.engine.latest_gaze())
                return
            if path == "/api/gaze/stream":
                query = self.path.split("?", 1)[1] if "?" in self.path else ""
                self._gaze_stream(include_performance="performance=1" in query.split("&"))
                return
            if path == "/api/frame.mjpg":
                self._frame_stream()
                return
            if path == "/api/frame.jpg":
                payload = application.engine.frame_jpeg()
                if payload is None:
                    self.send_error(503, "No camera frame")
                    return
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
                return
            static = {
                "/": "index.html", "/index.html": "index.html",
                "/app.js": "app.js", "/styles.css": "styles.css",
                "/i18n.js": "i18n.js",
                "/video.js": "video.js",
                "/video-plan.js": "video-plan.js",
                "/prediction-plan.js": "prediction-plan.js",
                "/unified-plan.js": "unified-plan.js",
            }.get(path)
            if static is None:
                self.send_error(404)
                return
            payload = (WEB_DIR / static).read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", mimetypes.guess_type(static)[0] or "application/octet-stream")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_POST(self) -> None:
            path = self.path.split("?", 1)[0]
            try:
                body = self._body()
                if path == "/api/config":
                    self._json({"ok": True, "config": application.update_config(body)})
                elif path == "/api/tracking/start":
                    application.start_tracking()
                    self._json({"ok": True})
                elif path == "/api/tracking/stop":
                    application.engine.stop_tracking()
                    self._json({"ok": True})
                elif path == "/api/application/background":
                    application.enter_background()
                    self._json({"ok": True})
                elif path == "/api/application/exit":
                    application.shutdown_application()
                    self._json({"ok": True})
                elif path == "/api/pairing/accept":
                    self._json({"ok": True, **application.accept_pairing(
                        str(body.get("phone_id") or ""),
                    )})
                elif path == "/api/pairing/forget":
                    self._json({"ok": True, **application.forget_pairing()})
                elif path == "/api/calibration/start":
                    self._json({"ok": True, **application.start_calibration()})
                elif path == "/api/video/clock":
                    self._json({"ok": True, "pc_ms": runtime_clock.monotonic() * 1000})
                elif path == "/api/video/start":
                    self._json({"ok": True, **application.start_video()})
                elif path == "/api/prediction/start":
                    self._json({"ok": True, **application.start_video("prediction")})
                elif path == "/api/calibration/unified/start":
                    self._json({"ok": True, **application.start_video("unified", body.get("plan"))})
                elif path in ("/api/video/pause", "/api/video/resume", "/api/video/review"):
                    if not isinstance(application.calibration, VideoSession):
                        raise RuntimeError("VIDEO session has not started")
                    session = application.calibration
                    result = session.pause(body.get("discard_segment")) if path.endswith("pause") else session.resume() if path.endswith("resume") else session.review()
                    self._json({"ok": True, **result})
                elif path == "/api/video/events":
                    if not isinstance(application.calibration, VideoSession):
                        raise RuntimeError("VIDEO session has not started")
                    self._json({"ok": True, **application.calibration.add_events(body)})
                elif path == "/api/video/finish":
                    if not isinstance(application.calibration, VideoSession):
                        raise RuntimeError("VIDEO session has not started")
                    self._json({"ok": True, **application.calibration.finish()})
                elif path == "/api/calibration/lighting-profiles":
                    selected = body.get("profile_names") or []
                    if not isinstance(selected, list):
                        raise ValueError("profile_names must be a list")
                    self._json(application.complete_calibration_lighting_profiles(selected))
                elif path == "/api/lighting-profiles/delete":
                    self._json({"ok": True, **application.delete_lighting_profile(
                        str(body.get("profile_name", "")),
                    )})
                elif path == "/api/lighting-profiles/retrain":
                    selected = body.get("profile_names") or []
                    if not isinstance(selected, list):
                        raise ValueError("profile_names must be a list")
                    self._json({"ok": True, **application.retrain_lighting_profiles(selected)})
                elif path == "/api/lighting-adaptation/start":
                    self._json({"ok": True, **application.start_lighting_adaptation(
                        str(body.get("profile_name", "")),
                        float(body.get("screen_level", 0.42)),
                    )})
                elif path == "/api/calibration/sample":
                    if application.calibration is None:
                        raise RuntimeError("calibration has not started")
                    self._json(application.calibration.capture_next(int(body.get("index", -1))))
                elif path == "/api/calibration/pose":
                    if application.calibration is None:
                        raise RuntimeError("calibration has not started")
                    self._json(application.calibration.capture_pose(int(body.get("index", -1))))
                elif path == "/api/calibration/finish":
                    if application.calibration is None:
                        raise RuntimeError("calibration has not started")
                    self._json(application.calibration.finish())
                elif path == "/api/calibration/cancel":
                    if application.calibration is not None:
                        application.calibration.cancel()
                    self._json({"ok": True})
                else:
                    self.send_error(404)
            except Exception as error:
                self._error(error)

    return Handler


class ControlServer:
    def __init__(self, application: ControlApplication) -> None:
        self.application = application
        config = application.config
        self.server = ThreadingHTTPServer(
            (config.control_bind, config.control_port), _handler(application),
        )
        self.server.daemon_threads = True
        self.url = f"http://{config.control_bind}:{config.control_port}/"
        self._thread: threading.Thread | None = None

    def start(self, open_browser: bool = True) -> str:
        if self._thread is None:
            self._thread = threading.Thread(
                target=self.server.serve_forever,
                kwargs={"poll_interval": 0.2},
                name="control-http",
                daemon=True,
            )
            self._thread.start()
            print(f"OpenGazeLink control: {self.url}")
        if open_browser:
            threading.Timer(0.25, lambda: webbrowser.open(self.url)).start()
        return self.url

    def close(self) -> None:
        if self._thread is None:
            return
        self.server.shutdown()
        self.server.server_close()
        self._thread.join(timeout=2.0)
        self._thread = None


def run_control_server(
    config: ProviderConfig,
    config_path: Path = DEFAULT_CONFIG_PATH,
    open_browser: bool = True,
) -> None:
    application = ControlApplication(config, config_path)
    server = ControlServer(application)
    server.start(open_browser=open_browser)
    try:
        server._thread.join()
    finally:
        server.close()
        application.close()
