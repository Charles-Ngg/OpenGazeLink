from __future__ import annotations

import threading
import time

import cv2
import numpy as np

from .camera import (
    UdpYuvCamera, UdpYuvConfig, WindowsCamera, WindowsCameraConfig,
)
from .config import ProviderConfig
from .extrapolation import FixedHorizonExtrapolator2D
from .model_registry import ModelRegistry
from .motion_diagnostics import MotionDiagnosticsRecorder
from .normalized_eye import (
    NormalizedEyeBackend,
    angles_to_camera_direction,
    intersect_screen_plane,
    screen_camera_origin,
    screen_point_to_pixels,
)
from .shared_memory import GazeSharedMemoryWriter
from .one_euro import OneEuroFilter2D
from .types import GazeSample


class EyeTrackingEngine:
    def __init__(self, config: ProviderConfig, registry: ModelRegistry) -> None:
        self.config = config
        self.registry = registry
        self._lock = threading.Lock()
        self._gaze_condition = threading.Condition(self._lock)
        self._camera = self._open_camera()
        self._tracking_thread: threading.Thread | None = None
        self._tracking_stop = threading.Event()
        self._backend: NormalizedEyeBackend | None = None
        self._writer: GazeSharedMemoryWriter | None = None
        self._latest_gaze: dict = {}
        self._gaze_sequence = 0
        self._last_error = ""
        self._inference_times: list[tuple[float, float]] = []
        self._gaze_filter_timestamp_ms: float | None = None
        self._gaze_filter = OneEuroFilter2D(
            config.one_euro_min_cutoff,
            config.one_euro_beta,
            config.one_euro_derivative_cutoff,
        )
        self._gaze_extrapolator = FixedHorizonExtrapolator2D(
            config.extrapolation_horizon_ms,
            config.extrapolation_max_lead_fraction,
        )
        self._motion_diagnostics = MotionDiagnosticsRecorder()
        self._motion_diagnostics.configure(
            config.motion_diagnostics_enabled,
            self._motion_diagnostic_metadata(config),
        )

    @property
    def camera(self):
        return self._camera

    def _open_camera(self):
        if self.config.input_source == "windows_camera":
            return WindowsCamera(WindowsCameraConfig(
                device_index=self.config.windows_camera_index,
                width=self.config.windows_camera_width,
                height=self.config.windows_camera_height,
                fps=self.config.windows_camera_fps,
                backend=self.config.windows_camera_backend,
                fov_x_degrees=self.config.windows_camera_fov_x_degrees,
                rotate=self.config.rotate,
                mirror=self.config.mirror,
            ))
        return UdpYuvCamera(UdpYuvConfig(
            bind=self.config.udp_bind,
            port=self.config.udp_port,
            rotate=self.config.rotate,
            mirror=self.config.mirror,
        ))

    def reconfigure(self, config: ProviderConfig) -> None:
        was_tracking = self.is_tracking()
        self.stop_tracking()
        old_camera_key = (
            self.config.input_source,
            self.config.udp_bind, self.config.udp_port,
            self.config.rotate, self.config.mirror,
            self.config.windows_camera_index, self.config.windows_camera_width,
            self.config.windows_camera_height, self.config.windows_camera_fps,
            self.config.windows_camera_backend, self.config.windows_camera_fov_x_degrees,
        )
        new_camera_key = (
            config.input_source,
            config.udp_bind, config.udp_port, config.rotate, config.mirror,
            config.windows_camera_index, config.windows_camera_width,
            config.windows_camera_height, config.windows_camera_fps,
            config.windows_camera_backend, config.windows_camera_fov_x_degrees,
        )
        self.config = config
        self._gaze_filter.configure(
            config.one_euro_min_cutoff,
            config.one_euro_beta,
            config.one_euro_derivative_cutoff,
        )
        self._gaze_extrapolator.configure(
            config.extrapolation_horizon_ms,
            config.extrapolation_max_lead_fraction,
        )
        if old_camera_key != new_camera_key:
            self._camera.release()
            self._camera = self._open_camera()
        self._motion_diagnostics.configure(
            config.motion_diagnostics_enabled,
            self._motion_diagnostic_metadata(config),
        )
        if was_tracking:
            try:
                self.start_tracking()
            except Exception as error:
                with self._lock:
                    self._last_error = str(error)

    def is_tracking(self) -> bool:
        return self._tracking_thread is not None and self._tracking_thread.is_alive()

    def _motion_diagnostic_metadata(self, config: ProviderConfig) -> dict:
        return {
            "target_horizon_ms": 80.0,
            "screen": {"width": config.screen_width, "height": config.screen_height},
            "landmarker": config.landmarker,
            "lighting_profile": config.lighting_profile,
            "one_euro": {
                "enabled": config.one_euro_enabled,
                "min_cutoff": config.one_euro_min_cutoff,
                "beta": config.one_euro_beta,
                "derivative_cutoff": config.one_euro_derivative_cutoff,
            },
            "extrapolation": {
                "enabled": config.extrapolation_enabled,
                "horizon_ms": config.extrapolation_horizon_ms,
                "max_lead_fraction": config.extrapolation_max_lead_fraction,
            },
            "camera": self._camera.reported_mode(),
        }

    def start_tracking(self) -> None:
        if self.is_tracking():
            return
        self.config.require_geometry()
        model = self.registry.load(self.config.landmarker)
        model.set_lighting_profile(self.config.lighting_profile)
        screen = model.metadata["screen"]
        if screen != {"width": self.config.screen_width, "height": self.config.screen_height}:
            raise ValueError(
                f"configured screen is {self.config.screen_width}x{self.config.screen_height}, "
                f"model uses {screen.get('width')}x{screen.get('height')}"
            )
        model_payload = model.metadata
        model_input_source = model_payload.get("input_source", "phone_udp")
        if model_input_source != self.config.input_source:
            raise ValueError(
                f"model was calibrated for {model_input_source}, "
                f"but current input source is {self.config.input_source}; recalibrate"
            )
        if self.config.input_source == "windows_camera":
            model_camera = model_payload.get("windows_camera") or {}
            current_camera = {
                "device_index": self.config.windows_camera_index,
                "width": self.config.windows_camera_width,
                "height": self.config.windows_camera_height,
                "fov_x_degrees": self.config.windows_camera_fov_x_degrees,
                "rotate": self.config.rotate,
                "mirror": self.config.mirror,
            }
            for key, value in current_camera.items():
                model_value = model_camera.get(key)
                if isinstance(value, float):
                    matches = model_value is not None and abs(float(model_value) - value) <= 0.01
                else:
                    matches = model_value == value
                if not matches:
                    raise ValueError(
                        "Windows camera configuration differs from calibration; recalibrate"
                    )
        configured_screen_origin = tuple(float(value) for value in screen_camera_origin(
            self.config.screen_width,
            self.config.screen_height,
            self.config.screen_diagonal_inches,
            self.config.camera_position_screen_cm,
        ))
        model_screen_origin = model_payload.get("screen_camera_origin_cm")
        if model_screen_origin is None:
            raise ValueError("model predates configurable screen/camera geometry; recalibrate")
        model_screen_origin = tuple(float(value) for value in model_screen_origin)
        model_diagonal = float(model_payload.get("screen_diagonal_inches", 0.0))
        if not np.allclose(model_screen_origin, configured_screen_origin, atol=0.05) or not np.isclose(
            model_diagonal, self.config.screen_diagonal_inches, atol=0.01,
        ):
            raise ValueError("configured screen/camera geometry differs from calibration; recalibrate")
        self._backend = NormalizedEyeBackend(self.config.landmarker)
        self._writer = GazeSharedMemoryWriter(self.config.shared_memory_name)
        self._writer.open()
        self._tracking_stop.clear()
        self._last_error = ""
        self._tracking_thread = threading.Thread(
            target=self._tracking_loop,
            args=(model, configured_screen_origin),
            name="gaze-runtime",
            daemon=True,
        )
        self._tracking_thread.start()

    def stop_tracking(self) -> None:
        self._tracking_stop.set()
        if self._tracking_thread is not None:
            self._tracking_thread.join(timeout=2.0)
            self._tracking_thread = None
        if self._backend is not None:
            self._backend.close()
            self._backend = None
        if self._writer is not None:
            self._writer.write_invalid(self.config.screen_width, self.config.screen_height)
            self._writer.close()
            self._writer = None
        self._gaze_filter.reset()
        self._gaze_filter_timestamp_ms = None
        self._gaze_extrapolator.reset()
        self._publish_gaze({"valid": False})

    def _tracking_loop(self, model, model_screen_origin) -> None:
        last_sequence = -1
        while not self._tracking_stop.is_set():
            ok, frame, t_ms, sequence = self._camera.read_latest(last_sequence, timeout_s=0.05)
            if not ok or frame is None:
                continue
            last_sequence = sequence
            frame_timing = self._camera.latest_frame_timing(sequence)
            process_start_ns = time.monotonic_ns()
            wall_time_ns = time.time_ns()
            started = time.perf_counter()
            try:
                camera_model = self._camera.camera_model()
                if camera_model.get("source") == "estimated_frame_center":
                    raise RuntimeError("waiting for Camera2 intrinsics")
                assert self._backend is not None
                observation = self._backend.predict(frame, t_ms, camera_model)
                if observation is None:
                    raise RuntimeError("landmarker produced no face")
                right_image = model.model_image(observation.right, "right")
                left_image = model.model_image(observation.left, "left")
                right, left = model.predict_images(
                    [right_image, left_image], ["right", "left"],
                )
                points = {}
                for name, prediction, eye in (
                    ("right", right, observation.right),
                    ("left", left, observation.left),
                ):
                    direction = angles_to_camera_direction(
                        prediction.yaw, prediction.pitch, observation.rotation,
                    )
                    intersection = intersect_screen_plane(
                        eye.eye_center_camera, direction, model_screen_origin,
                    )
                    points[name] = screen_point_to_pixels(
                        intersection,
                        self.config.screen_width,
                        self.config.screen_height,
                        self.config.screen_diagonal_inches,
                        model_screen_origin,
                    )
                combined = (
                    0.5 * (points["right"][0] + points["left"][0]),
                    0.5 * (points["right"][1] + points["left"][1]),
                )
                valid = bool(np.isfinite(combined).all())
                raw_combined = combined
                if (
                    self._gaze_filter_timestamp_ms is not None
                    and (t_ms <= self._gaze_filter_timestamp_ms or t_ms - self._gaze_filter_timestamp_ms > 250.0)
                ):
                    self._gaze_filter.reset()
                    self._gaze_extrapolator.reset()
                self._gaze_filter_timestamp_ms = float(t_ms)
                stable_combined = (
                    self._gaze_filter.update(
                        (
                            raw_combined[0] / max(1.0, self.config.screen_width - 1),
                            raw_combined[1] / max(1.0, self.config.screen_height - 1),
                        ),
                        float(t_ms) / 1000.0,
                    )
                    if valid and self.config.one_euro_enabled
                    else raw_combined
                )
                if valid and self.config.one_euro_enabled:
                    stable_combined = (
                        stable_combined[0] * max(1.0, self.config.screen_width - 1),
                        stable_combined[1] * max(1.0, self.config.screen_height - 1),
                    )
                else:
                    self._gaze_filter.reset()
                extrapolation = {
                    "horizon_ms": self.config.extrapolation_horizon_ms,
                    "velocity_px_per_ms": [0.0, 0.0],
                    "lead_px": [0.0, 0.0],
                    "lead_distance_px": 0.0,
                    "sample_count": 0,
                    "mode": "disabled",
                    "spatial_span_px": 0.0,
                }
                if valid and self.config.extrapolation_enabled:
                    output_combined, extrapolation = self._gaze_extrapolator.update(
                        raw_combined, stable_combined, t_ms,
                        (self.config.screen_width, self.config.screen_height),
                    )
                    if extrapolation.get("reset_filter"):
                        self._gaze_filter.reset()
                else:
                    self._gaze_extrapolator.reset()
                    if not valid:
                        self._gaze_filter_timestamp_ms = None
                    output_combined = stable_combined
                gaze = GazeSample(
                    t_ms=float(t_ms), x=float(output_combined[0]), y=float(output_combined[1]),
                    raw_x=float(raw_combined[0]), raw_y=float(raw_combined[1]),
                    confidence=1.0 if valid else 0.0,
                    valid=valid, status="TRACKING" if valid else "INVALID",
                )
                assert self._writer is not None
                self._writer.write(gaze, self.config.screen_width, self.config.screen_height)
                processing_ms = (time.perf_counter() - started) * 1000.0
                process_done_ns = time.monotonic_ns()
                postprocess = {
                    "one_euro": bool(self.config.one_euro_enabled),
                    "min_cutoff": self.config.one_euro_min_cutoff,
                    "beta": self.config.one_euro_beta,
                    "derivative_cutoff": self.config.one_euro_derivative_cutoff,
                    "extrapolation": bool(self.config.extrapolation_enabled),
                    "extrapolation_horizon_ms": self.config.extrapolation_horizon_ms,
                    "extrapolation_max_lead_fraction": self.config.extrapolation_max_lead_fraction,
                    "extrapolation_state": extrapolation,
                }
                self._motion_diagnostics.record({
                    "type": "frame",
                    "phone_sensor_time_ns": int(frame_timing.get(
                        "phone_sensor_time_ns", round(float(t_ms) * 1_000_000.0),
                    )),
                    "phone_send_time_ns": int(frame_timing.get("phone_send_time_ns", 0)),
                    "phone_frame_sequence": int(frame_timing.get("phone_frame_sequence", -1)),
                    "pc_read_sequence": int(sequence),
                    "pc_first_packet_monotonic_ns": int(frame_timing.get("pc_first_packet_monotonic_ns", 0)),
                    "pc_decode_done_monotonic_ns": int(frame_timing.get("pc_decode_done_monotonic_ns", 0)),
                    "pc_process_start_monotonic_ns": process_start_ns,
                    "pc_process_done_monotonic_ns": process_done_ns,
                    "pc_wall_time_ns": wall_time_ns,
                    "valid": valid,
                    "right_px": list(points["right"]),
                    "left_px": list(points["left"]),
                    "raw_combined_px": list(raw_combined),
                    "filtered_combined_px": list(stable_combined),
                    "output_combined_px": list(output_combined),
                    "right_angles_rad": [right.yaw, right.pitch],
                    "left_angles_rad": [left.yaw, left.pitch],
                    "head_rotation_rad": [
                        observation.head_yaw, observation.head_pitch, observation.head_roll,
                    ],
                    "head_translation_cm": list(observation.translation),
                    "pnp_reprojection_error_px": observation.pnp_reprojection_error_px,
                    "landmarker_detection_ms": observation.detection_ms,
                    "eye_normalization_ms": observation.normalization_ms,
                    "processing_ms": processing_ms,
                    "postprocess": postprocess,
                })
                self._publish_gaze({
                        "t_ms": float(t_ms), "valid": valid,
                        "right": list(points["right"]), "left": list(points["left"]),
                        "combined": list(output_combined),
                        "raw_combined": list(raw_combined),
                        "filtered_combined": list(stable_combined),
                        "postprocess": postprocess,
                        "right_angles": [right.yaw, right.pitch],
                        "left_angles": [left.yaw, left.pitch],
                        "head": [observation.head_yaw, observation.head_pitch, observation.head_roll],
                        "landmarker": self.config.landmarker,
                        "model": "cnn",
                        "processing_ms": processing_ms,
                    })
                with self._lock:
                    self._last_error = ""
            except Exception as error:
                processing_ms = (time.perf_counter() - started) * 1000.0
                self._motion_diagnostics.record({
                    "type": "frame",
                    "phone_sensor_time_ns": int(frame_timing.get(
                        "phone_sensor_time_ns", round(float(t_ms) * 1_000_000.0),
                    )),
                    "phone_send_time_ns": int(frame_timing.get("phone_send_time_ns", 0)),
                    "phone_frame_sequence": int(frame_timing.get("phone_frame_sequence", -1)),
                    "pc_read_sequence": int(sequence),
                    "pc_process_start_monotonic_ns": process_start_ns,
                    "pc_process_done_monotonic_ns": time.monotonic_ns(),
                    "pc_wall_time_ns": wall_time_ns,
                    "valid": False,
                    "error": str(error),
                    "processing_ms": processing_ms,
                })
                with self._lock:
                    self._last_error = str(error)
                if self._writer is not None:
                    self._writer.write_invalid(self.config.screen_width, self.config.screen_height)
                self._publish_gaze({
                    "valid": False,
                    "error": str(error),
                    "processing_ms": processing_ms,
                })
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            now = time.monotonic()
            with self._lock:
                self._inference_times.append((now, elapsed_ms))
                cutoff = now - 2.0
                while self._inference_times and self._inference_times[0][0] < cutoff:
                    self._inference_times.pop(0)

    def latest_gaze(self) -> dict:
        with self._lock:
            return dict(self._latest_gaze)

    def wait_for_gaze(self, after_sequence: int, timeout_s: float = 1.0) -> dict:
        with self._gaze_condition:
            self._gaze_condition.wait_for(
                lambda: int(self._latest_gaze.get("output_seq", -1)) != after_sequence,
                timeout=timeout_s,
            )
            return dict(self._latest_gaze)

    def _publish_gaze(self, payload: dict) -> None:
        with self._gaze_condition:
            self._gaze_sequence += 1
            self._latest_gaze = {
                **payload,
                "output_seq": self._gaze_sequence,
                "published_at_ms": time.monotonic() * 1000.0,
            }
            self._gaze_condition.notify_all()

    def preview_frame_jpeg(
        self, after_sequence: int = -1, quality: int = 82, timeout_s: float = 0.25,
    ) -> tuple[bytes | None, int]:
        ok, frame, _, sequence = self._camera.read_latest(after_sequence, timeout_s=timeout_s)
        if not ok or frame is None:
            return None, after_sequence
        encoded, payload = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
        return (payload.tobytes() if encoded else None), sequence

    def frame_jpeg(self, quality: int = 82) -> bytes | None:
        payload, _ = self.preview_frame_jpeg(quality=quality, timeout_s=0.02)
        return payload

    def input_status(self) -> dict:
        mode = self._camera.reported_mode()
        if self.config.input_source == "windows_camera" and (
            int(mode.get("width") or 0) <= 0 or int(mode.get("height") or 0) <= 0
        ):
            return {
                "ready": False,
                "error": mode.get("error") or "Windows camera frame is unavailable",
            }
        if int(mode.get("width") or 0) <= 0 or int(mode.get("height") or 0) <= 0:
            return {"ready": False, "error": "未收到手机画面"}
        camera_model = self._camera.camera_model()
        if camera_model.get("source") == "estimated_frame_center":
            return {
                "ready": False,
                "error": "未收到 Camera2 内参，请在手机端点击“发送内参”",
            }
        return {"ready": True, "error": ""}

    def status(self) -> dict:
        camera_model = self._camera.camera_model()
        with self._lock:
            inference = [value for _, value in self._inference_times]
            error = self._last_error
        return {
            "tracking": self.is_tracking(),
            "input": self.input_status(),
            "selection": {
                "landmarker": self.config.landmarker,
                "model": "cnn",
            },
            "camera": self._camera.reported_mode(),
            "intrinsics": camera_model,
            "inference_ms": {
                "mean": float(np.mean(inference)) if inference else 0.0,
                "p95": float(np.percentile(inference, 95.0)) if inference else 0.0,
            },
            "shared_memory": {
                "name": self.config.shared_memory_name,
                "active": self.is_tracking(),
            },
            "motion_diagnostics": self._motion_diagnostics.status(),
            "error": error,
        }

    def close(self) -> None:
        self.stop_tracking()
        self._motion_diagnostics.stop()
        self._camera.release()
