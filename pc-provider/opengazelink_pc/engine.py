from __future__ import annotations

import threading
import time
from . import runtime_clock

import cv2
import numpy as np

from .camera import (
    UdpYuvCamera, UdpYuvConfig, WindowsCamera, WindowsCameraConfig,
)
from .config import ProviderConfig
from .extrapolation import FixedHorizonExtrapolator2D
from .latency import ADDITIVE_FIELDS, PipelineLatencyTracker, finite_milliseconds
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
from .shared_eye_models import fuse_screen_points
from .one_euro import OneEuroFilter2D
from .event_temporal import EventTemporalFilter
from .stability_profile import read_profile
from .prediction_timing import PredictionClock
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
        self._preprocess_queue_drops = 0
        self._gaze_filter_timestamp_ms: float | None = None
        self._prediction_filter_timestamp_ms: float | None = None
        self._prediction_filter: OneEuroFilter2D | None = None
        self._prediction_filter_config: dict = {"one_euro_enabled": False}
        self._prediction_clock = PredictionClock()
        self._latency = PipelineLatencyTracker()
        self._gaze_filter = OneEuroFilter2D(
            config.one_euro_min_cutoff,
            config.one_euro_beta,
            config.one_euro_derivative_cutoff,
        )
        self._gaze_extrapolator = FixedHorizonExtrapolator2D(
            config.extrapolation_horizon_ms,
            config.extrapolation_max_lead_fraction,
        )
        self._event_temporal = EventTemporalFilter(config.one_euro_min_cutoff, config.one_euro_beta,
                                                   config.one_euro_derivative_cutoff)
        self._motion_diagnostics = MotionDiagnosticsRecorder()
        self._motion_diagnostics.configure(
            config.motion_diagnostics_enabled,
            self._motion_diagnostic_metadata(config),
        )

    @staticmethod
    def _configured_filter(config: ProviderConfig) -> dict:
        return {
            "one_euro_enabled": bool(config.one_euro_enabled),
            "one_euro_min_cutoff": float(config.one_euro_min_cutoff),
            "one_euro_beta": float(config.one_euro_beta),
            "one_euro_derivative_cutoff": float(config.one_euro_derivative_cutoff),
        }

    @staticmethod
    def _filter_configs_match(left: dict, right: dict) -> bool:
        if bool(left.get("one_euro_enabled")) != bool(right.get("one_euro_enabled")):
            return False
        if not bool(left.get("one_euro_enabled")):
            return True
        keys = ("one_euro_min_cutoff", "one_euro_beta", "one_euro_derivative_cutoff")
        try:
            return all(abs(float(left[key]) - float(right[key])) <= 1e-6 for key in keys)
        except (KeyError, TypeError, ValueError):
            return False

    def _configure_prediction_filter(self, model) -> None:
        self._event_temporal.reset()
        getter = getattr(model, "forecast_filter_config", None)
        expected = getter() if callable(getter) else {"one_euro_enabled": False}
        self._prediction_filter_config = dict(expected)
        if expected.get("one_euro_enabled"):
            self._prediction_filter = OneEuroFilter2D(
                expected["one_euro_min_cutoff"],
                expected["one_euro_beta"],
                expected["one_euro_derivative_cutoff"],
            )
        else:
            self._prediction_filter = None
        self._prediction_filter_timestamp_ms = None

    def _configure_automatic_stability(self, model) -> None:
        camera = getattr(self._camera, "camera_model", lambda: {})()
        profile = read_profile(self.config, camera, getattr(model, "metadata", {}))
        self._event_temporal.set_stability_profile(profile["parameters"] if profile else None)

    def _reset_live_state(self) -> None:
        self._event_temporal.reset()
        self._gaze_filter.reset()
        self._gaze_filter_timestamp_ms = None
        if self._prediction_filter is not None:
            self._prediction_filter.reset()
        self._prediction_filter_timestamp_ms = None
        self._gaze_extrapolator.reset()
        self._prediction_clock.reset()

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
            allowed_source_ip=(self.config.paired_phone_address or "0.0.0.0") if self.config.paired_phone_id else None,
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
        self._event_temporal.configure(config.one_euro_min_cutoff, config.one_euro_beta,
                                       config.one_euro_derivative_cutoff)
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
            "gaze_model": config.gaze_model,
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
        model = self.registry.load(self.config.landmarker, self.config.gaze_model)
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
        self._backend = NormalizedEyeBackend(
            self.config.landmarker, conditioned=bool(getattr(model, "is_conditioned", False)),
        )
        self._configure_prediction_filter(model)
        self._configure_automatic_stability(model)
        self._reset_live_state()
        self._latency.reset()
        self._writer = GazeSharedMemoryWriter(self.config.shared_memory_name)
        self._writer.open()
        self._tracking_stop.clear()
        from .runtime_scheduling import configure_tracking_process
        self._scheduling = configure_tracking_process()
        import logging
        logging.getLogger("eyetracing").info("tracking scheduling: %s", self._scheduling)
        if getattr(model, "temporal", False):
            model.reset_temporal()
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
            self._tracking_thread.join()
            self._tracking_thread = None
        from .runtime_scheduling import restore_tracking_process
        restore_tracking_process(getattr(self, "_scheduling", {}))
        self._scheduling = {}
        if self._backend is not None:
            self._backend.close()
            self._backend = None
        if self._writer is not None:
            self._writer.write_invalid(self.config.screen_width, self.config.screen_height)
            self._writer.close()
            self._writer = None
        self._reset_live_state()
        self._prediction_filter = None
        self._prediction_filter_config = {"one_euro_enabled": False}
        self._publish_gaze({"valid": False})

    def _latency_sample(
        self, frame_timing: dict, prediction_timing: dict, extrapolation: dict,
        process_start_ns: int, process_done_ns: int, backend_ms: float,
        detection_ms: float, normalization_ms: float, gaze_model_ms: float,
        projection_fusion_ms: float, prediction_postprocess_ms: float,
        shared_memory_write_ms: float,
    ) -> dict:
        processing_ms = (process_done_ns - process_start_ns) / 1_000_000.0
        backend_overhead_ms = max(0.0, backend_ms - detection_ms - normalization_ms)
        accounted_processing = sum((
            detection_ms, normalization_ms, backend_overhead_ms, gaze_model_ms,
            projection_fusion_ms, prediction_postprocess_ms, shared_memory_write_ms,
        ))
        other_processing_ms = max(0.0, processing_ms - accounted_processing)

        first_packet_ns = float(frame_timing.get("pc_first_packet_monotonic_ns", 0))
        decode_done_ns = float(frame_timing.get("pc_decode_done_monotonic_ns", 0))
        first_packet_ms = finite_milliseconds(first_packet_ns / 1_000_000.0, maximum=1e15) if first_packet_ns > 0 else None
        decode_done_ms = finite_milliseconds(decode_done_ns / 1_000_000.0, maximum=1e15) if decode_done_ns > 0 else None
        source_pc_ms = finite_milliseconds(
            prediction_timing.get("source_pc_ms_proxy"), maximum=1e15,
        )
        basis = str(prediction_timing.get("clock_basis") or "unavailable")
        receive_decode_ms = None
        pc_queue_ms = None
        if first_packet_ms is not None and decode_done_ms is not None and decode_done_ms >= first_packet_ms:
            receive_decode_ms = decode_done_ms - first_packet_ms
            pc_queue_ms = max(0.0, process_start_ns / 1_000_000.0 - decode_done_ms)
        elif basis == "camera_read_completion_proxy" and source_pc_ms is not None:
            pc_queue_ms = max(0.0, process_start_ns / 1_000_000.0 - source_pc_ms)

        source_to_shared_ms = None
        if source_pc_ms is not None:
            source_to_shared_ms = finite_milliseconds(process_done_ns / 1_000_000.0 - source_pc_ms)
        display_delay_ms = finite_milliseconds(prediction_timing.get("display_delay_ms_assumed"))
        source_to_display_ms = (
            source_to_shared_ms + display_delay_ms
            if source_to_shared_ms is not None and display_delay_ms is not None else None
        )
        sample = {
            "clock_basis": basis,
            "unknown_transport_floor": bool(prediction_timing.get("unknown_transport_floor")),
            "is_sensor_to_photon_measurement": False,
            "phone_capture_to_send_ms": prediction_timing.get("phone_capture_to_send_ms"),
            "transport_excess_ms_proxy": prediction_timing.get("transport_excess_ms_proxy"),
            "transport_to_first_packet_ms": prediction_timing.get("transport_to_first_packet_ms", prediction_timing.get("transport_excess_ms_proxy")),
            "clock_probe_rtt_ms": prediction_timing.get("clock_probe_rtt_ms"),
            "clock_probe_uncertainty_ms": prediction_timing.get("clock_probe_uncertainty_ms"),
            "clock_probe_age_ms": prediction_timing.get("clock_probe_age_ms"),
            "transport_signed_estimate_ms": prediction_timing.get("transport_signed_estimate_ms"),
            "phone_to_pc_offset_ns": prediction_timing.get("phone_to_pc_offset_ns"),
            "receive_decode_ms": receive_decode_ms,
            "packet_assembly_ms": self._timing_interval_ms(frame_timing, "pc_first_packet_monotonic_ns", "pc_last_packet_monotonic_ns"),
            "decode_queue_ms": self._timing_interval_ms(frame_timing, "pc_last_packet_monotonic_ns", "pc_decode_start_monotonic_ns"),
            "jpeg_decode_ms": self._timing_interval_ms(frame_timing, "pc_decode_start_monotonic_ns", "pc_decode_done_monotonic_ns"),
            "h264_reference_decode_ms": frame_timing.get("h264_reference_decode_ms"),
            "h264_image_queue_ms": frame_timing.get("h264_image_queue_ms"),
            "h264_image_conversion_ms": frame_timing.get("h264_image_conversion_ms"),
            "pc_clock": runtime_clock.info(),
            "pc_queue_ms": pc_queue_ms,
            "face_landmarks_ms": detection_ms,
            "eye_normalization_ms": normalization_ms,
            "backend_overhead_ms": backend_overhead_ms,
            "gaze_model_ms": gaze_model_ms,
            "projection_fusion_ms": projection_fusion_ms,
            "prediction_postprocess_ms": prediction_postprocess_ms,
            "shared_memory_write_ms": shared_memory_write_ms,
            "other_processing_ms": other_processing_ms,
            "pc_processing_ms": processing_ms,
            "source_to_shared_memory_ms_proxy": source_to_shared_ms,
            "display_delay_ms_assumed": display_delay_ms,
            "source_to_display_ms_estimate": source_to_display_ms,
            "prediction_horizon_ms": extrapolation.get("horizon_ms"),
        }
        sample["stage_sum_ms"] = sum(
            value for field in ADDITIVE_FIELDS
            if (value := finite_milliseconds(sample.get(field))) is not None
        )
        if source_to_shared_ms is not None:
            sample["sum_error_ms"] = sample["stage_sum_ms"] - source_to_shared_ms
        return sample

    @staticmethod
    def _timing_interval_ms(timing: dict, start: str, end: str):
        a, b = timing.get(start, 0), timing.get(end, 0)
        return finite_milliseconds((b - a) / 1_000_000.0) if a > 0 and b >= a else None

    def _tracking_loop(self, model, model_screen_origin) -> None:
        from .runtime_scheduling import set_realtime_thread_priority, TrackingThreadSwitch
        set_realtime_thread_priority()
        from .frame_pacing import FramePacer
        from .latest_preprocessor import LatestEyePreprocessor
        from .runtime_diagnostics import GCPauseMonitor, FrameTimingProbe, TrackingGCPolicy
        with TrackingThreadSwitch() as thread_switch, TrackingGCPolicy() as gc_policy, FramePacer() as pacer:
            if hasattr(self, '_scheduling'):
                self._scheduling['gc'] = gc_policy.status
                self._scheduling.update(thread_switch.status)
            monitor = GCPauseMonitor()
            monitor.start()
            self._frame_timing_probe = FrameTimingProbe(monitor)
            pipeline = None
            try:
                assert self._backend is not None
                pipeline = LatestEyePreprocessor(self._camera, self._backend, self._tracking_stop, pacer)
                pipeline.start()
                self._tracking_loop_paced(model, model_screen_origin, pipeline)
            finally:
                try:
                    if pipeline is not None:
                        pipeline.close()
                finally:
                    monitor.close()
                    if pipeline is not None:
                        self._preprocess_queue_drops = pipeline.dropped

    def _tracking_loop_paced(self, model, model_screen_origin, pipeline) -> None:
        from .latest_preprocessor import is_pipeline_end
        while not self._tracking_stop.is_set():
            prepared = pipeline.get(timeout_s=.1)
            self._preprocess_queue_drops = pipeline.dropped
            if prepared is None:
                continue
            if is_pipeline_end(prepared):
                break
            sequence = prepared["sequence"]
            t_ms = prepared["t_ms"]
            frame_timing = prepared["frame_timing"]
            process_start_ns = prepared["process_start_ns"]
            wall_time_ns = prepared["wall_time_ns"]
            started = prepared["started"]
            backend_start_ns = prepared["backend_start_ns"]
            backend_done_ns = prepared["backend_done_ns"]
            observation = prepared["observation"]
            try:
                if prepared["error"]:
                    raise RuntimeError(prepared["error"])
                assert observation is not None
                if prepared["pipeline_discontinuity"]:
                    # Parameters are loaded at start/reconfigure. A missed face
                    # resets temporal state, not the profile file on disk.
                    self._event_temporal.reset()
                if prepared["pipeline_discontinuity"] and getattr(model, "temporal", False):
                    model.reset_temporal()
                    self._gaze_filter.reset()
                    if self._prediction_filter is not None:
                        self._prediction_filter.reset()
                    self._gaze_extrapolator.reset()
                # Geometry runs independently; these eye inputs contain pixels
                # from the current camera frame, with its original timestamp.
                gaze_model_start_ns = runtime_clock.monotonic_ns()
                if getattr(model, "is_conditioned", False):
                    if observation.conditioned_inputs is None:
                        raise RuntimeError("conditioned-eye inputs are unavailable")
                    if getattr(model, "temporal", False):
                        if min(observation.right.aperture_ratio, observation.left.aperture_ratio) < .18:
                            raise RuntimeError("blink: VIDEO memory reset")
                        right, left = model.predict_inputs(observation.conditioned_inputs, ["right", "left"], timestamp_ms=t_ms,
                                                           spatial_only=True)
                        reset_active = (model.temporal_reset_active()
                                        if hasattr(model, "temporal_reset_active")
                                        else model.reset_probability >= .8)
                        if reset_active:
                            self._gaze_filter.reset()
                            if self._prediction_filter is not None:
                                self._prediction_filter.reset()
                            self._gaze_extrapolator.reset()
                    else:
                        right, left = model.predict_inputs(observation.conditioned_inputs, ["right", "left"])
                else:
                    right_image = model.model_image(observation.right, "right")
                    left_image = model.model_image(observation.left, "left")
                    right, left = model.predict_images(
                        [right_image, left_image], ["right", "left"],
                    )
                gaze_model_done_ns = runtime_clock.monotonic_ns()
                projection_start_ns = gaze_model_done_ns
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
                fusion_weights = [right.fusion_weight, left.fusion_weight]
                combined = fuse_screen_points([points["right"], points["left"]], fusion_weights)
                valid = bool(np.isfinite(combined).all())
                raw_combined = combined
                projection_done_ns = runtime_clock.monotonic_ns()
                postprocess_start_ns = projection_done_ns
                if (
                    self._gaze_filter_timestamp_ms is not None
                    and (t_ms <= self._gaze_filter_timestamp_ms or t_ms - self._gaze_filter_timestamp_ms > 250.0)
                ):
                    self._gaze_filter.reset()
                    self._gaze_extrapolator.reset()
                if (
                    self._prediction_filter_timestamp_ms is not None
                    and (t_ms <= self._prediction_filter_timestamp_ms or t_ms - self._prediction_filter_timestamp_ms > 250.0)
                    and self._prediction_filter is not None
                ):
                    self._prediction_filter.reset()
                self._gaze_filter_timestamp_ms = float(t_ms)
                self._prediction_filter_timestamp_ms = float(t_ms)
                # A single event-aware output path. The public switch controls
                # only prediction; disabling it must not restore a legacy model,
                # a standalone filter or a different extrapolator.
                prediction_timing = self._prediction_clock.observe(
                    t_ms, frame_timing, runtime_clock.monotonic() * 1000.0,
                    self.config.prediction_display_delay_ms,
                )
                if valid:
                    size_cm = self.config.screen_diagonal_inches * 2.54
                    depth = abs(float(np.mean([observation.right.eye_center_camera[2],
                                              observation.left.eye_center_camera[2]])) - float(model_screen_origin[2]))
                    pixels_per_degree = np.hypot(self.config.screen_width, self.config.screen_height) / size_cm * max(1., depth) * np.pi / 180.
                    horizon = 0.
                    if self.config.event_temporal_enabled:
                        horizon = prediction_timing.get("horizon_ms_proxy") or 0.
                        if (prediction_timing.get("clock_probe_age_ms") or 0) > 5000 or (prediction_timing.get("clock_probe_uncertainty_ms") or 0) > 20:
                            horizon = 0.
                    output_combined, extrapolation = self._event_temporal.update(
                        raw_combined, t_ms, (self.config.screen_width, self.config.screen_height),
                        pixels_per_degree=pixels_per_degree, head_rotation=observation.rotation,
                        horizon_ms=horizon, max_lead_fraction=self.config.extrapolation_max_lead_fraction,
                        smooth=True)
                    stable_combined = extrapolation["stable_px"]
                else:
                    self._event_temporal.reset()
                    output_combined = stable_combined = raw_combined
                    extrapolation = {"mode": "invalid", "horizon_ms": 0., "prediction_active": False,
                                     "stability_active": False, "lead_px": [0., 0.], "lead_distance_px": 0.}
                postprocess_done_ns = runtime_clock.monotonic_ns()
                gaze = GazeSample(
                    t_ms=float(t_ms), x=float(output_combined[0]), y=float(output_combined[1]),
                    raw_x=float(raw_combined[0]), raw_y=float(raw_combined[1]),
                    confidence=1.0 if valid else 0.0,
                    valid=valid, status="TRACKING" if valid else "INVALID",
                )
                assert self._writer is not None
                shared_memory_start_ns = runtime_clock.monotonic_ns()
                self._writer.write(gaze, self.config.screen_width, self.config.screen_height)
                process_done_ns = runtime_clock.monotonic_ns()
                processing_ms = (process_done_ns - process_start_ns) / 1_000_000.0
                latency_sample = self._latency_sample(
                    frame_timing, prediction_timing, extrapolation,
                    process_start_ns, process_done_ns,
                    (backend_done_ns - backend_start_ns) / 1_000_000.0,
                    float(observation.detection_ms), float(observation.normalization_ms),
                    (gaze_model_done_ns - gaze_model_start_ns) / 1_000_000.0,
                    (projection_done_ns - projection_start_ns) / 1_000_000.0,
                    (postprocess_done_ns - postprocess_start_ns) / 1_000_000.0,
                    (process_done_ns - shared_memory_start_ns) / 1_000_000.0,
                )
                latency_sample.update(self._frame_timing_probe.sample(prepared, process_done_ns))
                performance = self._latency.add(latency_sample, observed_at_ms=process_done_ns / 1_000_000.0)
                configured_filter = self._configured_filter(self.config)
                postprocess = {
                    "face_geometry": prepared.get("face_geometry", {}),
                    "one_euro": bool(extrapolation.get("stability_active", False)),
                    "automatic_stability": True,
                    "stability_calibrated": self._event_temporal.stability_calibrated,
                    "min_cutoff": self.config.one_euro_min_cutoff,
                    "beta": self.config.one_euro_beta,
                    "derivative_cutoff": self.config.one_euro_derivative_cutoff,
                    "extrapolation": False,
                    "extrapolation_horizon_ms": self.config.extrapolation_horizon_ms,
                    "extrapolation_max_lead_fraction": self.config.extrapolation_max_lead_fraction,
                    "extrapolation_state": extrapolation,
                    "event_temporal_enabled": bool(self.config.event_temporal_enabled),
                    "video_forecast_enabled": False,
                    "video_forecast_available": getattr(model, "forecast", None) is not None,
                    "video_forecast_active": False,
                    "prediction_filter": dict(self._prediction_filter_config),
                    "prediction_filter_matches_config": self._filter_configs_match(
                        self._prediction_filter_config, configured_filter,
                    ),
                    "temporal_reset_probability": float(getattr(model, "reset_probability", 0.)),
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
                    "eye_fusion_weights": fusion_weights,
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
                    "performance": performance["current"],
                    "postprocess": postprocess,
                })
                self._publish_gaze({
                        "t_ms": float(t_ms), "valid": valid,
                        "right": list(points["right"]), "left": list(points["left"]),
                        "combined": list(output_combined),
                        "raw_combined": list(raw_combined),
                        "eye_fusion_weights": fusion_weights,
                        "filtered_combined": list(stable_combined),
                        "postprocess": postprocess,
                        "right_angles": [right.yaw, right.pitch],
                        "left_angles": [left.yaw, left.pitch],
                        "head": [observation.head_yaw, observation.head_pitch, observation.head_roll],
                        "landmarker": self.config.landmarker,
                        "model": self.config.gaze_model,
                        "processing_ms": processing_ms,
                        "performance": performance,
                    })
                with self._lock:
                    self._last_error = ""
            except Exception as error:
                failed_done_ns = runtime_clock.monotonic_ns()
                probe = getattr(self, "_frame_timing_probe", None)
                if probe is not None:
                    self._latency.add({**probe.sample(prepared, failed_done_ns), "error": str(error),
                                       "pc_processing_ms": (failed_done_ns-process_start_ns)/1e6},
                                      observed_at_ms=failed_done_ns/1e6)
                event_temporal = getattr(self, "_event_temporal", None)
                if event_temporal is not None:
                    event_temporal.reset()
                processing_ms = (time.perf_counter() - started) * 1000.0
                if getattr(model, "temporal", False):
                    model.reset_temporal()
                    self._gaze_filter.reset()
                    if self._prediction_filter is not None:
                        self._prediction_filter.reset()
                    self._gaze_extrapolator.reset()
                self._motion_diagnostics.record({
                    "type": "frame",
                    "phone_sensor_time_ns": int(frame_timing.get(
                        "phone_sensor_time_ns", round(float(t_ms) * 1_000_000.0),
                    )),
                    "phone_send_time_ns": int(frame_timing.get("phone_send_time_ns", 0)),
                    "phone_frame_sequence": int(frame_timing.get("phone_frame_sequence", -1)),
                    "pc_read_sequence": int(sequence),
                    "pc_process_start_monotonic_ns": process_start_ns,
                    "pc_process_done_monotonic_ns": runtime_clock.monotonic_ns(),
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
            now = runtime_clock.monotonic()
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
                "published_at_ms": runtime_clock.monotonic() * 1000.0,
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
                "model": self.config.gaze_model,
            },
            "camera": self._camera.reported_mode(),
            "intrinsics": camera_model,
            "preprocessing": {
                "mode": "asynchronous_latest_only",
                "queue_drops": self._preprocess_queue_drops,
                # MediaPipe always receives the complete decoded frame.  The
                # old ROI state machine was removed; keep the field for API
                # compatibility so status polling never touches a stale
                # backend attribute while tracking is active.
                "face_crop": None,
            },
            "scheduling": getattr(self, "_scheduling", {"platform": "unknown", "priority": False, "affinity": False, "cores": []}),
            "inference_ms": {
                "mean": float(np.mean(inference)) if inference else 0.0,
                "p95": float(np.percentile(inference, 95.0)) if inference else 0.0,
            },
            "performance": self._latency.summary(),
            "shared_memory": {
                "name": self.config.shared_memory_name,
                "active": self.is_tracking(),
            },
            "motion_diagnostics": self._motion_diagnostics.status(),
            "error": error,
        }

    def close(self) -> None:
        self.stop_tracking()
        self._latency.close()
        self._motion_diagnostics.stop()
        self._camera.release()
