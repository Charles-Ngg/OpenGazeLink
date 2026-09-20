from __future__ import annotations

import unittest
import json
from pathlib import Path
import tempfile

from opengazelink_pc.config import ProviderConfig, load_config


class ProviderConfigTest(unittest.TestCase):
    def test_motion_diagnostics_is_disabled_by_default(self) -> None:
        self.assertFalse(ProviderConfig().motion_diagnostics_enabled)

    def test_old_models_and_processing_preferences_migrate_to_the_current_pipeline(self) -> None:
        config = ProviderConfig()
        for model in ("calibrated", "conditioned_without_iris", "unknown", "conditioned_video"):
            config.update({"gaze_model": model, "landmarker": "legacy", "lighting_profile": "dark",
                           "one_euro_enabled": False, "extrapolation_enabled": True,
                           "video_forecast_enabled": True, "event_temporal_enabled": False,
                           "prediction_auto_horizon_enabled": False})
            self.assertEqual("conditioned_video", config.gaze_model)
            self.assertEqual("tasks", config.landmarker)
            self.assertEqual("reference", config.lighting_profile)
            self.assertTrue(config.one_euro_enabled)
            self.assertTrue(config.prediction_auto_horizon_enabled)
            self.assertFalse(config.video_forecast_enabled)
            self.assertFalse(config.extrapolation_enabled)
            self.assertFalse(config.event_temporal_enabled)

    def test_fresh_install_uses_pretrained_pipeline(self) -> None:
        config = ProviderConfig()
        self.assertEqual(("tasks", "conditioned_video"), (config.landmarker, config.gaze_model))
        self.assertTrue(config.event_temporal_enabled)
        self.assertFalse(config.video_forecast_enabled)

    def test_load_migrates_preferences_without_overwriting_saved_file(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "config.json"
            saved = json.dumps({"gaze_model": "calibrated", "landmarker": "legacy",
                                "lighting_profile": "dark", "one_euro_enabled": False,
                                "event_temporal_enabled": False, "udp_port": 5012})
            path.write_text(saved, encoding="utf-8")
            config = load_config(path)
            self.assertEqual((config.gaze_model, config.landmarker), ("conditioned_video", "tasks"))
            self.assertEqual(config.lighting_profile, "reference")
            self.assertTrue(config.one_euro_enabled)
            self.assertFalse(config.event_temporal_enabled)
            self.assertEqual(config.udp_port, 5012)
            self.assertEqual(path.read_text(encoding="utf-8"), saved)

    def test_udp_port_rejects_zero_instead_of_silently_becoming_one(self) -> None:
        config = ProviderConfig()
        with self.assertRaisesRegex(ValueError, "UDP port"):
            config.update({"udp_port": 0})
        self.assertEqual(5007, config.udp_port)

    def test_udp_port_accepts_valid_value(self) -> None:
        config = ProviderConfig()
        config.update({"udp_port": 5008})
        self.assertEqual(5008, config.udp_port)

    def test_extrapolation_settings_are_clamped(self) -> None:
        config = ProviderConfig()
        config.update({
            "extrapolation_horizon_ms": 999,
            "extrapolation_max_lead_fraction": 0.9,
        })
        self.assertEqual(300.0, config.extrapolation_horizon_ms)
        self.assertEqual(0.5, config.extrapolation_max_lead_fraction)

    def test_windows_camera_configuration_is_validated(self) -> None:
        config = ProviderConfig()
        config.update({
            "input_source": "windows_camera",
            "windows_camera_index": 2,
            "windows_camera_width": 1920,
            "windows_camera_height": 1080,
            "windows_camera_fps": 60,
            "windows_camera_backend": "dshow",
            "windows_camera_fov_x_degrees": 72.5,
        })
        self.assertEqual("windows_camera", config.input_source)
        self.assertEqual(2, config.windows_camera_index)
        self.assertEqual((1920, 1080, 60), (
            config.windows_camera_width,
            config.windows_camera_height,
            config.windows_camera_fps,
        ))
        self.assertEqual("dshow", config.windows_camera_backend)
        self.assertEqual(72.5, config.windows_camera_fov_x_degrees)

    def test_invalid_windows_camera_values_fall_back_or_clamp(self) -> None:
        config = ProviderConfig()
        config.update({
            "input_source": "unknown",
            "windows_camera_index": -3,
            "windows_camera_width": 10,
            "windows_camera_height": 10,
            "windows_camera_fps": 999,
            "windows_camera_backend": "unknown",
            "windows_camera_fov_x_degrees": 200.0,
        })
        self.assertEqual("phone_udp", config.input_source)
        self.assertEqual(0, config.windows_camera_index)
        self.assertEqual((320, 240, 120), (
            config.windows_camera_width,
            config.windows_camera_height,
            config.windows_camera_fps,
        ))
        self.assertEqual("auto", config.windows_camera_backend)
        self.assertEqual(140.0, config.windows_camera_fov_x_degrees)


if __name__ == "__main__":
    unittest.main()
