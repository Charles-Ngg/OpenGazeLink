from __future__ import annotations

import unittest

from opengazelink_pc.config import ProviderConfig


class ProviderConfigTest(unittest.TestCase):
    def test_motion_diagnostics_is_disabled_by_default(self) -> None:
        self.assertFalse(ProviderConfig().motion_diagnostics_enabled)

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
