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


if __name__ == "__main__":
    unittest.main()
