import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from opengazelink_pc.app_host import ApplicationHost
from opengazelink_pc.config import ProviderConfig, load_config, save_config


class PairingRestartTests(unittest.TestCase):
    def test_accepted_address_survives_expired_discovery_save_and_restart(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "config.json"
            save_config(ProviderConfig(paired_phone_id="phone", paired_phone_name="Test"), path)
            with patch("opengazelink_pc.app_host.EyeTrackingEngine") as engine, \
                 patch("opengazelink_pc.app_host.PairingService") as pairing:
                pairing.return_value.pending_phone.return_value = None
                host = ApplicationHost(path)
                host._remember_paired_source("192.168.1.42")
                host.application.update_config({"camera_offset_x_cm": 1})
                engine.return_value.camera.set_allowed_source_ip.assert_called_with("192.168.1.42")
                self.assertEqual("192.168.1.42", load_config(path).paired_phone_address)
                restarted = ApplicationHost(path)
                engine.return_value.camera.set_allowed_source_ip.assert_called_with("192.168.1.42")
                restarted._remember_paired_source("192.168.1.43")
                self.assertEqual("192.168.1.43", load_config(path).paired_phone_address)
                restarted.forget_pairing()
                self.assertEqual("", load_config(path).paired_phone_address)
                engine.return_value.camera.set_allowed_source_ip.assert_called_with(None)

    def test_recreated_receiver_starts_with_saved_allowlist(self):
        from opengazelink_pc.engine import EyeTrackingEngine
        engine = EyeTrackingEngine.__new__(EyeTrackingEngine)
        engine.config = ProviderConfig(paired_phone_id="phone", paired_phone_address="10.0.0.2")
        with patch("opengazelink_pc.engine.UdpYuvCamera") as camera:
            engine._open_camera()
            self.assertEqual("10.0.0.2", camera.call_args.args[0].allowed_source_ip)


if __name__ == "__main__":
    unittest.main()
