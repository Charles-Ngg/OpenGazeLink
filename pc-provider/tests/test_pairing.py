import json
import socket
import tempfile
import unittest
from pathlib import Path

from opengazelink_pc.pairing import DISCOVERY_MAGIC, PairingService


class PairingServiceTest(unittest.TestCase):
    def test_discovery_requires_pairing_then_accepts_same_phone(self) -> None:
        paired = ["", ""]
        sources: list[str] = []
        with tempfile.TemporaryDirectory() as temporary:
            service = PairingService(
                "127.0.0.1", 0, lambda: 5007,
                lambda: (paired[0], paired[1]), sources.append,
                Path(temporary) / "instance.json",
            )
            try:
                client = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                client.settimeout(2.0)
                request = {
                    "magic": DISCOVERY_MAGIC,
                    "type": "discover",
                    "version": 1,
                    "nonce": "test-nonce",
                    "phone_id": "phone-1",
                    "phone_name": "Test phone",
                    "instance_id": "",
                }
                client.sendto(json.dumps(request).encode(), ("127.0.0.1", service.port))
                response = json.loads(client.recvfrom(4096)[0])
                self.assertFalse(response["accepted"])
                self.assertEqual(response["data_port"], 5007)
                self.assertIsNotNone(service.pending_phone("phone-1"))

                paired[:] = ["phone-1", "Test phone"]
                request["instance_id"] = service.instance_id
                client.sendto(json.dumps(request).encode(), ("127.0.0.1", service.port))
                response = json.loads(client.recvfrom(4096)[0])
                self.assertTrue(response["accepted"])
                self.assertEqual(sources, ["127.0.0.1"])
            finally:
                client.close()
                service.close()


if __name__ == "__main__":
    unittest.main()
