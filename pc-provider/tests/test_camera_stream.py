from __future__ import annotations

import socket
import json
import time
import unittest

import cv2
import numpy as np

from opengazelink_pc.camera import (
    FORMAT_JPEG, HEADER, INTRINSICS_HEADER, INTRINSICS_MAGIC, MAGIC,
    UdpYuvCamera, UdpYuvConfig,
)


class CameraStreamLifecycleTest(unittest.TestCase):
    def setUp(self) -> None:
        self.camera = UdpYuvCamera(UdpYuvConfig(
            bind="127.0.0.1",
            port=0,
            rotate=0,
            frame_stale_after_s=0.2,
            sequence_reset_after_s=0.1,
            intrinsics_cache_path="",
        ))
        self.port = self.camera._socket.getsockname()[1]
        self.sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def tearDown(self) -> None:
        self.camera.release()
        self.sender.close()

    def send_frame(self, sequence: int, bgr: tuple[int, int, int]) -> None:
        frame = np.full((48, 64, 3), bgr, dtype=np.uint8)
        encoded, payload = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        self.assertTrue(encoded)
        data = payload.tobytes()
        now = time.monotonic_ns()
        packet = HEADER.pack(
            MAGIC, 1, HEADER.size, sequence, 0, 1, 64, 48,
            FORMAT_JPEG, 0, now, now, len(data),
        ) + data
        self.sender.sendto(packet, ("127.0.0.1", self.port))

    def test_lower_sequence_takes_over_after_sender_restart(self) -> None:
        self.send_frame(20_000, (255, 0, 0))
        ok, _, _, local_sequence = self.camera.read_latest(timeout_s=1.0)
        self.assertTrue(ok)

        self.send_frame(1, (0, 255, 0))
        ignored, _, _, _ = self.camera.read_latest(local_sequence, timeout_s=0.05)
        self.assertFalse(ignored)

        time.sleep(0.1)
        self.send_frame(2, (0, 255, 0))
        ok, frame, _, _ = self.camera.read_latest(local_sequence, timeout_s=1.0)
        self.assertTrue(ok)
        self.assertIsNotNone(frame)
        assert frame is not None
        self.assertGreater(float(frame[:, :, 1].mean()), float(frame[:, :, 0].mean()) + 100.0)
        self.assertEqual(self.camera.reported_mode()["sequenceResets"], 1)

    def test_last_frame_expires_when_sender_stops(self) -> None:
        self.send_frame(1, (255, 0, 0))
        ok, _, _, local_sequence = self.camera.read_latest(timeout_s=1.0)
        self.assertTrue(ok)
        time.sleep(0.25)

        mode = self.camera.reported_mode()
        self.assertEqual(mode["width"], 0)
        self.assertEqual(mode["height"], 0)
        self.assertEqual(mode["fps"], 0.0)
        ok, _, _, _ = self.camera.read_latest(local_sequence, timeout_s=0.05)
        self.assertFalse(ok)

    def test_latest_frame_exposes_phone_and_pc_timing(self) -> None:
        self.send_frame(37, (255, 0, 0))
        ok, _, source_time_ms, local_sequence = self.camera.read_latest(timeout_s=1.0)
        self.assertTrue(ok)
        timing = self.camera.latest_frame_timing(local_sequence)
        self.assertEqual(37, timing["phone_frame_sequence"])
        self.assertGreater(timing["phone_sensor_time_ns"], 0)
        self.assertAlmostEqual(source_time_ms, timing["phone_sensor_time_ns"] / 1_000_000.0)
        self.assertGreaterEqual(
            timing["pc_decode_done_monotonic_ns"],
            timing["pc_first_packet_monotonic_ns"],
        )

    def test_auto_rotation_uses_phone_intrinsics_hint(self) -> None:
        self.camera.config.rotate = "auto"
        message = {
            "source": "camera2_factory_calibration",
            "frameRotation": 90,
            "lensFacing": "front",
            "streamIntrinsics": {
                "width": 64, "height": 48,
                "fx": 50.0, "fy": 60.0, "cx": 20.0, "cy": 15.0,
            },
        }
        raw = json.dumps(message).encode("utf-8")
        packet = INTRINSICS_HEADER.pack(
            INTRINSICS_MAGIC, 1, INTRINSICS_HEADER.size, len(raw),
        ) + raw
        self.sender.sendto(packet, ("127.0.0.1", self.port))
        self.send_frame(1, (255, 0, 0))
        ok, frame, _, _ = self.camera.read_latest(timeout_s=1.0)
        self.assertTrue(ok)
        self.assertEqual(frame.shape[:2], (64, 48))
        model = self.camera.camera_model()
        self.assertEqual(model["rotate"], 90)
        self.assertEqual((model["width"], model["height"]), (48, 64))
        self.assertEqual(model["sourceMetadata"]["lensFacing"], "front")

    def test_manual_rotation_overrides_phone_hint(self) -> None:
        self.camera.config.rotate = "180"
        self.camera._packet_rotation = 90
        self.assertEqual(self.camera._effective_rotation(), 180)


if __name__ == "__main__":
    unittest.main()
