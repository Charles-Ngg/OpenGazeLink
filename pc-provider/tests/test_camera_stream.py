from __future__ import annotations

import socket
import json
import time
import unittest
from unittest.mock import patch
import threading

import cv2
import numpy as np

from opengazelink_pc.camera import (
    FORMAT_JPEG, HEADER, INTRINSICS_HEADER, INTRINSICS_MAGIC, MAGIC,
    UdpYuvCamera, UdpYuvConfig,
)
from opengazelink_pc import runtime_clock


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

    def test_identical_intrinsics_heartbeats_do_not_rewrite_cache(self):
        message = {"streamIntrinsics": dict(width=1280, height=720, fx=900, fy=900, cx=640, cy=360)}
        payload = json.dumps(message).encode()
        packet = INTRINSICS_HEADER.pack(INTRINSICS_MAGIC, 1, INTRINSICS_HEADER.size, len(payload)) + payload
        with patch.object(self.camera, "_save_cached_intrinsics") as save:
            self.camera._handle_intrinsics(packet)
            self.camera._handle_intrinsics(packet)
            message["timestampNs"] = 123456789
            payload = json.dumps(message).encode()
            packet = INTRINSICS_HEADER.pack(INTRINSICS_MAGIC, 1, INTRINSICS_HEADER.size, len(payload)) + payload
            self.camera._handle_intrinsics(packet)
            self.assertEqual(1, save.call_count)
        self.camera.set_allowed_source_ip("0.0.0.0")
        self.assertEqual("", self.camera.source_status()["allowed_source_ip"])
        self.assertTrue(self.camera.source_status()["awaiting_paired_discovery"])

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

    def test_shared_snapshot_is_stable_readonly_and_public_copy_is_independent(self):
        self.send_frame(1, (20, 40, 80))
        ok, shared, timestamp, sequence = self.camera.read_latest_shared(timeout_s=1.)
        self.assertTrue(ok)
        self.assertFalse(shared.flags.writeable)
        original = shared.copy()
        with self.assertRaises(ValueError):
            shared[0, 0] = 0
        _, independent, _, _ = self.camera.read_latest()
        self.assertTrue(independent.flags.writeable)
        independent[:] = 0
        np.testing.assert_array_equal(shared, original)
        self.send_frame(2, (120, 200, 240))
        ok, newer, new_timestamp, new_sequence = self.camera.read_latest_shared(sequence, timeout_s=1.)
        self.assertTrue(ok)
        self.assertGreater(new_sequence, sequence)
        # This fixture uses legacy time.monotonic_ns (15.625ms on Python 3.8).
        self.assertGreaterEqual(new_timestamp, timestamp)
        self.assertIsNot(shared, newer)
        np.testing.assert_array_equal(shared, original)

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
        self.assertLessEqual(timing["pc_first_packet_monotonic_ns"], timing["pc_last_packet_monotonic_ns"])
        self.assertLessEqual(timing["pc_last_packet_monotonic_ns"], timing["pc_decode_start_monotonic_ns"])
        self.assertLessEqual(timing["pc_decode_start_monotonic_ns"], timing["pc_decode_done_monotonic_ns"])

    def test_waiting_for_next_frame_is_not_charged_to_receive(self) -> None:
        self.send_frame(1, (0, 0, 0))
        ok, _, _, sequence = self.camera.read_latest(timeout_s=1.0)
        self.assertTrue(ok)
        time.sleep(0.03)  # Receiver is blocked awaiting the next frame.
        before_send = runtime_clock.monotonic_ns()
        self.send_frame(2, (0, 0, 0))
        ok, _, _, sequence = self.camera.read_latest(sequence, timeout_s=1.0)
        self.assertTrue(ok)
        self.assertGreaterEqual(
            self.camera.latest_frame_timing(sequence)["pc_first_packet_monotonic_ns"], before_send,
        )

    def test_slow_decode_recovers_to_latest_complete_frame(self) -> None:
        from opengazelink_pc.camera import decode_udp_frame
        entered, resume = threading.Event(), threading.Event()

        def slow_decode(*args):
            entered.set()
            resume.wait(2.0)
            return decode_udp_frame(*args)

        with patch("opengazelink_pc.camera.decode_udp_frame", slow_decode):
            try:
                self.send_frame(1, (255, 0, 0))
                self.assertTrue(entered.wait(1.0))
                for seq in range(2, 9):
                    self.send_frame(seq, (0, 255, 0))
                deadline = time.perf_counter() + 1.0
                while self.camera._last_frame_seq != 8 and time.perf_counter() < deadline:
                    time.sleep(0.001)
                self.assertEqual(self.camera._last_frame_seq, 8)
                resume.set()
                ok, _, _, sequence = self.camera.read_latest(timeout_s=1.0)
                self.assertTrue(ok)
                self.assertEqual(self.camera.latest_frame_timing(sequence)["phone_frame_sequence"], 8)
            finally:
                resume.set()

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

    def test_clock_reply_reaches_subsequent_frame_timing(self) -> None:
        from opengazelink_pc.transport_clock import CLOCK_MAGIC, CLOCK_PACKET
        self.send_frame(1, (0, 0, 0))
        self.sender.settimeout(1)
        packet, address = self.sender.recvfrom(100)
        magic, version, kind, t1, _, _ = CLOCK_PACKET.unpack(packet)
        self.assertEqual((magic, version, kind), (CLOCK_MAGIC, 1, 1))
        t2 = runtime_clock.monotonic_ns()
        self.sender.sendto(CLOCK_PACKET.pack(magic, 1, 2, t1, t2, runtime_clock.monotonic_ns()), address)
        deadline = time.perf_counter() + 1
        while not self.camera._transport_clock.samples and time.perf_counter() < deadline:
            time.sleep(.001)
        self.assertTrue(self.camera._transport_clock.samples)
        _, _, _, previous = self.camera.read_latest(timeout_s=1)
        self.send_frame(2, (0, 0, 0))
        ok, _, _, seq = self.camera.read_latest(previous, timeout_s=1)
        self.assertTrue(ok)
        self.assertIn('clock_probe_rtt_ms', self.camera.latest_frame_timing(seq))

    def test_manual_rotation_overrides_phone_hint(self) -> None:
        self.camera.config.rotate = "180"
        self.camera._packet_rotation = 90
        self.assertEqual(self.camera._effective_rotation(), 180)


if __name__ == "__main__":
    unittest.main()
