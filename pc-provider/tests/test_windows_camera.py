from __future__ import annotations

import threading
import time
import unittest
from unittest.mock import patch

import numpy as np

from opengazelink_pc.camera import WindowsCamera, WindowsCameraConfig


class FakeCapture:
    def __init__(self, frames: list[np.ndarray], opened: bool = True) -> None:
        self.frames = list(frames)
        self.opened = opened
        self.released = False
        self.properties: dict[int, float] = {}
        self.lock = threading.Lock()

    def isOpened(self) -> bool:
        return self.opened and not self.released

    def set(self, prop: int, value: float) -> bool:
        self.properties[prop] = value
        return True

    def get(self, prop: int) -> float:
        return self.properties.get(prop, 0.0)

    def read(self):
        with self.lock:
            if self.released:
                return False, None
            if self.frames:
                frame = self.frames.pop(0)
                time.sleep(0.005)
                return True, frame.copy()
        time.sleep(0.005)
        return False, None

    def release(self) -> None:
        self.released = True


class WindowsCameraTest(unittest.TestCase):
    def test_capture_slot_returns_only_the_newest_frame(self) -> None:
        frames = [
            np.full((24, 32, 3), value, dtype=np.uint8)
            for value in (20, 80, 180)
        ]
        capture = FakeCapture(frames)
        with patch("opengazelink_pc.camera.cv2.VideoCapture", return_value=capture):
            camera = WindowsCamera(WindowsCameraConfig(
                width=32, height=24, fps=30, rotate=90,
                frame_stale_after_s=1.0,
            ))
            try:
                deadline = time.monotonic() + 1.0
                while camera.reported_mode()["capturedFrames"] < 3 and time.monotonic() < deadline:
                    time.sleep(0.01)
                ok, frame, timestamp_ms, sequence = camera.read_latest(timeout_s=0.2)
                self.assertTrue(ok)
                self.assertIsNotNone(frame)
                assert frame is not None
                self.assertEqual(frame.shape[:2], (32, 24))
                self.assertAlmostEqual(float(frame.mean()), 180.0)
                self.assertGreater(timestamp_ms, 0.0)
                self.assertEqual(sequence, 3)

                # Earlier frames were replaced, not left waiting in a queue.
                ok, _, _, returned_sequence = camera.read_latest(sequence, timeout_s=0.03)
                self.assertFalse(ok)
                self.assertEqual(returned_sequence, sequence)
                mode = camera.reported_mode()
                self.assertEqual(mode["source"], "windows_camera")
                self.assertEqual((mode["width"], mode["height"]), (24, 32))
                self.assertEqual(mode["buffered"], 1)
                self.assertGreaterEqual(mode["dropFrames"], 2)
                model = camera.camera_model()
                self.assertEqual(model["source"], "estimated_windows_camera")
                self.assertEqual((model["width"], model["height"]), (24, 32))
            finally:
                camera.release()

    def test_unavailable_device_reports_error_without_blocking(self) -> None:
        captures: list[FakeCapture] = []

        def unavailable(*_args):
            capture = FakeCapture([], opened=False)
            captures.append(capture)
            return capture

        with patch("opengazelink_pc.camera.cv2.VideoCapture", side_effect=unavailable):
            started = time.monotonic()
            camera = WindowsCamera(WindowsCameraConfig(device_index=7))
            try:
                ok, frame, _, _ = camera.read_latest(timeout_s=0.03)
                self.assertFalse(ok)
                self.assertIsNone(frame)
                self.assertLess(time.monotonic() - started, 0.5)
                mode = camera.reported_mode()
                self.assertEqual(mode["source"], "windows_camera")
                self.assertIn("could not be opened", mode["error"])
                self.assertEqual(mode["width"], 0)
            finally:
                camera.release()
        self.assertTrue(captures)
        self.assertTrue(all(capture.released for capture in captures))


if __name__ == "__main__":
    unittest.main()
