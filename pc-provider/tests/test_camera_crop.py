"""Focused crop metadata and rotation contract tests; no camera or network listener."""
import json
import threading
import unittest

from opengazelink_pc.camera import INTRINSICS_HEADER, INTRINSICS_MAGIC, UdpYuvCamera, UdpYuvConfig


class CameraCropTest(unittest.TestCase):
    def camera(self, message):
        camera = UdpYuvCamera.__new__(UdpYuvCamera)
        camera.config = UdpYuvConfig(rotate="auto", intrinsics_cache_path="")
        camera._lock = threading.Lock()
        camera._source_camera_model = {}
        camera._packet_rotation = 0
        camera._intrinsics_cache_error = ""
        camera._raw_width = camera._raw_height = 0
        payload = json.dumps(message).encode()
        camera._handle_intrinsics(INTRINSICS_HEADER.pack(
            INTRINSICS_MAGIC, 1, INTRINSICS_HEADER.size, len(payload)) + payload)
        return camera

    def test_asymmetric_crop_then_rotation_transforms_intrinsics_once(self):
        # Original stream 200x100, display insets L10/R20/T30/B40 percent.
        cases = {
            0: (20, 30, 140, 30, 40, 40, 103, 12, 160, 150),
            90: (60, 20, 60, 70, 80, 10, 47, 63, 150, 160),
            180: (40, 40, 140, 30, 20, 30, 56, 27, 160, 150),
            270: (80, 10, 60, 70, 60, 20, 32, 16, 150, 160),
        }
        for rotation, (left, top, w, h, right, bottom, cx, cy, fx, fy) in cases.items():
            with self.subTest(rotation=rotation):
                model = self.camera({
                    "frameRotation": rotation,
                    "softwareCrop": dict(sourceWidth=200, sourceHeight=100,
                        left=left, top=top, right=right, bottom=bottom, width=w, height=h),
                    "streamIntrinsics": dict(width=w, height=h, fx=160, fy=150, cx=123-left, cy=42-top),
                }).camera_model()
                self.assertEqual((model["cx"], model["cy"], model["fx"], model["fy"]), (cx, cy, fx, fy))
                self.assertEqual((model["width"], model["height"]), (w, h) if rotation % 180 == 0 else (h, w))

    def test_tight_crop_does_not_fail_full_sensor_sanity_check(self):
        model = self.camera({
            "frameRotation": 0,
            "softwareCrop": dict(sourceWidth=2000, sourceHeight=1000,
                left=900, top=450, right=900, bottom=450, width=200, height=100),
            "streamIntrinsics": dict(width=200, height=100, fx=3000, fy=3000, cx=-600, cy=-350),
        }).camera_model()
        self.assertEqual((model["fx"], model["cx"], model["cy"]), (3000, -600, -350))

    def test_inconsistent_crop_metadata_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "crop metadata"):
            self.camera({
                "softwareCrop": dict(sourceWidth=2000, sourceHeight=1000,
                    left=900, top=450, right=899, bottom=450, width=200, height=100),
                "streamIntrinsics": dict(width=200, height=100, fx=3000, fy=3000, cx=100, cy=50),
            })


if __name__ == "__main__":
    unittest.main()
