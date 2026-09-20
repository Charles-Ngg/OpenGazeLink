import unittest
from unittest.mock import patch

import numpy as np

from opengazelink_pc.normalized_eye import NormalizedEyeBackend


class FullFrameLandmarkerTest(unittest.TestCase):
    def test_backend_always_passes_complete_frame_to_mediapipe(self):
        seen = []

        class Landmarker:
            def detect_bgr(self, frame, timestamp_ms):
                seen.append((frame, timestamp_ms))
                return type("Result", (), {"face_landmarks": []})()

            def close(self):
                pass

        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        with patch(
            "opengazelink_pc.normalized_eye.create_normalized_eye_landmarker",
            return_value=Landmarker(),
        ):
            backend = NormalizedEyeBackend("tasks", conditioned=True)
            backend.record_diagnostics = True
            self.assertIsNone(backend.predict(frame, 123.0, {}))
            self.assertIs(seen[0][0], frame)
            self.assertEqual([720, 1280], backend.last_diagnostics["processing_shape"])
            self.assertIsNone(backend.last_diagnostics["processing_crop"])
            self.assertFalse(backend.state_discontinuity)


if __name__ == "__main__":
    unittest.main()
