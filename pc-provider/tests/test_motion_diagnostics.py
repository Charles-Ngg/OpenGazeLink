from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from opengazelink_pc.motion_diagnostics import MotionDiagnosticsRecorder


class MotionDiagnosticsRecorderTest(unittest.TestCase):
    def test_background_recorder_writes_session_frames_and_summary(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            recorder = MotionDiagnosticsRecorder(Path(temporary))
            path = recorder.start({"target_horizon_ms": 80.0})
            recorder.record({
                "type": "frame",
                "phone_sensor_time_ns": 1_000_000,
                "valid": True,
                "raw_combined_px": [10.0, 20.0],
            })
            recorder.stop()
            records = [
                json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(["session", "frame", "summary"], [item["type"] for item in records])
            self.assertEqual(1, records[-1]["sample_count"])
            self.assertFalse(recorder.status()["active"])
            self.assertEqual(str(path), recorder.status()["path"])

    def test_start_is_idempotent_while_recording(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            recorder = MotionDiagnosticsRecorder(Path(temporary))
            first = recorder.start({})
            second = recorder.start({"ignored": True})
            self.assertEqual(first, second)
            recorder.stop()


if __name__ == "__main__":
    unittest.main()
