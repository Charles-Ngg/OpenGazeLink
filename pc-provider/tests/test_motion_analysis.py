from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from opengazelink_pc.motion_analysis import analyze_motion_recording


class MotionAnalysisTest(unittest.TestCase):
    def test_two_point_velocity_beats_hold_on_constant_motion(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "motion.jsonl"
            records = [{
                "type": "session",
                "metadata": {"screen": {"width": 1920, "height": 1080}},
            }]
            for index in range(30):
                point = [100.0 * index, 200.0]
                records.append({
                    "type": "frame",
                    "phone_sensor_time_ns": index * 33_000_000,
                    "valid": True,
                    "raw_combined_px": point,
                    "filtered_combined_px": point,
                    "output_combined_px": point,
                    "postprocess": {"extrapolation_state": {"mode": "continuous_motion"}},
                })
            path.write_text(
                "\n".join(json.dumps(item) for item in records) + "\n",
                encoding="utf-8",
            )
            result = analyze_motion_recording(path, horizon_ms=66.0)
            json.dumps(result)
            hold = result["methods"]["raw_hold"]["all"]["mean_px"]
            velocity = result["methods"]["velocity_2point"]["all"]["mean_px"]
            self.assertLess(velocity, hold * 0.05)
            self.assertGreater(result["saccade_candidate_frames"], 10)
            self.assertIn("learned_endpoint_ridge", result["methods"])
            self.assertIn("rising_only_gain20ms", result["methods"])
            self.assertIn("confidence_signals", result)

    def test_reports_complete_rising_episode_against_settled_endpoint(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "episode.jsonl"
            phases = ["steady", "steady", "rising", "rising", "landing", "settled", "steady"]
            xs = [100.0, 100.0, 300.0, 700.0, 900.0, 1000.0, 1000.0]
            records = [{
                "type": "session",
                "metadata": {"screen": {"width": 1920, "height": 1080}},
            }]
            for index in range(14):
                phase = phases[index] if index < len(phases) else "steady"
                x = xs[index] if index < len(xs) else 1000.0
                point = [x, 500.0]
                records.append({
                    "type": "frame",
                    "phone_sensor_time_ns": index * 33_000_000,
                    "valid": True,
                    "right_px": point,
                    "left_px": point,
                    "right_angles_rad": [x / 1000.0, 0.0],
                    "left_angles_rad": [x / 1000.0, 0.0],
                    "head_rotation_rad": [0.0, 0.0, 0.0],
                    "head_translation_cm": [0.0, 0.0, 50.0],
                    "raw_combined_px": point,
                    "filtered_combined_px": point,
                    "output_combined_px": point,
                    "postprocess": {"extrapolation_state": {
                        "mode": "jump_or_landing" if phase in {"rising", "landing"} else "fixation",
                        "phase": phase,
                    }},
                })
            path.write_text(
                "\n".join(json.dumps(item) for item in records) + "\n",
                encoding="utf-8",
            )
            result = analyze_motion_recording(path, horizon_ms=80.0)
            endpoint = result["settled_endpoint_analysis"]
            self.assertEqual(endpoint["complete_episode_count"], 1)
            self.assertEqual(endpoint["rising_frames_with_endpoint"], 2)
            self.assertEqual(endpoint["episodes"][0]["endpoint_px"], [1000.0, 500.0])
            json.dumps(result)


if __name__ == "__main__":
    unittest.main()
