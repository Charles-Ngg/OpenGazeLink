"""Completion telemetry must survive the display/camera boundary mismatch."""
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

from opengazelink_pc.unified_capture import coverage_report
from opengazelink_pc.video_dataset import align_frames
from opengazelink_pc.video_session import VideoSession


def capture(*, next_trial=True, camera_offset=0):
    step = dict(trial_id="train-line", split="train", motion_profile="line",
                duration=7200, calibration_stage="spatial_v1")
    events = []
    for ms in range(0, 7201, 10):
        progress = min(1., max(0., (ms - 1100) / 5000))
        moving = 1100 <= ms < 6100
        events.append(dict(pc_ms=ms, x=.05 + .9 * progress, y=.5,
                           phase="pursuit" if moving else "anchor",
                           block=step["trial_id"], trial_id=step["trial_id"], split="train",
                           capture_segment=1, visible=True, sync_rtt_ms=1,
                           drag_progress=progress, trial_complete=ms == 7200,
                           motion_age_ms=ms-1100 if moving else 0,
                           rail=dict(a=[.05, .5], b=[.95, .5]) if moving else None))
    if next_trial:
        events.append(dict(events[-1], pc_ms=7210, capture_segment=2,
                           trial_id="test-next", block="test-next", phase="pause",
                           trial_complete=False))
    else:
        # The last line completes then immediately pauses, often between frames.
        events.append(dict(events[-1], pc_ms=7200.5, phase="pause"))
    frames = [dict(index=i, source_ms=ms, pc_read_ms=ms, valid=True, input="raw-camera",
                   timing={"source_capture_monotonic_ns": ms * 1e6})
              for i, ms in enumerate(range(5 + camera_offset, 7220, 8))]
    return step, frames, events


class SpatialCompletionTests(unittest.TestCase):
    def test_terminal_marker_does_not_need_a_supported_camera_frame(self):
        for next_trial in (True, False):
            for offset in range(8):
                with self.subTest(next_trial=next_trial, offset=offset):
                    step, frames, events = capture(next_trial=next_trial, camera_offset=offset)
                    rows = align_frames(frames, events)
                    # This reproduces the old all-retry result without weakening
                    # cross-segment alignment or synthesizing coordinate labels.
                    if next_trial or offset != 3:
                        self.assertFalse(coverage_report(rows, [step])["ready"])
                    self.assertTrue(coverage_report(rows, [step], events=events)["ready"])

    def test_completion_cannot_be_borrowed_from_another_attempt_or_trial(self):
        step, frames, events = capture()
        rows = align_frames(frames, events)
        terminal = events[-2]
        for changes in (dict(capture_segment=2), dict(trial_id="test-other"),
                        dict(trial_complete=False), dict(visible=False),
                        dict(phase="pause"), dict(sync_rtt_ms=41),
                        dict(pc_ms=8000), dict(pc_ms=-1)):
            with self.subTest(changes=changes):
                self.assertFalse(coverage_report(rows, [step],
                                 events=[dict(terminal, **changes)])["ready"])

    def test_completion_does_not_replace_missing_observations(self):
        step, frames, events = capture()
        rows = align_frames(frames, events)
        variants = [[],
                    [dict(r, valid=False) for r in rows],
                    [dict(r, constraint=None) for r in rows],
                    [dict(r, weight=0) for r in rows],
                    [dict(r, drag_progress=.5) for r in rows],
                    align_frames(frames, events, discarded_segments=[1])]
        for i, variant in enumerate(variants):
            with self.subTest(variant=i):
                self.assertFalse(coverage_report(variant, [step], events=events)["ready"])

    def test_review_and_finish_accept_boundary_completion(self):
        step, frames, events = capture(next_trial=False)
        with tempfile.TemporaryDirectory() as folder:
            session = VideoSession.__new__(VideoSession)
            session.path = Path(folder)
            for name, records in (("frames", frames), ("stimulus", events)):
                (session.path / (name + ".jsonl")).write_text(
                    "".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
            session._lock = threading.RLock()
            session.plan = [step]
            session.state = "paused"
            session.purpose = "unified"
            session.error = ""
            session.discarded_segments = set()
            with patch.object(session, "_persist"), patch.object(session, "_stop_capture"), \
                 patch.object(session, "status", return_value={}), \
                 patch("opengazelink_pc.video_session.threading.Thread") as worker:
                self.assertTrue(session.review()["ready"])
                session.finish()
                worker.return_value.start.assert_called_once()
                self.assertEqual("training", session.state)
                self.assertEqual(set(), session.discarded_segments)


if __name__ == "__main__":
    unittest.main()
