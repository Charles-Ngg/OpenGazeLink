import json
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from opengazelink_pc.video_archive import RawVideoArchive
from opengazelink_pc.video_replay import prepare_replay


class FakeVideoBackend:
    lock = threading.Lock()
    active = 0
    peak = 0
    closed = 0

    def __init__(self, *args, **kwargs):
        self.last = -1
        self.last_diagnostics = {}
        with self.lock:
            type(self).active += 1
            type(self).peak = max(self.peak, self.active)

    def predict(self, image, source_ms, camera):
        assert source_ms > self.last, "Each VIDEO instance must receive chronological frames"
        self.last = source_ms
        time.sleep(.002)
        return None

    def close(self):
        with self.lock:
            type(self).active -= 1
            type(self).closed += 1


class ParallelReplayTests(unittest.TestCase):
    def make_capture(self, root, fps=100):
        root.joinpath("session.json").write_text(json.dumps({"camera_model": {}, "plan": []}))
        root.joinpath("stimulus.jsonl").write_text("\n".join(json.dumps(dict(
            pc_ms=1000+i*200, capture_segment=i+1, trial_id=f"train-{i}")) for i in range(3)))
        archive = RawVideoArchive(root / "raw-camera")
        image = np.zeros((8, 8, 3), np.uint8)
        for i in range(60):
            archive.submit("windows_bgr", dict(pc_capture_ms=1000+i*1000/fps, source_ms=1000+i*1000/fps,
                           sequence=i, dtype="uint8", shape=list(image.shape)), image.tobytes())
        archive.close()

    def test_parallel_sequences_preserve_every_frame_and_merge_in_order(self):
        with tempfile.TemporaryDirectory() as folder, patch(
                "opengazelink_pc.video_replay.NormalizedEyeBackend", FakeVideoBackend):
            root = Path(folder)
            self.make_capture(root)
            FakeVideoBackend.peak = 0
            progress = []
            parallel = prepare_replay(root, workers=3, progress=progress.append)
            rows = [json.loads(line) for line in Path(parallel["manifest"]).read_text().splitlines()]
            self.assertGreater(FakeVideoBackend.peak, 1)
            self.assertEqual(0, FakeVideoBackend.active)
            self.assertEqual(list(range(60)), [r["sequence"] for r in rows])
            self.assertEqual(list(range(60)), [r["index"] for r in rows])
            self.assertEqual(60, parallel["archived_frames"])
            self.assertEqual(3, parallel["segments"])
            self.assertEqual(3, progress[-1]["completed"])
            serial = prepare_replay(root, workers=1, progress=lambda _: None)
            self.assertEqual(rows, [json.loads(line) for line in Path(serial["manifest"]).read_text().splitlines()])

    def test_cancellation_keeps_previous_manifest_and_closes_backends(self):
        with tempfile.TemporaryDirectory() as folder, patch(
                "opengazelink_pc.video_replay.NormalizedEyeBackend", FakeVideoBackend):
            root = Path(folder)
            self.make_capture(root)
            manifest = root / "replay-frames.jsonl"
            manifest.write_text("previous-result")
            with self.assertRaisesRegex(RuntimeError, "cancelled"):
                prepare_replay(root, workers=3, cancelled=lambda: True, progress=lambda _: None)
            self.assertEqual("previous-result", manifest.read_text())
            self.assertEqual(0, FakeVideoBackend.active)

    def test_120hz_ceiling_keeps_real_frames_without_upsampling_slow_cameras(self):
        for fps in (30, 60, 120):
            with self.subTest(fps=fps), tempfile.TemporaryDirectory() as folder, patch(
                    "opengazelink_pc.video_replay.NormalizedEyeBackend", FakeVideoBackend):
                root = Path(folder)
                self.make_capture(root, fps)
                (root / 'session.json').write_text(json.dumps({'camera_model': {}, 'plan': [
                    {'calibration_stage': 'spatial_v1', 'sample_rate_hz': 120}]}))
                report = prepare_replay(root, workers=1, progress=lambda _: None)
                rows = [json.loads(line) for line in Path(report['manifest']).read_text().splitlines()]
                self.assertEqual(list(range(60)), [r['sequence'] for r in rows])
                self.assertAlmostEqual(report['processed_fps_estimate'], fps)
