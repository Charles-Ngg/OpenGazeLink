import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from tools.compact_video_sessions import compact, inventory


class VideoCompactionTests(unittest.TestCase):
    def test_compaction_preserves_processed_inputs_and_moves_landmarks_into_npz(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);session=root/"session";(session/"inputs").mkdir(parents=True)
            np.savez_compressed(session/"inputs/0000000.npz",images=np.zeros((2,2,36,64),np.uint8),
                                head=np.zeros((2,10)),points=np.zeros((2,2,52)),crop=np.zeros((2,8)),
                                rotation=np.zeros((2,3,3)),center=np.zeros((2,3)))
            row={"index":0,"input":"inputs/0000000.npz","camera_model":{"large":"copy"},
                 "prediction_timing":{"duplicate":True},"diagnostics":{"landmarks":[[[.1,.2,.3]]*478],"reason":"ok"}}
            (session/"frames.jsonl").write_text(json.dumps(row)+"\n",encoding="utf8")
            (session/"session.json").write_text(json.dumps({"raw_archive":"raw-camera/index.jsonl"}),encoding="utf8")
            (session/"stimulus-requests.jsonl").write_bytes(b"duplicate")
            (session/"raw-camera").mkdir();(session/"raw-camera/data.bin").write_bytes(bytes(1000))
            self.assertEqual(1000,inventory(root)["raw_bytes"])
            result=compact(root)
            self.assertGreaterEqual(result["removed_bytes"],1009)
            self.assertFalse((session/"raw-camera").exists())
            compacted=json.loads((session/"frames.jsonl").read_text())
            self.assertNotIn("landmarks",compacted["diagnostics"])
            self.assertNotIn("camera_model",compacted)
            self.assertEqual(0,compacted["mediapipe_landmarks_index"])
            with np.load(session/"mediapipe-landmarks.npz") as arrays:
                self.assertEqual((1,478,3),arrays["landmarks"].shape)
                self.assertEqual([0],arrays["frame_index"].tolist())


if __name__=="__main__":
    unittest.main()
