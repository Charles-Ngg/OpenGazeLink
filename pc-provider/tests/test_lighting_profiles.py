from __future__ import annotations

import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest import mock

from opengazelink_pc import calibration_session as sessions
from opengazelink_pc.config import ProviderConfig


class LightingProfileTest(unittest.TestCase):
    def test_full_calibration_defers_old_samples_until_post_training_selection(self) -> None:
        class Backend:
            def __init__(self, name: str) -> None:
                self.name = name

            def close(self) -> None:
                return

        record = {
            "samples": {
                backend: [{
                    "condition": "lighting_anchor",
                    "lighting": {"name": "room"},
                    "landmarker_backend": backend,
                }] * 10
                for backend in ("legacy", "tasks")
            },
        }
        with tempfile.TemporaryDirectory() as temporary, \
             mock.patch.object(sessions, "ROOT", Path(temporary)), \
             mock.patch.object(sessions, "NormalizedEyeBackend", Backend), \
             mock.patch.object(sessions, "_load_lighting_profile_library", return_value={
                 "profiles": {"room": record},
             }), \
             mock.patch.object(sessions, "_library_record_matches", return_value=True):
            calibration = sessions.CalibrationSession(None, ProviderConfig(), object())
            try:
                self.assertEqual(["room"], [
                    item["name"] for item in calibration._candidate_status()
                ])
                self.assertTrue(all(
                    not dataset["samples"] for dataset in calibration.datasets.values()
                ))
            finally:
                calibration._close_backends()

    def test_delete_removes_active_samples_and_both_cnn_adapters(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset_paths = {
                backend: root / f"{backend}-dataset.json"
                for backend in ("legacy", "tasks")
            }
            model_paths = {}
            for backend in ("legacy", "tasks"):
                metadata = root / f"{backend}-cnn.json"
                module = root / f"{backend}-cnn.pt"
                metadata.write_text(json.dumps({
                    "lighting_profiles": {
                        "reference": {"gain": [], "bias": []},
                        "room": {"gain": [], "bias": []},
                    },
                    "diagnostics": {"lighting_adapters": {"room": {"error_deg": 1.0}}},
                }), encoding="utf-8")
                model_paths[backend] = (metadata, module)
            for backend, path in dataset_paths.items():
                path.write_text(json.dumps({
                    "samples": [
                        {"condition": "lighting_anchor", "lighting": {"name": "room"}},
                        {"condition": "lighting_anchor", "lighting": {"name": "keep"}},
                        {"condition": "fixed_head", "lighting": {"name": "reference-mid"}},
                    ],
                    "passes": [
                        {"condition": "lighting_adaptation", "profile_name": "room"},
                        {"condition": "lighting_adaptation", "profile_name": "keep"},
                    ],
                }), encoding="utf-8")

            with mock.patch.object(sessions, "ROOT", root), \
                 mock.patch.object(sessions, "DATASET_PATHS", dataset_paths), \
                 mock.patch.object(sessions, "MODEL_PATHS", model_paths), \
                 mock.patch.object(
                     sessions, "LIGHTING_PROFILE_LIBRARY_PATH", root / "lighting-library.json",
                 ):
                result = sessions.delete_lighting_profile("room")

            self.assertEqual({"legacy": 1, "tasks": 1}, result["removed_samples"])
            for path in dataset_paths.values():
                payload = json.loads(path.read_text(encoding="utf-8"))
                names = [sample.get("lighting", {}).get("name") for sample in payload["samples"]]
                self.assertNotIn("room", names)
                self.assertEqual(["keep"], [
                    item["profile_name"] for item in payload["passes"]
                ])
            for metadata, _ in model_paths.values():
                payload = json.loads(metadata.read_text(encoding="utf-8"))
                self.assertNotIn("room", payload["lighting_profiles"])
                self.assertNotIn("room", payload["diagnostics"]["lighting_adapters"])

    def test_selected_profiles_are_refit_after_base_training_before_commit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            staging = root / "capture" / "trained-artifacts"
            staging.mkdir(parents=True)
            dataset_paths = {
                backend: root / f"{backend}-dataset.json"
                for backend in ("legacy", "tasks")
            }
            model_paths = {}
            staged_artifacts = []
            for backend in ("legacy", "tasks"):
                cnn = root / f"{backend}-cnn.json"
                module = root / f"{backend}-cnn.pt"
                model_paths[backend] = (cnn, module)
                for destination in (dataset_paths[backend], cnn, module):
                    staged = staging / destination.name
                    staged.write_text("{}", encoding="utf-8")
                    staged_artifacts.append((staged, destination))

            calibration = sessions.CalibrationSession.__new__(sessions.CalibrationSession)
            calibration.config = ProviderConfig()
            calibration.registry = mock.Mock()
            calibration.datasets = {
                backend: {"samples": [], "passes": []}
                for backend in ("legacy", "tasks")
            }
            calibration._lighting_profile_candidates = {
                "room": {"samples": {
                    backend: [
                        {"condition": "lighting_anchor", "lighting": {"name": "room"}}
                    ] * 10
                    for backend in ("legacy", "tasks")
                }}
            }
            calibration._staging_dir = staging
            calibration._staged_artifacts = staged_artifacts
            calibration._diagnostics = {}
            calibration.capture_dir = root / "capture"
            calibration.pass_id = "test"
            calibration.phase = "lighting_profile_selection"
            calibration.state = "awaiting_profiles"
            calibration.error = ""
            calibration._lock = threading.Lock()

            with mock.patch.object(sessions, "DATASET_PATHS", dataset_paths), \
                 mock.patch.object(sessions, "MODEL_PATHS", model_paths), \
                 mock.patch.object(
                     sessions, "LIGHTING_PROFILE_LIBRARY_PATH", root / "lighting-library.json",
                 ), \
                 mock.patch.object(
                     sessions.SharedTinyCnnModel, "fit_lighting_profile",
                     return_value=(None, {"median_deg": 1.0}),
                 ) as fit_profile, \
                 mock.patch.object(sessions.SharedTinyCnnModel, "load"):
                result = calibration.complete_lighting_profiles(["room"])

            self.assertEqual(["room"], result["retained_lighting_profiles"])
            self.assertEqual(2, fit_profile.call_count)
            for path in dataset_paths.values():
                dataset = json.loads(path.read_text(encoding="utf-8"))
                self.assertEqual(10, len(dataset["samples"]))
                self.assertEqual("room", dataset["passes"][0]["profile_name"])
            library = json.loads((root / "lighting-library.json").read_text(encoding="utf-8"))
            self.assertEqual(["room"], list(library["profiles"]))

    def test_archived_profile_can_be_refit_on_the_current_base(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset_paths = {
                backend: root / f"{backend}-dataset.json"
                for backend in ("legacy", "tasks")
            }
            model_paths = {}
            for backend, dataset_path in dataset_paths.items():
                dataset_path.write_text(json.dumps({"samples": [], "passes": []}), encoding="utf-8")
                metadata = root / f"{backend}-cnn.json"
                module = root / f"{backend}-cnn.pt"
                metadata.write_text("{}", encoding="utf-8")
                module.write_bytes(b"module")
                model_paths[backend] = (metadata, module)
            record = {"samples": {
                backend: [
                    {"condition": "lighting_anchor", "lighting": {"name": "room"}}
                ] * 10
                for backend in ("legacy", "tasks")
            }}
            with mock.patch.object(sessions, "ROOT", root), \
                 mock.patch.object(sessions, "DATASET_PATHS", dataset_paths), \
                 mock.patch.object(sessions, "MODEL_PATHS", model_paths), \
                 mock.patch.object(
                     sessions, "_load_lighting_profile_library",
                     return_value={"profiles": {"room": record}},
                 ), \
                 mock.patch.object(sessions, "_library_record_matches", return_value=True), \
                 mock.patch.object(sessions, "_dataset_geometry_matches", return_value=True), \
                 mock.patch.object(
                     sessions.SharedTinyCnnModel, "fit_lighting_profile",
                     return_value=(None, {"median_deg": 1.0}),
                 ) as fit_profile, \
                 mock.patch.object(sessions.SharedTinyCnnModel, "load"):
                result = sessions.retrain_lighting_profiles(
                    ProviderConfig(), ["room"],
                )
            self.assertEqual(["room"], result["profile_names"])
            self.assertEqual(2, fit_profile.call_count)
            for dataset_path in dataset_paths.values():
                dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
                self.assertEqual(10, len(dataset["samples"]))

    def test_reference_profile_cannot_be_deleted(self) -> None:
        with self.assertRaisesRegex(ValueError, "reference"):
            sessions.delete_lighting_profile("reference")


if __name__ == "__main__":
    unittest.main()
