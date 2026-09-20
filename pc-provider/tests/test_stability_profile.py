import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from opengazelink_pc.config import ProviderConfig
from opengazelink_pc.event_temporal import EventTemporalFilter
from opengazelink_pc.stability_profile import estimate_profile, publish_profile, read_profile, validate_profile


def observations():
    rng = np.random.default_rng(712)
    rows, raw, times = [], [], []
    for split, count in (("train", 3), ("validation", 2), ("test", 2)):
        for trial in range(count):
            for start in (0, 1200, 2700):
                for t in range(start + 720, start + 1160, 20):
                    rows.append(dict(trial_id=split + str(trial), split=split, weight=1., trial_age_ms=t))
                    raw.append(rng.normal(0, 4, 2) + [900, 550])
                    times.append(trial * 5000 + t)
    return np.asarray(raw), np.asarray(times, float), np.full(len(rows), 40.), rows


class StabilityEstimationTests(unittest.TestCase):
    def test_training_only_estimation_does_not_read_held_out_points(self):
        raw, times, scales, rows = observations()
        expected = estimate_profile(raw, times, scales, rows)
        self.assertIsNotNone(expected)
        self.assertEqual((expected["training_trials"], expected["training_fixations"]), (3, 9))
        self.assertEqual(expected["settle_ms"], 40.)
        for i, row in enumerate(rows):
            if row["split"] != "train":
                raw[i], times[i], scales[i] = np.nan, np.nan, np.nan
        self.assertEqual(estimate_profile(raw, times, scales, rows), expected)

    def test_insufficient_or_discontinuous_training_abstains(self):
        raw, times, scales, rows = observations()
        self.assertIsNone(estimate_profile(raw, times, scales, [dict(row, split="test") for row in rows]))
        self.assertIsNone(estimate_profile(raw, times * 10, scales, rows))

    def test_validation_gates_jitter_error_and_false_prediction_independently(self):
        raw, times, scales, rows = observations()
        profile = estimate_profile(raw, times, scales, rows)
        target = np.tile([900., 550.], (len(rows), 1))
        states = [dict(prediction_active=False) for _ in rows]
        improved = target + (raw - target) * .5
        validate = lambda candidate, candidate_states=states: validate_profile(
            profile, raw, candidate, target, rows, states, candidate_states)
        self.assertTrue(validate(improved)["accepted"])
        self.assertFalse(validate(target + (raw - target) * 3)["accepted"])
        self.assertFalse(validate(improved + 100)["accepted"])
        false_predictions = [dict(prediction_active=True) for _ in rows]
        self.assertFalse(validate(improved, false_predictions)["accepted"])
        # Saccade gating can disrupt stability even with extrapolation disabled.
        false_saccades = [dict(prediction_active=False, mode='event_saccade_observed') for _ in rows]
        self.assertFalse(validate(improved, false_saccades)["accepted"])
        chatter = [dict(stability_active=bool(i % 2)) for i in range(len(rows))]
        self.assertFalse(validate(improved, chatter)["accepted"])
        # Test-set outputs cannot affect selection either.
        improved[[i for i, row in enumerate(rows) if row["split"] == "test"]] = np.nan
        self.assertTrue(validate(improved)["accepted"])

    def test_invalid_validation_output_is_rejected(self):
        raw, times, scales, rows = observations()
        profile = estimate_profile(raw, times, scales, rows)
        states = [{} for _ in rows]
        bad = raw.copy()
        bad[next(i for i, row in enumerate(rows) if row["split"] == "validation")] = np.inf
        self.assertEqual(validate_profile(profile, raw, bad, raw, rows, states, states)["reason"],
                         "invalid_validation_output")


class StabilityPublicationTests(unittest.TestCase):
    def test_roundtrip_requires_same_model_camera_and_geometry(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path, model_path = root / "profile.json", root / "model.json"
            metadata = {"variants": {"conditioned_video": {"module_sha256": "model-a"}}}
            model_path.write_text(json.dumps(metadata), encoding="utf-8")
            parameters = dict(calibrated=True, noise_prior_deg=.2, noise_floor_deg=.1, settle_ms=24.)
            report = dict(training_directory=temporary, model_sha256="model-a",
                          stability_calibration=dict(accepted=True, parameters=parameters))
            config = ProviderConfig()
            camera = {"width": 1280, "height": 720, "sourceMetadata": {"cameraId": "0"}}
            self.assertTrue(publish_profile(report, config, camera, path=path, model_path=model_path))
            read = lambda cfg=config, cam=camera, meta=metadata: read_profile(cfg, cam, meta, path=path)
            self.assertEqual(read()["parameters"], parameters)
            config.event_temporal_enabled = False  # Prediction preference does not disable stability.
            self.assertIsNotNone(read())
            self.assertIsNone(read(cam=dict(camera, width=1920)))
            self.assertIsNone(read(cam=dict(camera, sourceMetadata={"cameraId": "1"})))
            self.assertIsNone(read(meta={"variants": {"conditioned_video": {"module_sha256": "model-b"}}}))
            config.screen_diagonal_inches += 1
            self.assertIsNone(read())
            self.assertFalse(path.with_suffix(".json.tmp").exists())

    def test_rejected_or_changed_model_does_not_replace_existing_profile(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path, model_path = root / "profile.json", root / "model.json"
            path.write_text("previous", encoding="utf-8")
            model_path.write_text(json.dumps({"variants": {"conditioned_video": {"module_sha256": "new"}}}), encoding="utf-8")
            report = dict(model_sha256="old", stability_calibration=dict(accepted=True, parameters={"calibrated": True}))
            self.assertFalse(publish_profile(report, ProviderConfig(), {}, path=path, model_path=model_path))
            self.assertFalse(publish_profile({}, ProviderConfig(), {}, path=path, model_path=model_path))
            self.assertEqual(path.read_text(encoding="utf-8"), "previous")

    def test_missing_corrupt_and_nonfinite_profiles_are_ignored(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "profile.json"
            self.assertIsNone(read_profile(ProviderConfig(), {}, path=path))
            for contents in ("not json", "null", "[]", '{"parameters": {"noise_prior_deg": NaN}}'):
                path.write_text(contents, encoding="utf-8")
                self.assertIsNone(read_profile(ProviderConfig(), {}, {}, path=path))


class AutomaticStabilityTests(unittest.TestCase):
    def test_prediction_off_keeps_fixation_stability(self):
        predictor = EventTemporalFilter()
        rng = np.random.default_rng(32)
        raw = rng.normal(0, 3, (300, 2)) + [900, 550]
        points, stable_frames = [], 0
        for i, point in enumerate(raw):
            output, state = predictor.update(point, i * 1000 / 120, (1920, 1080),
                                            pixels_per_degree=40., horizon_ms=0.)
            self.assertFalse(state["prediction_active"])
            stable_frames += state["stability_active"]
            points.append(output)
        self.assertGreater(stable_frames, len(raw) * .95)
        self.assertLess(np.std(points[100:], axis=0).mean(), np.std(raw[100:], axis=0).mean() * .6)

    def test_movement_and_head_guard_bypass_stability_and_reacquire(self):
        predictor = EventTemporalFilter()
        moving = 0
        for t in np.arange(0., 600., 8.):
            raw = (700 + 700 * (1 - np.exp(-(max(0., t - 200) / 32) ** 3)), 500.)
            output, state = predictor.update(raw, t, (1920, 1080), pixels_per_degree=40., horizon_ms=0.)
            if state["mode"] != "event_fixation":
                moving += 1
                self.assertFalse(state["stability_active"])
                np.testing.assert_array_equal(output, raw)
        self.assertGreater(moving, 0)
        self.assertTrue(state["stability_active"])
        for i in range(20):
            angle = i * .02
            rotation = np.array([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]])
            output, state = predictor.update((1000., 500.), 700 + i * 8., (1920, 1080),
                                            pixels_per_degree=40., head_rotation=rotation)
            if i:
                self.assertFalse(state["stability_active"])
                self.assertEqual(output, (1000., 500.))

    def test_profile_survives_resets_and_rejects_nonfinite_values(self):
        predictor = EventTemporalFilter()
        predictor.set_stability_profile(dict(calibrated=True, noise_prior_deg=.3, noise_floor_deg=.1, settle_ms=32))
        predictor.reset()
        _, state = predictor.update((700, 500), 0, (1920, 1080), pixels_per_degree=40.)
        self.assertTrue(state["stability_calibrated"])
        self.assertEqual(state["noise_deg"], .3)
        for field in ("noise_prior_deg", "noise_floor_deg", "settle_ms"):
            for value in (np.nan, np.inf, -np.inf):
                with self.assertRaises(ValueError):
                    predictor.set_stability_profile({field: value})


if __name__ == "__main__":
    unittest.main()
