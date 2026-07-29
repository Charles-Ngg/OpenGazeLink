from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np

from opengazelink_pc.calibration_session import (
    STATIC_SAMPLES_PER_TARGET,
    _commit_artifacts,
    calibration_targets,
    light_anchor_targets,
    pose_targets,
)
from opengazelink_pc.normalized_eye import (
    SCREEN_CAMERA_MOUNT,
    screen_camera_origin,
    screen_point_to_pixels,
    target_camera_point,
)
from opengazelink_pc.shared_eye_appearance import CNN_MODEL_HEIGHT, CNN_MODEL_WIDTH
from opengazelink_pc.shared_eye_models import (
    CNN_ACTIVATION_NEGATIVE_SLOPE,
    CNN_AUGMENTATION_MODES,
    CNN_VALIDATION_SEEDS,
    PreparedSharedData,
    _balanced_source_weights,
    _cnn_gain_summary,
    _tiny_cnn_class,
    cnn_validation_grid_indices,
    require_torch,
)


class CalibrationPlanTest(unittest.TestCase):
    def test_bottom_bezel_screen_origin_is_used_for_new_geometry(self) -> None:
        origin = screen_camera_origin(3840, 2160, 27.0)
        self.assertEqual(SCREEN_CAMERA_MOUNT, "configured_screen_camera_position_v2")
        self.assertAlmostEqual(0.0, float(origin[0]), places=6)
        self.assertAlmostEqual(0.0, float(origin[2]), places=6)
        self.assertAlmostEqual(-16.81, float(origin[1]), places=2)
        target = target_camera_point((1919.5, 1079.5), 3840, 2160, 27.0)
        np.testing.assert_allclose(target, origin, atol=1e-6)
        pixels = screen_point_to_pixels(target, 3840, 2160, 27.0, origin)
        self.assertAlmostEqual(1919.5, pixels[0], places=5)
        self.assertAlmostEqual(1079.5, pixels[1], places=5)

    def test_configured_camera_position_uses_user_facing_axes(self) -> None:
        origin = screen_camera_origin(
            3840, 2160, 27.0,
            camera_position_screen_cm=(2.0, 18.0, 1.5),
        )
        np.testing.assert_allclose(origin, (2.0, -18.0, -1.5), atol=1e-9)
        expected = (2879.25, 539.75)
        target = target_camera_point(
            expected, 3840, 2160, 27.0, origin,
        )
        actual = screen_point_to_pixels(target, 3840, 2160, 27.0, origin)
        np.testing.assert_allclose(actual, expected, atol=1e-6)

    def test_screen_mount_round_trips_edges_and_centre(self) -> None:
        origin = screen_camera_origin(3840, 2160, 27.0)
        for expected in (
            (0.0, 0.0), (3839.0, 0.0), (1919.5, 1079.5),
            (0.0, 2159.0), (3839.0, 2159.0),
        ):
            point = target_camera_point(expected, 3840, 2160, 27.0)
            actual = screen_point_to_pixels(point, 3840, 2160, 27.0, origin)
            np.testing.assert_allclose(actual, expected, atol=1e-6)

    def test_static_plan_is_balanced_five_by_five(self) -> None:
        targets = calibration_targets(3840, 2160)
        self.assertEqual(25, len(targets))
        self.assertEqual(5, len({target["x"] for target in targets}))
        self.assertEqual(5, len({target["y"] for target in targets}))
        self.assertEqual(18.0, min(target["x"] for target in targets))
        self.assertEqual(3821.0, max(target["x"] for target in targets))
        self.assertEqual(18.0, min(target["y"] for target in targets))
        self.assertEqual(2141.0, max(target["y"] for target in targets))
        profiles = [target["lighting"] for target in targets]
        self.assertTrue(all(profile["mode"] == "steady" for profile in profiles))
        self.assertEqual({"reference-mid"}, {profile["name"] for profile in profiles})
        self.assertEqual({0.42}, {profile["start"] for profile in profiles})

    def test_light_anchors_repeat_the_same_five_targets(self) -> None:
        targets = light_anchor_targets(3840, 2160)
        self.assertEqual(10, len(targets))
        dark = [target for target in targets if target["light_name"] == "dark"]
        bright = [target for target in targets if target["light_name"] == "bright"]
        self.assertEqual([12, 0, 4, 24, 20], [target["grid_index"] for target in dark])
        self.assertEqual(
            [target["grid_index"] for target in dark],
            [target["grid_index"] for target in bright],
        )
        self.assertEqual({0.22}, {target["lighting"]["start"] for target in dark})
        self.assertEqual({0.68}, {target["lighting"]["start"] for target in bright})
        self.assertTrue(all(target["lighting"]["mode"] == "steady" for target in targets))

    def test_third_version_sampling_and_augmentation_mix(self) -> None:
        self.assertEqual(9, STATIC_SAMPLES_PER_TARGET)
        self.assertEqual(("geometry", "geometry"), CNN_AUGMENTATION_MODES)
        self.assertEqual((4022, 5022, 6022), CNN_VALIDATION_SEEDS)

    def test_cnn_gain_diagnostics_reject_constant_predictions(self) -> None:
        targets = np.asarray([
            [-0.4, -0.2], [-0.2, 0.1], [0.2, 0.3], [0.4, 0.5],
        ], dtype=np.float64)
        collapsed = _cnn_gain_summary(targets, np.zeros_like(targets))
        self.assertTrue(collapsed["collapsed"])
        healthy = _cnn_gain_summary(
            targets, targets * np.asarray([0.95, 1.05]) + np.asarray([0.02, -0.01]),
        )
        self.assertFalse(healthy["collapsed"])

    def test_tiny_cnn_keeps_gradients_on_negative_activations(self) -> None:
        torch, nn = require_torch()
        model = _tiny_cnn_class(torch, nn)()
        activations = [
            module for module in model.modules() if isinstance(module, nn.LeakyReLU)
        ]
        self.assertEqual(4, len(activations))
        self.assertTrue(all(
            module.negative_slope == CNN_ACTIVATION_NEGATIVE_SLOPE
            for module in activations
        ))
        self.assertFalse(any(isinstance(module, nn.ReLU) for module in model.modules()))
        output = model(torch.zeros(2, 2, CNN_MODEL_HEIGHT, CNN_MODEL_WIDTH))
        self.assertEqual((2, 2), tuple(output.shape))

    def test_validation_layout_tracks_dataset_grid(self) -> None:
        self.assertEqual((0, 5, 7, 9, 14), cnn_validation_grid_indices(("grid-00", "grid-14")))
        self.assertEqual((0, 6, 12, 18, 24), cnn_validation_grid_indices(("grid-00", "grid-24")))

    def test_pose_plan_crosses_four_head_poses_with_five_fixation_targets(self) -> None:
        targets = pose_targets(3840, 2160)
        self.assertEqual(20, len(targets))
        self.assertEqual(
            ["left", "right", "up", "down"],
            list(dict.fromkeys(target["pose_condition"] for target in targets)),
        )
        expected_gaze = ["center", "upper-left", "upper-right", "lower-right", "lower-left"]
        for condition in ("left", "right", "up", "down"):
            block = [target for target in targets if target["pose_condition"] == condition]
            self.assertEqual(expected_gaze, [target["gaze_name"] for target in block])
            self.assertEqual([12, 0, 4, 24, 20], [target["grid_index"] for target in block])
        self.assertEqual(["head_pose"] * 20, [target["phase"] for target in targets])
        self.assertEqual([0.42] * 20, [target["lighting"]["start"] for target in targets])

    def test_training_mass_is_eighty_twenty_despite_pose_frame_count(self) -> None:
        groups = ("grid-00", "grid-00", "grid-01", "grid-01", "pose-00", "pose-00", "pose-00", "pose-01")
        count = len(groups)
        data = PreparedSharedData(
            base_images=tuple([None] * count),
            gray=np.zeros((count, CNN_MODEL_HEIGHT, CNN_MODEL_WIDTH), dtype=np.float32),
            alpha=np.ones((count, CNN_MODEL_HEIGHT, CNN_MODEL_WIDTH), dtype=np.float32),
            targets=np.zeros((count, 2), dtype=np.float64),
            groups=groups, sides=tuple(["right"] * count),
            corner_targets=np.zeros((2, 2), dtype=np.float64),
        )
        weights = _balanced_source_weights(data, np.arange(count, dtype=np.int64))
        static_mass = float(np.sum(weights[:4]))
        pose_mass = float(np.sum(weights[4:]))
        self.assertAlmostEqual(0.80, static_mass / (static_mass + pose_mass), places=6)

    def test_artifact_commit_rolls_back_an_interrupted_replacement(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = root / "first.json"
            second = root / "second.json"
            first.write_text("old-first", encoding="utf-8")
            second.write_text("old-second", encoding="utf-8")
            staged_first = root / "staged-first.json"
            staged_first.write_text("new-first", encoding="utf-8")
            missing_second = root / "missing-second.json"
            with self.assertRaises(FileNotFoundError):
                _commit_artifacts(
                    [(staged_first, first), (missing_second, second)],
                    root / "backup",
                )
            self.assertEqual("old-first", first.read_text(encoding="utf-8"))
            self.assertEqual("old-second", second.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
