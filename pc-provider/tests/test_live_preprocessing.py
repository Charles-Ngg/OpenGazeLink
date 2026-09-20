from dataclasses import replace
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from opengazelink_pc.conditioned_eye import build_eye_sampling_plan, runtime_eye_inputs, sample_eye_plan
from opengazelink_pc.normalized_eye import FaceGeometry, NormalizedEyeBackend
from tests.test_conditioned_eye import synthetic_scene


class LiveSamplingTests(unittest.TestCase):
    def test_plans_preserve_both_eyes_exactly_including_out_of_frame_masks(self):
        rng = np.random.RandomState(710)
        for shift in (0., -.5, .5):
            frame, landmarks, pose, camera = synthetic_scene()
            for point in landmarks:
                point.x += shift
            for side in ('right', 'left'):
                plan = build_eye_sampling_plan(frame, landmarks, pose, camera, side, perspective_only=True)
                for _ in range(3):
                    pixels = rng.randint(0, 256, frame.shape, dtype=np.uint8)
                    expected, old_patch = runtime_eye_inputs(pixels, landmarks, pose, camera, side)
                    actual, new_patch = sample_eye_plan(cv2.cvtColor(pixels, cv2.COLOR_BGR2GRAY), plan)
                    self.assertIsNone(actual['images'][0])
                    np.testing.assert_array_equal(actual['images'][1], expected['images'][1])
                    np.testing.assert_array_equal(actual['points'][1], expected['points'][1])
                    for name in ('head', 'crop', 'rotation', 'center'):
                        np.testing.assert_array_equal(actual[name], expected[name])
                    self.assertEqual(old_patch.eye_center_camera, new_patch.eye_center_camera)
                    self.assertEqual(old_patch.aperture_ratio, new_patch.aperture_ratio)
                self.assertFalse(plan.matrices[0].flags.writeable)
                self.assertFalse(plan.masks[0].flags.writeable)

    def test_cached_backend_uses_fresh_pixels_without_rebuilding_geometry(self):
        frame, lm, pose, camera = synthetic_scene()
        pose.update(yaw=0., pitch=0., roll=0.)
        geometry = FaceGeometry(100., frame.shape[:2], tuple(lm), pose, camera, 10.)
        backend = NormalizedEyeBackend.__new__(NormalizedEyeBackend)
        backend.conditioned, backend.landmarker_backend, backend.record_diagnostics = True, 'tasks', False
        cached = backend.prepare_live_geometry(frame, geometry)
        self.assertIsNone(geometry.conditioned_plans)
        self.assertIsNotNone(cached.conditioned_plans)
        with patch('opengazelink_pc.conditioned_eye.build_eye_sampling_plan', side_effect=AssertionError('rebuild')):
            first = backend.prepare_with_geometry(frame, 110., camera, cached)
            second = backend.prepare_with_geometry(255-frame, 120., camera, cached)
        self.assertEqual((110., 120.), (first.t_ms, second.t_ms))
        self.assertFalse(np.array_equal(first.conditioned_inputs[0]['images'][1][0],
                                       second.conditioned_inputs[0]['images'][1][0]))
        for side in range(2):
            self.assertIs(first.conditioned_inputs[side]['head'], second.conditioned_inputs[side]['head'])
        with self.assertRaisesRegex(ValueError, 'camera geometry changed'):
            backend.prepare_with_geometry(frame, 120., dict(camera, fx=600.), cached)
        with self.assertRaisesRegex(ValueError, 'camera geometry changed'):
            backend.prepare_with_geometry(frame[:100], 120., camera, cached)
        changed = replace(geometry, camera_model=dict(camera, fx=600.))
        changed_cached = backend.prepare_live_geometry(frame, changed)
        self.assertIsNot(changed_cached.conditioned_plans, cached.conditioned_plans)

    def test_legacy_backend_does_not_build_conditioned_plans(self):
        backend = NormalizedEyeBackend.__new__(NormalizedEyeBackend)
        backend.conditioned = False
        sentinel = object()
        self.assertIs(sentinel, backend.prepare_live_geometry(None, sentinel))


if __name__ == '__main__':
    unittest.main()
