"""Top/bottom diagnostics and training/runtime projection parity."""
import unittest

import cv2
import numpy as np
import torch

from opengazelink_pc.normalized_eye import (
    angles_to_camera_direction, eye_in_head_angles, intersect_screen_plane,
    screen_camera_origin, screen_point_to_pixels, target_camera_point,
)
from opengazelink_pc.spatial_metrics import spatial_metrics
from opengazelink_pc.video_training import project
from opengazelink_pc.training_runtime import spatial_epoch_limit, training_batch_size, spatial_update_budget
from unittest.mock import patch
import os
import math


class SpatialBiasTests(unittest.TestCase):
    def test_capture_rate_does_not_multiply_training_update_budget(self):
        with patch.dict(os.environ, {}, clear=True):
            batch = training_batch_size()
        for frames in (1144, 2288, 4576):
            steps = math.ceil(frames / batch)
            epochs = spatial_epoch_limit(60, frames, batch)
            self.assertGreaterEqual(epochs * steps, 300)
            self.assertLess(epochs * steps, 300 + steps)
            self.assertEqual(spatial_epoch_limit(2, frames, batch), 2)

    def test_opposite_edge_biases_are_visible_without_rail_point_labels(self):
        target = np.array([[.5, .01], [.5, .99], [.5, .5], [.5, .01]])
        prediction = target + [[0, .08], [0, -.08], [0, 0], [0, .8]]
        rows = [dict(trial_id=str(i), constraint=None) for i in range(4)]
        rows[-1]['constraint'] = {'normal': [0, 1]}
        result = spatial_metrics(prediction, target, [1, 1, 1, .25], [1921, 1001], rows=rows)
        bands = result['by_screen_band']
        self.assertEqual(bands['top']['frames'], 1)
        self.assertAlmostEqual(bands['top']['signed_y_median_px'], 80)
        self.assertAlmostEqual(bands['bottom']['signed_y_median_px'], -80)
        self.assertAlmostEqual(bands['middle']['signed_y_median_px'], 0)
        self.assertEqual(bands['top']['trials'], 1)

    def test_explicit_experiment_can_increase_or_remove_update_budget(self):
        self.assertEqual(spatial_epoch_limit(120, 2200, 256, 540), 60)
        self.assertEqual(spatial_epoch_limit(120, 2200, 256, None), 120)
        self.assertEqual(spatial_epoch_limit(2, 2200, 256, 1080), 2)
        with self.assertRaises(ValueError):
            spatial_epoch_limit(120, 2200, 256, 0)

    def test_source_rate_selects_budget_without_upsampling_thirty_fps(self):
        for rate in (None, 0, 30, 60, float('nan'), float('inf'), 'unknown'):
            self.assertEqual(spatial_update_budget({'source_fps_estimate': rate, 'processing_rate_limit_hz': 120}), 300)
        self.assertEqual(spatial_update_budget({}), 300)
        budget = spatial_update_budget({'source_fps_estimate': 119.98, 'processed_fps_estimate': 60})
        self.assertEqual(budget, 2160)
        self.assertEqual(spatial_epoch_limit(120, 4478, 256, budget), 120)
        self.assertEqual(spatial_epoch_limit(120, 1144, 256, spatial_update_budget({'source_fps_estimate': 30})), 60)

    def test_invalid_or_missing_band_has_no_fabricated_bias(self):
        result = spatial_metrics([[.5, np.nan]], [[.5, .01]], [1], [1920, 1080])
        self.assertEqual(result['by_screen_band']['top']['invalid_predictions'], 1)
        self.assertIsNone(result['by_screen_band']['top']['signed_y_median_px'])
        self.assertIsNone(result['by_screen_band']['bottom']['signed_y_median_px'])
        self.assertEqual(result['by_screen_band']['bottom']['frames'], 0)

    def test_training_and_runtime_project_top_without_vertical_compression(self):
        wh = np.array([3840, 2160])
        origin = screen_camera_origin(*wh, 27, [0, 16.81, 0])
        size = 27 * 2.54 * wh / np.linalg.norm(wh)
        centers = np.array([[-3., -2., 50.], [3., -2., 50.]])
        for angle in (-.15, 0, .15):
            rotation = cv2.Rodrigues(np.array([angle, .08, 0.]))[0] @ np.diag([1., -1., -1.])
            for x in (18, 1920, 3821):
                for y in (18, 1080, 2141):
                    target = target_camera_point([x, y], *wh, 27, origin)
                    local = []
                    for side, center in enumerate(centers):
                        yaw, pitch = eye_in_head_angles(target, center, rotation)
                        camera_ray = angles_to_camera_direction(yaw, pitch, rotation)
                        point = intersect_screen_plane(center, camera_ray, origin)
                        np.testing.assert_allclose(screen_point_to_pixels(point, *wh, 27, origin), [x, y], atol=1e-8)
                        ray = rotation.T @ (target - center)
                        ray /= np.linalg.norm(ray)
                        if side:
                            ray[0] *= -1
                        local.append(ray)
                    xy = project(torch.tensor(np.array([local])), torch.tensor([[.3, .7]], dtype=torch.float64),
                                 torch.tensor(np.array([[rotation, rotation]])), torch.tensor(np.array([centers])),
                                 torch.tensor(origin), torch.tensor(size))
                    np.testing.assert_allclose(xy.numpy()[0] * (wh - 1), [x, y], atol=1e-8)


if __name__ == '__main__':
    unittest.main()
