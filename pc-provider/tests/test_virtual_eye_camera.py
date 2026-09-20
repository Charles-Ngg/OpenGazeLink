from __future__ import annotations

import unittest

import cv2
import numpy as np

from opengazelink_pc.shared_eye_appearance import _map_points
from opengazelink_pc.virtual_eye_camera import (
    direction_angles, restore_head_angles, virtual_eye_transform, virtual_gaze_label,
)


class VirtualEyeCameraTests(unittest.TestCase):
    def setUp(self):
        self.camera = {"fx": 700, "fy": 700, "cx": 320, "cy": 240}
        self.corners = np.asarray([[300, 240], [340, 240]], dtype=np.float64)
        self.origin = np.asarray([0.0, 0.0, 60.0])
        self.rotation = np.diag([1.0, -1.0, -1.0])

    def test_observed_eye_midpoint_maps_to_image_centre(self):
        for right in (False, True):
            warp, rotation = virtual_eye_transform(self.corners, self.origin, self.rotation, self.camera, right)
            midpoint = _map_points(warp, np.asarray([self.corners.mean(0)]))[0]
            np.testing.assert_allclose(midpoint, [31.5, 17.5], atol=1e-9)
            np.testing.assert_allclose(rotation @ rotation.T, np.eye(3), atol=1e-12)
            self.assertAlmostEqual(float(np.linalg.det(rotation)), 1.0)

    def test_gaze_round_trip_is_exact_across_poses_and_both_eyes(self):
        for yaw in np.linspace(-0.4, 0.4, 5):
            for pitch in (-0.2, 0.2):
                head = self.rotation @ cv2.Rodrigues(np.asarray([pitch, yaw, 0.05]))[0]
                for right in (False, True):
                    _, rotation = virtual_eye_transform(self.corners + [25, 15], self.origin, head, self.camera, right)
                    target = np.asarray([8.0, -12.0, 0.0])
                    label = virtual_gaze_label(target, self.origin, rotation, right)
                    actual = restore_head_angles(label, rotation, head, right)
                    expected = direction_angles(head.T @ (target - self.origin), right)
                    np.testing.assert_allclose(actual, expected, atol=1e-12)

    def test_distance_scaling_changes_magnification_without_scaling_gaze_angles(self):
        first_map, first_rotation = virtual_eye_transform(self.corners, self.origin, self.rotation, self.camera, True)
        second_map, second_rotation = virtual_eye_transform(self.corners, self.origin * 2, self.rotation, self.camera, True)
        first_points = _map_points(first_map, self.corners)
        second_points = _map_points(second_map, self.corners)
        self.assertAlmostEqual(float(np.linalg.norm(second_points[1] - second_points[0]) / np.linalg.norm(first_points[1] - first_points[0])), 2.0)
        np.testing.assert_allclose(first_rotation, second_rotation)
        a = virtual_gaze_label([10, -10, 0], self.origin, first_rotation, True)
        b = virtual_gaze_label([20, -20, 0], self.origin * 2, second_rotation, True)
        np.testing.assert_allclose(a, b, atol=1e-12)

    def test_left_reflection_changes_image_x_and_yaw_together(self):
        right_map, rotation = virtual_eye_transform(self.corners, self.origin, self.rotation, self.camera, True)
        left_map, _ = virtual_eye_transform(self.corners, self.origin, self.rotation, self.camera, False)
        right = _map_points(right_map, self.corners)
        left = _map_points(left_map, self.corners)
        np.testing.assert_allclose(left[:, 0], 63 - right[:, 0], atol=1e-12)
        np.testing.assert_allclose(left[:, 1], right[:, 1], atol=1e-12)
        a = virtual_gaze_label([10, 2, 0], self.origin, rotation, True)
        b = virtual_gaze_label([10, 2, 0], self.origin, rotation, False)
        np.testing.assert_allclose(b, a * [-1, 1])


if __name__ == "__main__":
    unittest.main()
