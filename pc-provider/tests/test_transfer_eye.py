import unittest

import cv2
import numpy as np

from opengazelink_pc.transfer_eye import align_transfer_eye, transfer_eye_input, TRANSFER_CORNERS


class TransferEyeTests(unittest.TestCase):
    def setUp(self):
        self.frame = np.full((70, 110, 3), 170, np.uint8)
        self.corners = np.array([[80., 34.], [25., 32.]])
        self.contour = np.array([[25., 32.], [40., 23.], [65., 24.], [80., 34.], [65., 41.], [40., 40.]])
        cv2.circle(self.frame, (48, 29), 7, (25, 25, 25), -1)

    def test_left_reflection_preserves_vertical_texture(self):
        right = transfer_eye_input(self.frame, self.corners, self.contour, "right")
        mirrored = self.frame[:, ::-1].copy()
        corners, contour = self.corners.copy(), self.contour.copy()
        corners[:, 0] = self.frame.shape[1] - 1 - corners[:, 0]
        contour[:, 0] = self.frame.shape[1] - 1 - contour[:, 0]
        left = transfer_eye_input(mirrored, corners, contour, "left")
        np.testing.assert_allclose(left, right, atol=1e-6)

    def test_intensity_retains_pupil_movement_and_mask_support(self):
        first = transfer_eye_input(self.frame, self.corners, self.contour, "right")
        changed = np.full_like(self.frame, 170)
        cv2.circle(changed, (62, 29), 7, (25, 25, 25), -1)
        second = transfer_eye_input(changed, self.corners, self.contour, "right")
        np.testing.assert_array_equal(first[1], second[1])
        self.assertGreater(float(np.abs(first[0] - second[0]).sum()), 5.)
        self.assertTrue(np.all(first[0][first[1] == 0] == 0))

    def test_subimage_coordinates_produce_same_eye(self):
        whole = align_transfer_eye(self.frame, self.corners, self.contour, "right")
        origin = np.array([10., 10.])
        crop = align_transfer_eye(self.frame[10:60, 10:100], self.corners - origin, self.contour - origin, "right")
        # Texture inside the actual aperture is invariant to source-image origin.
        np.testing.assert_allclose(whole.gray_base * whole.alpha_base, crop.gray_base * crop.alpha_base, atol=1e-6)
        np.testing.assert_allclose(whole.alpha_base, crop.alpha_base, atol=1e-6)
        np.testing.assert_allclose(whole.mapped_inner, TRANSFER_CORNERS[0])

    def test_collapsed_and_nonfinite_landmarks_fail(self):
        with self.assertRaises(ValueError):
            align_transfer_eye(self.frame, [[3, 3], [3, 3]], self.contour, "right")
        with self.assertRaises(ValueError):
            align_transfer_eye(self.frame, self.corners * np.nan, self.contour, "right")


if __name__ == "__main__":
    unittest.main()
