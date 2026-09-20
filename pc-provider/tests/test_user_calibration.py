import unittest
import numpy as np
from opengazelink_pc.calibration_split import split_masks, selection_score


class UserCalibrationTests(unittest.TestCase):
    def test_test_set_selects_epochs_and_never_gets_training_gradients(self):
        rows = [dict(block=s+"-line") for s in ("train", "validation", "test")]
        train, select, independent = split_masks(rows, [dict(calibration_stage="spatial_v1")])
        np.testing.assert_array_equal(train, [True, True, False])
        np.testing.assert_array_equal(select, [False, False, True])
        self.assertFalse(independent.any())
        self.assertFalse((train & select).any())
        _, research_select, research_test = split_masks(rows, [])
        np.testing.assert_array_equal(research_select, [False, True, False])
        np.testing.assert_array_equal(research_test, [False, False, True])

    def test_epoch_score_uses_rail_interval_without_inventing_point_ground_truth(self):
        rows = [dict(target=[.7, .3], weight=.25, constraint=dict(
            normal=[0., 1.], tangent=[1., 0.], normal_target=.3, lower=.2, upper=.8))]
        self.assertEqual(0., selection_score(np.array([[.4, .3]]), rows, [1920, 1080]))
        self.assertGreater(selection_score(np.array([[.4, .5]]), rows, [1920, 1080]), 0)
        self.assertGreater(selection_score(np.array([[.9, .3]]), rows, [1920, 1080]), 0)
        self.assertEqual(float("inf"), selection_score(np.array([[np.nan, .3]]), rows, [1920, 1080]))
