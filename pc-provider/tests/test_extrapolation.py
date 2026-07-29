from __future__ import annotations

import unittest

from opengazelink_pc.extrapolation import FixedHorizonExtrapolator2D


class FixedHorizonExtrapolatorTest(unittest.TestCase):
    SCREEN = (1920, 1080)

    def test_constant_motion_leads_the_stable_point(self) -> None:
        predictor = FixedHorizonExtrapolator2D(85.0, 0.5)
        diagnostics = {}
        for index in range(6):
            point = (100.0 + index * 12.0, 200.0)
            result, diagnostics = predictor.update(
                point, point, index * 30.0, self.SCREEN,
            )
        self.assertEqual("continuous_motion", diagnostics["mode"])
        self.assertGreater(result[0], point[0] + 20.0)
        self.assertAlmostEqual(point[1], result[1])

    def test_fixation_deadzone_does_not_amplify_tiny_motion(self) -> None:
        predictor = FixedHorizonExtrapolator2D(85.0, 0.5)
        predictor.update((100.0, 200.0), (100.0, 200.0), 0.0, self.SCREEN)
        result, diagnostics = predictor.update(
            (100.1, 200.0), (100.05, 200.0), 33.0, self.SCREEN,
        )
        self.assertEqual((100.05, 200.0), result)
        self.assertEqual("fixation", diagnostics["mode"])
        self.assertEqual(0.0, diagnostics["lead_distance_px"])

    def test_alternating_measurement_noise_does_not_become_motion(self) -> None:
        predictor = FixedHorizonExtrapolator2D(85.0, 0.5)
        diagnostics = {}
        result = (0.0, 0.0)
        for index in range(8):
            point = (100.0 + (6.0 if index % 2 else -6.0), 200.0)
            result, diagnostics = predictor.update(
                point, point, index * 33.0, self.SCREEN,
            )
        self.assertEqual("fixation", diagnostics["mode"])
        self.assertEqual(point, result)
        self.assertEqual(0.0, diagnostics["lead_distance_px"])

    def test_large_jump_bypasses_filter_and_uses_short_saccade_lead(self) -> None:
        predictor = FixedHorizonExtrapolator2D(85.0, 0.5)
        predictor.update((100.0, 200.0), (100.0, 200.0), 0.0, self.SCREEN)
        result, diagnostics = predictor.update(
            (600.0, 200.0), (140.0, 200.0), 33.0, self.SCREEN,
        )
        self.assertEqual("jump_or_landing", diagnostics["mode"])
        self.assertEqual("rising", diagnostics["phase"])
        self.assertTrue(diagnostics["filter_bypassed"])
        self.assertGreater(result[0], 600.0)
        self.assertLess(result[0], 1000.0)
        self.assertGreater(diagnostics["lead_distance_px"], 0.0)

    def test_large_reversal_is_a_new_jump_not_a_landing(self) -> None:
        predictor = FixedHorizonExtrapolator2D(85.0, 0.5)
        for index, point in enumerate(((124.0, 200.0), (112.0, 200.0), (100.0, 200.0))):
            predictor.update(point, point, index * 33.0, self.SCREEN)
        result, diagnostics = predictor.update(
            (600.0, 200.0), (130.0, 200.0), 99.0, self.SCREEN,
        )
        self.assertEqual("rising", diagnostics["phase"])
        self.assertGreater(result[0], 600.0)

    def test_prediction_is_clamped_to_screen_edges(self) -> None:
        predictor = FixedHorizonExtrapolator2D(85.0, 0.5)
        predictor.update((1800.0, 1000.0), (1800.0, 1000.0), 0.0, self.SCREEN)
        result, diagnostics = predictor.update(
            (2400.0, 1300.0), (1850.0, 1020.0), 33.0, self.SCREEN,
        )
        self.assertEqual((1919.0, 1079.0), result)
        self.assertTrue(diagnostics["screen_clamped"])

    def test_jump_landing_stops_lead_and_requests_filter_reset(self) -> None:
        predictor = FixedHorizonExtrapolator2D(85.0, 0.5)
        predictor.update((100.0, 200.0), (100.0, 200.0), 0.0, self.SCREEN)
        predictor.update((600.0, 200.0), (140.0, 200.0), 33.0, self.SCREEN)
        result, diagnostics = predictor.update(
            (604.0, 201.0), (250.0, 200.0), 66.0, self.SCREEN,
        )
        self.assertEqual((604.0, 201.0), result)
        self.assertEqual("jump_or_landing", diagnostics["mode"])
        self.assertEqual("landing", diagnostics["phase"])
        self.assertEqual(0.0, diagnostics["lead_distance_px"])
        self.assertTrue(diagnostics["reset_filter"])

    def test_non_monotonic_source_timestamp_resets_history(self) -> None:
        predictor = FixedHorizonExtrapolator2D(85.0, 0.5)
        predictor.update((100.0, 200.0), (100.0, 200.0), 100.0, self.SCREEN)
        predictor.update((150.0, 200.0), (150.0, 200.0), 133.0, self.SCREEN)
        result, diagnostics = predictor.update(
            (160.0, 210.0), (158.0, 208.0), 10.0, self.SCREEN,
        )
        self.assertEqual((158.0, 208.0), result)
        self.assertEqual(1, diagnostics["sample_count"])


if __name__ == "__main__":
    unittest.main()
