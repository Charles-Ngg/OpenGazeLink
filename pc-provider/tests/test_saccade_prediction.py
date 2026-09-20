import unittest

import numpy as np

from opengazelink_pc.event_temporal import EventTemporalFilter
from opengazelink_pc.saccade_prediction import braking_prediction, short_continuation
from opengazelink_pc.event_evaluation import future_reference, is_prediction


class SaccadeKinematicsTests(unittest.TestCase):
    def braking_arc(self, times):
        t = np.asarray(times) / 1000
        return np.c_[700 * t - .5 * 10000 * t ** 2, np.zeros(len(t))]

    def test_exposure_midpoint_is_corrected_before_extrapolation(self):
        times = [0., 8., 16., 24.]
        points = self.braking_arc(times)
        delta = short_continuation(times, points, 85., .04)
        # At 24 ms, instantaneous speed is 460 deg/s, not the preceding
        # exposure interval's average of 500 deg/s.
        expected = 460 * .008 - .5 * 10000 * .008 ** 2
        np.testing.assert_allclose(delta, [expected, 0.], atol=1e-10)

    def test_stopping_model_respects_requested_time_and_never_reverses(self):
        times = [0., 8., 16., 24.]
        points = self.braking_arc(times)
        short, endpoint, info = braking_prediction(times, points, 16., .04)
        long, long_endpoint, _ = braking_prediction(times, points, 150., .04)
        np.testing.assert_allclose(points[-1] + short, self.braking_arc([40.])[0], atol=1e-10)
        np.testing.assert_allclose(endpoint, [24.5, 0.], atol=1e-10)
        np.testing.assert_allclose(points[-1] + long, endpoint, atol=1e-10)
        np.testing.assert_allclose(long_endpoint, endpoint)
        self.assertAlmostEqual(info['remaining_ms'], 46.)
        self.assertGreater(long[0], short[0])

    def test_nonuniform_samples_and_rigid_coordinate_change(self):
        times = np.array([0., 6., 15., 24.])
        rotation = np.array([[.6, -.8], [.8, .6]])
        translation = np.array([37., -24.])
        points = self.braking_arc(times) @ rotation.T + translation
        delta, endpoint, info = braking_prediction(times + 5e9, points, 16., .04)
        expected = self.braking_arc([40.])[0] @ rotation.T + translation
        np.testing.assert_allclose(points[-1] + delta, expected, atol=1e-7)
        np.testing.assert_allclose(endpoint, np.array([24.5, 0.]) @ rotation.T + translation, atol=1e-7)

    def test_sparse_or_reversed_timestamps_do_not_support_a_forecast(self):
        for times in ([0., 33., 66., 99.], [0., 8., 8., 16.], [0., 16., 8., 24.]):
            points = self.braking_arc(times)
            self.assertIsNone(short_continuation(times, points, 85., .04))
            self.assertIsNone(braking_prediction(times, points, 85., .04))

    def test_direction_reversal_and_zero_horizon_abstain(self):
        times = [0., 8., 16., 24.]
        reversed_points = np.array([[0., 0.], [5., 0.], [8., 0.], [6., 0.]])
        for function in (short_continuation, braking_prediction):
            self.assertIsNone(function(times, reversed_points, 85., .04))
            self.assertIsNone(function(times, self.braking_arc(times), 0., .04))


class SaccadeRuntimeTests(unittest.TestCase):
    def test_fast_observed_jump_improves_future_error_over_hold(self):
        f = EventTemporalFilter()
        errors, hold = [], []
        def trajectory(t):
            return np.array([700 + 700 * (1 - np.exp(-(max(0., t - 200) / 32) ** 3)), 500.])
        for t in np.arange(0., 500., 1000 / 120):
            raw = trajectory(t)
            point, _ = f.update(raw, t, (1920, 1080), pixels_per_degree=40., horizon_ms=85., smooth=False)
            if 200 <= t <= 320:
                future = trajectory(t + 85)
                errors.append(np.linalg.norm(np.asarray(point) - future))
                hold.append(np.linalg.norm(raw - future))
        self.assertLess(np.mean(errors), np.mean(hold) * .9)

    def test_one_millisecond_does_not_jump_to_full_endpoint(self):
        filters = [EventTemporalFilter(), EventTemporalFilter()]
        tiny, normal = [], []
        for t in np.arange(0., 440., 8.):
            raw = (700 + 700 * (1 - np.exp(-(max(0., t - 200) / 32) ** 3)), 500.)
            for predictor, horizon, records in zip(filters, (1., 85.), (tiny, normal)):
                point, state = predictor.update(raw, t, (1920, 1080), pixels_per_degree=40.,
                                                 horizon_ms=horizon, smooth=False)
                records.append(np.linalg.norm(np.asarray(point) - raw))
        self.assertGreater(max(normal), 30.)
        self.assertLess(max(tiny), max(normal) * .25)

    def test_lead_cap_and_landing_recovery(self):
        f = EventTemporalFilter()
        maximum = .01 * np.hypot(1919, 1079)
        for t in np.arange(0., 600., 8.):
            raw = (400 + 1200 * (1 - np.exp(-(max(0., t - 200) / 35) ** 3)), 500.)
            point, state = f.update(raw, t, (1920, 1080), pixels_per_degree=40.,
                                    horizon_ms=85., max_lead_fraction=.01, smooth=False)
            self.assertLessEqual(np.linalg.norm(np.asarray(point) - raw), maximum + 1e-8)
        self.assertEqual(point, raw)
        self.assertFalse(state['prediction_active'])

    def test_isolated_head_impulse_suppresses_its_history(self):
        f = EventTemporalFilter()
        for t in np.arange(0., 360., 8.):
            a = .1 if t >= 208 else 0.
            rotation = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
            raw = (600 + max(0., min(100., t - 184)) * 6, 500.)
            _, state = f.update(raw, t, (1920, 1080), pixels_per_degree=40., horizon_ms=85.,
                                 head_rotation=rotation, smooth=False)
            if 208 <= t < 268:
                self.assertTrue(state['head_guard_active'])
                self.assertFalse(state['prediction_active'])

    def test_invalid_input_clears_motion_state(self):
        for kwargs in ({'pixels_per_degree': np.nan}, {'horizon_ms': np.inf},
                       {'max_lead_fraction': np.nan}, {'screen_size': (np.nan, 1080)}):
            f = EventTemporalFilter()
            for t in range(0, 120, 8):
                f.update((600 + t * 5, 500), t, (1920, 1080), pixels_per_degree=40, horizon_ms=85)
            options = dict(screen_size=(1920, 1080), pixels_per_degree=40., horizon_ms=85.)
            options.update(kwargs)
            with self.assertRaises(ValueError):
                f.update((800, 500), 120., **options)
            point, state = f.update((800, 500), 128., (1920, 1080), pixels_per_degree=40., horizon_ms=85.)
            self.assertEqual(point, (800., 500.))
            self.assertEqual(state['sample_count'], 1)

    def test_stationary_noise_has_no_predictive_displacement(self):
        for fps in (30, 60, 120, 240):
            rng = np.random.default_rng(9231)
            f = EventTemporalFilter()
            for i, raw in enumerate(rng.normal(0, 3, (fps * 2, 2)) + [900, 550]):
                point, state = f.update(raw, i * 1000 / fps, (1920, 1080), pixels_per_degree=40.,
                                        horizon_ms=85., smooth=False)
                np.testing.assert_array_equal(point, raw)
                self.assertFalse(state['prediction_active'])


class EventReferenceTests(unittest.TestCase):
    def test_future_reference_does_not_cross_trials_gaps_or_clock_restarts(self):
        points = np.c_[np.arange(10), np.arange(10)]
        times = [0, 8, 16, 24, 32, 40, 200, 208, 0, 8]
        segments = np.zeros(10)
        trials = ['a'] * 3 + ['b'] * 7
        ref = future_reference(points, times, segments, trials, 8.)
        for i in (2, 5, 7, 9):
            self.assertFalse(np.isfinite(ref[i]).any())
        for i in (0, 1, 3, 4, 6, 8):
            np.testing.assert_array_equal(ref[i], points[i+1])

    def test_continuation_and_legacy_endpoint_are_both_counted(self):
        self.assertTrue(is_prediction({'mode': 'event_saccade_velocity_prediction', 'prediction_active': True}))
        self.assertTrue(is_prediction({'mode': 'event_saccade_landing_prediction'}))
        self.assertFalse(is_prediction({'mode': 'event_saccade_observed', 'prediction_active': False}))


if __name__ == '__main__':
    unittest.main()
