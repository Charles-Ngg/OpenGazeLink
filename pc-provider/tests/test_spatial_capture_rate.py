"""Spatial completion must not depend on camera/display sampling phase."""
import unittest

from opengazelink_pc.unified_capture import coverage_report
from opengazelink_pc.video_dataset import align_frames


def capture(fps=30, offset=0., hold=1000, uncertainty=0.):
    step = dict(trial_id='train-line', split='train', motion_profile='line',
                duration=5200, calibration_stage='spatial_v1')
    finish = hold * 2 + 4200
    events = []
    for ms in range(0, finish + 1, 10):
        moving = hold <= ms < hold + 4200
        p = min(1., max(0., (ms - hold) / 4200))
        events.append(dict(pc_ms=ms, x=.01 + .98*p, y=.5, phase='pursuit' if moving else 'anchor',
                           block='train-line', trial_id='train-line', split='train',
                           capture_segment=1, visible=True, sync_rtt_ms=uncertainty*2,
                           drag_progress=p, trial_complete=ms == finish,
                           motion_age_ms=ms-hold if moving else 0,
                           rail=dict(a=[.01, .5], b=[.99, .5]) if moving else None))
    events.append(dict(events[-1], pc_ms=finish+.5, phase='pause'))
    frames = []
    for i in range(int(finish * fps / 1000) + 1):
        ms = .1 + offset + i*1000/fps
        frames.append(dict(index=i, source_ms=ms, pc_read_ms=ms, valid=True, input='synthetic',
                           timing=dict(source_capture_monotonic_ns=ms*1e6)))
    return step, frames, events


class SpatialCaptureRateTests(unittest.TestCase):
    def test_updated_holds_across_camera_rates_phases_and_uncertainty(self):
        for fps in (30, 60, 120):
            for phase in range(12):
                for uncertainty in (0., 5., 20.):
                    with self.subTest(fps=fps, phase=phase, uncertainty=uncertainty):
                        step, frames, events = capture(fps, phase*1000/fps/12, uncertainty=uncertainty)
                        report = coverage_report(align_frames(frames, events), [step], events=events)
                        self.assertTrue(report['ready'], report)

    def test_existing_900ms_30fps_capture_with_ten_frames_is_usable(self):
        step, frames, events = capture(offset=5, hold=900, uncertainty=10)
        report = coverage_report(align_frames(frames, events), [step], events=events)
        self.assertEqual(10, report['trials'][0]['settled_frames'])
        self.assertTrue(report['ready'], report)

    def test_single_dropped_frame_does_not_force_new_capture(self):
        step, frames, events = capture()
        frames = [f for f in frames if f['index'] not in (25, 181)]
        self.assertTrue(coverage_report(align_frames(frames, events), [step], events=events)['ready'])

    def test_missing_endpoint_or_short_burst_cannot_pass(self):
        step, frames, events = capture()
        rows = align_frames(frames, events)
        for side in (0, 1):
            changed = [dict(r, weight=0.) if r['phase'] == 'anchor' and r['drag_progress'] == side
                       else r for r in rows]
            self.assertFalse(coverage_report(changed, [step], events=events)['ready'])
        # Plenty of frames do not substitute for observed time at an endpoint.
        changed = [dict(r) for r in rows]
        for side in (0, 1):
            local = [r for r in changed if r['weight'] > .5 and r['drag_progress'] == side]
            base = local[0]['pc_ms']
            for i, row in enumerate(local):
                row['pc_ms'] = base + i*.1
        self.assertFalse(coverage_report(changed, [step], events=events)['ready'])

    def test_gaps_and_clock_resets_are_not_stable_observation_time(self):
        step, frames, events = capture()
        rows = align_frames(frames, events)
        for mode in ('gaps', 'reset', 'clock_epoch'):
            changed = []
            for r in rows:
                if r['weight'] > .5:
                    if mode == 'gaps' and r['index'] % 3:
                        continue
                    if mode == 'reset':
                        r = dict(r, reset=True)
                    if mode == 'clock_epoch':
                        r = dict(r, clock_epoch=r['index'])
                changed.append(r)
            with self.subTest(mode=mode):
                self.assertFalse(coverage_report(changed, [step], events=events)['ready'])


if __name__ == '__main__':
    unittest.main()
