"""End-to-end schedule/alignment regressions for variable-distance capture."""
import copy
import json
from pathlib import Path
import subprocess
import unittest

from opengazelink_pc.unified_capture import validate_plan, coverage_report
from opengazelink_pc.video_dataset import align_frames
from opengazelink_pc.stability_profile import fixation_windows
from opengazelink_pc.event_evaluation import target_windows


class EventCaptureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan = json.loads(subprocess.check_output(
            ['node', '-e', "console.log(JSON.stringify(require('./web/unified-plan').eventPlan(71)))"],
            cwd=Path(__file__).resolve().parents[1], text=True, encoding='utf-8'))

    def test_browser_plan_passes_backend_and_rejects_mislabeled_distance(self):
        self.assertEqual(len(validate_plan(self.plan)), 12)
        for alteration in ('points', 'changes', 'plan_version', 'direction_axis'):
            bad = copy.deepcopy(self.plan)
            bad[0][alteration] = {'points': [[.5,.5]]*3, 'changes': [500,800],
                                  'plan_version': 4, 'direction_axis': 9}[alteration]
            with self.assertRaises(ValueError):
                validate_plan(bad)

    def test_legacy_extended_plan_still_passes(self):
        old = json.loads(subprocess.check_output(
            ['node', '-e', "console.log(JSON.stringify(require('./web/unified-plan').eventPlan(71,{extended:true})))"],
            cwd=Path(__file__).resolve().parents[1], text=True, encoding='utf-8'))
        self.assertEqual(len(validate_plan(old)), 48)

    def test_short_plan_rejects_missing_band_split_and_wrong_actual_axis(self):
        bad = copy.deepcopy(self.plan)
        # Keep identifiers internally consistent while breaking 6/3/3 balance.
        step = next(s for s in bad if s['split'] == 'validation')
        step['split'] = 'train'
        step['trial_id'] = step['block'] = step['trial_id'].replace('validation-', 'train-extra-')
        with self.assertRaises(ValueError):
            validate_plan(bad)
        bad = copy.deepcopy(self.plan)
        step = bad[0]
        a, b, _ = step['points']
        dx, dy = b[0]-a[0], b[1]-a[1]
        step['points'] = [[.5,.5],[.5-dy,.5+dx],[.5,.5]]
        with self.assertRaisesRegex(ValueError, '声明方向'):
            validate_plan(bad)

    def test_variable_display_times_drive_labels_windows_and_coverage(self):
        step = self.plan[0]
        # Deliberately different from the intended change times: rAF scheduling
        # is an observation, not a perfect playback of the stored plan.
        changes = [step['changes'][0]+24, step['changes'][1]+48]
        events, frames = [], []
        for age in range(0, step['duration'], 20):
            k = sum(age >= c for c in changes)
            x, y = step['points'][k]
            events.append(dict(pc_ms=1000+age, trial_age_ms=age, x=x, y=y, phase='jump',
                block=step['block'], trial_id=step['trial_id'], split=step['split'],
                capture_segment=1, visible=True, sync_rtt_ms=8))
            frames.append(dict(index=len(frames),source_ms=1000+age,pc_read_ms=1000+age,
                valid=True,input='eyes.npz',timing={'source_capture_monotonic_ns':(1000+age)*1e6}))
        rows = align_frames(frames, events)
        self.assertTrue(coverage_report(rows, [step])['ready'])
        windows = target_windows(rows, range(len(rows)-1))
        self.assertEqual(len(windows), 3)
        actual = sorted(key[-1] for key in windows)
        self.assertEqual(actual, [0, *[20*((c+19)//20) for c in changes]])
        self.assertEqual(len(fixation_windows(rows, step['split'])), 3)
        for row in rows[:-1]:
            self.assertEqual(row['weight'] > .5, row['target_age_ms'] >= 704)
        missing_last = [r for r in rows if r['trial_age_ms'] < actual[-1]+700]
        self.assertFalse(coverage_report(missing_last, [step])['ready'])
        restarted = rows + [dict(r, capture_segment=2) for r in rows]
        self.assertEqual(len(fixation_windows(restarted, step['split'])), 6)
        # Unsupported final telemetry is not a phantom old-style 2700 ms hold.
        self.assertEqual(len(target_windows(rows, range(len(rows)))), 3)


if __name__ == '__main__':
    unittest.main()
