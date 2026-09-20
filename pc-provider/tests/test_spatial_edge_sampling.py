import copy
import json
from pathlib import Path
import subprocess
import unittest

import numpy as np

from opengazelink_pc.head_coverage import pose_balance
from opengazelink_pc.unified_capture import validate_plan


class SpatialEdgeSamplingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plans = json.loads(subprocess.check_output(
            ['node', '-e', "const {spatialPlan}=require('./web/unified-plan');"
             "console.log(JSON.stringify([[640,480],[1366,768],[1920,1080],[3840,2160]]"
             ".flatMap(([width,height])=>Array.from({length:100},(_,seed)=>spatialPlan(seed,{width,height})))));"],
            cwd=Path(__file__).resolve().parents[1], text=True, encoding='utf-8'))

    def test_plans_cover_all_regions_in_training_and_selection(self):
        for plan in self.plans:
            validate_plan(plan)
            for split in ('train', 'test'):
                local = [s for s in plan if s['split'] == split]
                cells = set()
                for step in local:
                    positions = np.linspace(step['point'], step['end'], 301)
                    cells.update(map(tuple, np.clip((positions*3).astype(int), 0, 2)))
                    self.assertFalse(any(k in step for k in ('head_pose', 'head_start', 'head_end')))
                self.assertEqual(9, len(cells))

    def test_actual_pixel_margins_and_corner_endpoint_counts(self):
        for plan in self.plans[::100]:
            size = np.array([plan[0]['viewport_width'], plan[0]['viewport_height']])
            train = [s for s in plan if s['split'] == 'train']
            endpoints = np.array([point for s in train for point in (s['point'],s['end'])])
            np.testing.assert_allclose((endpoints*size).min(0), 18)
            np.testing.assert_allclose(((1-endpoints)*size).min(0), 18)
            for x in (0,1):
                for y in (0,1):
                    corner = np.where([x,y], size-18, 18)
                    self.assertEqual(2, int((np.linalg.norm(endpoints*size-corner,axis=1)<1e-5).sum()))

    def test_contract_rejects_missing_edge_and_center_or_mixed_versions(self):
        for changes in (dict(point=[.1,.1]), dict(plan_version=7),
                        dict(head_pose='head_left'), dict(viewport_width=800)):
            plan = copy.deepcopy(self.plans[0])
            plan[0].update(changes)
            with self.assertRaises(ValueError):
                validate_plan(plan)

    def test_region_weights_balance_different_capture_durations_without_holdout(self):
        rows = []
        for y in range(3):
            for x in range(3):
                # Different dwell times and repeated trials cannot grow a cell's budget.
                for trial in range(1 + (x+y)%2):
                    for _ in range(20*(1+x)):
                        rows.append(dict(split='train', trial_id=f'train-{x}-{y}-{trial}',
                                         weight=.25, target=[(x+.5)/3,(y+.5)/3]))
        train = np.arange(len(rows))
        rows.extend([dict(split='test',trial_id='test-0',weight=1.,target=[.5,.5])]*100)
        rotations = np.tile(np.eye(3),(len(rows),2,1,1))
        weights,audit = pose_balance(rotations,rows,train,self.plans[0])
        mass = [c['balance_mass'] for c in audit['spatial_region_balance'].values()]
        np.testing.assert_allclose(mass, np.mean(mass), rtol=1e-5)
        self.assertTrue((weights[len(train):] == 0).all())
        more_rows = rows + [dict(split='test',trial_id='test-other',weight=1.,target=[.01,.01])]*100
        more_rotations = np.tile(np.eye(3),(len(more_rows),2,1,1))
        other,_ = pose_balance(more_rotations,more_rows,train,self.plans[0])
        np.testing.assert_allclose(weights[train],other[train], rtol=1e-6)

    def test_sampling_ceiling_accepts_legacy_60_and_new_120_but_not_mixed(self):
        from opengazelink_pc.video_replay import processing_rate_limit
        for hz in (60, 120):
            plan = copy.deepcopy(self.plans[0])
            for step in plan:
                step['sample_rate_hz'] = hz
            validate_plan(plan)
            self.assertEqual(processing_rate_limit({'plan': plan}), hz)
        plan[0]['sample_rate_hz'] = 60
        with self.assertRaises(ValueError):
            validate_plan(plan)

    def test_missing_region_remains_visible_and_is_not_synthesized(self):
        rows = [dict(split='train',trial_id='train-0',weight=1.,target=[.5,.5])]*20
        weights,audit = pose_balance(np.tile(np.eye(3),(20,2,1,1)),rows,np.arange(20),self.plans[0])
        self.assertEqual(8,sum(c['frames']==0 for c in audit['spatial_region_balance'].values()))
        self.assertTrue(np.isfinite(weights).all())


if __name__ == '__main__':
    unittest.main()
