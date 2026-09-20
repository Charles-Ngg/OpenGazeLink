import json
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from opengazelink_pc.personal_binocular_training import spatial_supervision, spatial_supervision_loss
from opengazelink_pc.unified_calibration_training import train_calibration
from opengazelink_pc.unified_capture import coverage_report
from opengazelink_pc.head_coverage import pose_balance


class SpatialStageTests(unittest.TestCase):
    @staticmethod
    def rotation(yaw, pitch):
        y,p = np.radians([yaw,pitch])
        ry=np.array([[np.cos(y),0,np.sin(y)],[0,1,0],[-np.sin(y),0,np.cos(y)]])
        rx=np.array([[1,0,0],[0,np.cos(p),-np.sin(p)],[0,np.sin(p),np.cos(p)]])
        value=(ry@rx).astype(np.float32)
        return np.stack((value,value))

    def test_measured_head_coverage_requires_both_axes_and_crossed_cells(self):
        rows=[];rotations=[]
        for cell,target in enumerate(([.1,.1],[.35,.35],[.6,.6],[.85,.85])):
            for yaw,pitch in ((-10,-10),(-10,10),(10,-10),(10,10)):
                rows.append(dict(split="train",weight=1.,target=target,trial_id=f"train-{cell}"))
                rotations.append(self.rotation(yaw,pitch))
        _,audit=pose_balance(np.asarray(rotations),rows,np.arange(len(rows)))
        self.assertTrue(audit["by_split"]["train"]["adequate"])
        _,static_audit=pose_balance(np.asarray([self.rotation(0,0)]*len(rows)),rows,np.arange(len(rows)))
        self.assertFalse(static_audit["by_split"]["train"]["adequate"])

    def test_measured_head_motion_must_follow_paired_and_opposite_instructions(self):
        rows=[];rotations=[];plan=[]
        pairs=[("yaw","head_left","head_right",[.1,.1],[.35,.1],1),
               ("pitch","head_up","head_down",[.6,.1],[.85,.1],1),
               ("yaw","head_right","head_left",[.1,.6],[.35,.6],-1),
               ("pitch","head_down","head_up",[.6,.6],[.85,.6],-1)]
        for pair_index,(axis,start,end,a,b,direction) in enumerate(pairs):
            for rail_index in (0,1):
                trial=f"train-{pair_index}-{rail_index}"
                plan.append(dict(trial_id=trial,split="train",pair_id=f"train-pair-{pair_index}",
                                 head_axis=axis,head_start=start if rail_index==0 else end,
                                 head_end=end if rail_index==0 else start))
                offset=np.array([0,.05] if axis=="yaw" else [.05,0])*rail_index
                for progress,target in ((0.,np.array(a)+offset),(1.,np.array(b)+offset)):
                    angle=direction*(1 if rail_index==0 else -1)*(-10 if progress==0 else 10)
                    for _ in range(4):
                        rows.append(dict(split="train",weight=1.,target=target,trial_id=trial,
                                         drag_progress=progress))
                        rotations.append(self.rotation(angle if axis=="yaw" else 0,
                                                       angle if axis=="pitch" else 0))
        _,audit=pose_balance(np.asarray(rotations),rows,np.arange(len(rows)),plan)
        self.assertTrue(audit["by_split"]["train"]["adequate"])
        self.assertTrue(audit["planned_movement_compliance"]["by_split"]["train"]["adequate"])
        # Reversing one traversal's measured head motion recreates a rail/head correlation.
        bad=np.asarray(rotations).copy()
        trial_ids=[i for i,row in enumerate(rows) if row["trial_id"]=="train-0-1"]
        bad[trial_ids]=bad[trial_ids[::-1]]
        _,bad_audit=pose_balance(bad,rows,np.arange(len(rows)),plan)
        self.assertFalse(bad_audit["by_split"]["train"]["adequate"])

    def loss(self, prediction, rows, target=None):
        target = torch.ones_like(prediction) if target is None else target
        direction = torch.tensor([[0., 0., 1.]] * (2 * len(rows)), requires_grad=True)
        truth = torch.tensor([[1., 0., 0.]] * (2 * len(rows)))
        return spatial_supervision_loss(prediction, target, direction, truth, spatial_supervision(rows)), direction

    def test_unknown_frames_have_no_coordinate_or_direction_gradient(self):
        point = torch.tensor([[.2, .3]], requires_grad=True)
        loss, direction = self.loss(point, [dict(weight=0., constraint=None)])
        loss.sum().backward()
        self.assertEqual(0., float(loss.sum()))
        self.assertEqual(0., float(point.grad.abs().sum()))
        self.assertEqual(0., float(direction.grad.abs().sum()))

    def test_rail_does_not_pull_toward_screen_target_or_exact_eye_direction(self):
        rows = [dict(weight=.25, constraint=dict(normal=[0., 1.], tangent=[1., 0.],
                                               normal_target=.3, lower=.2, upper=.8))]
        point = torch.tensor([[.5, .3]], requires_grad=True)
        loss, direction = self.loss(point, rows)
        loss.sum().backward()
        self.assertAlmostEqual(0., float(loss.sum()))
        self.assertEqual(0., float(direction.grad.abs().sum()))
        self.assertEqual(0., float(point.grad.abs().sum()))
        point = torch.tensor([[.5, .5]], requires_grad=True)
        loss, _ = self.loss(point, rows)
        loss.sum().backward()
        self.assertGreater(float(point.grad[0, 1]), 0.)
        self.assertEqual(0., float(point.grad[0, 0]))

    def test_settled_anchor_has_exact_position_and_eye_gradient(self):
        point = torch.tensor([[.2, .3]], requires_grad=True)
        loss, direction = self.loss(point, [dict(weight=1., constraint=None)])
        loss.sum().backward()
        self.assertGreater(float(loss.sum()), 0.)
        self.assertLess(float(point.grad[0, 0]), 0.)
        self.assertGreater(float(direction.grad.abs().sum()), 0.)

    def test_waiting_and_half_drag_cannot_pass_coverage(self):
        step = dict(trial_id='train-line', split='train', motion_profile='line',
                    duration=5200, calibration_stage='spatial_v1')
        rows = [dict(trial_id='train-line', capture_segment=1, valid=True, stimulus_supported=True,
                     pc_ms=i*30, weight=1., phase='anchor', trial_complete=True) for i in range(200)]
        self.assertFalse(coverage_report(rows, [step])['ready'])
        for i, row in enumerate(rows[25:175]):
            row.update(phase='pursuit', weight=.25, constraint={'normal':[0, 1]}, drag_progress=i/149)
        for row in rows[175:]:
            row['drag_progress'] = 1.
        self.assertTrue(coverage_report(rows, [step])['ready'])
        for row in rows:
            row['trial_complete'] = False
        self.assertFalse(coverage_report(rows, [step])['ready'])

    def test_spatial_stage_skips_prediction_and_preserves_rejected_model(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root/'session.json').write_text(json.dumps({'plan':[{'calibration_stage':'spatial_v1'}]}))
            result = {'publication_reason':'keep_existing_model', 'training_directory':folder}
            with patch('opengazelink_pc.video_training.train_session', return_value=result) as spatial, \
                 patch('opengazelink_pc.unified_prediction_training.train_unified') as temporal, \
                 patch('opengazelink_pc.unified_calibration_training.DATA_DIR', root):
                report = train_calibration(root, progress=lambda _:None)
            self.assertTrue(report['spatial_only'])
            self.assertFalse(report['published'])
            self.assertFalse(spatial.call_args.kwargs['train_temporal'])
            temporal.assert_not_called()

    def test_spatial_publication_skips_prediction_without_accuracy_gate(self):
        for worse in (False, True):
            with self.subTest(worse=worse), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                session = root/'session'; session.mkdir()
                candidate = root/'candidate'; candidate.mkdir()
                (session/'session.json').write_text(json.dumps({'plan':[{'calibration_stage':'spatial_v1'}]}))
                old = {'variants':{'conditioned_video':{'module_file':'old.pt'}}}
                (root/'conditioned-video-model.json').write_text(json.dumps(old))
                (root/'old.pt').write_bytes(b'old')
                (root/'conditioned-video-forecast.json').write_text('{}')
                module = candidate/'new.pt'; module.write_bytes(b'new')
                meta = {'variants':{'conditioned_video':{'module_file':'new.pt', 'module_sha256':hashlib.sha256(b'new').hexdigest()}}}
                (candidate/'conditioned-video-model.json').write_text(json.dumps(meta))
                baseline = dict(mean_px=100.,median_px=90.,p95_px=180.)
                current = dict(mean_px=101. if worse else 80.,median_px=70.,p95_px=170.)
                result = dict(publication_reason='eligible', training_directory=str(candidate),
                              base_source='public_pretrained',candidate=current,incumbent_selection=baseline,
                              independent_test={'baseline':dict(mean_px=200.,median_px=190.,p95_px=380.),
                                                'incumbent':baseline,'candidate':current})
                with patch('opengazelink_pc.video_training.train_session', return_value=result), \
                     patch('opengazelink_pc.unified_prediction_training.train_unified') as temporal, \
                     patch('opengazelink_pc.unified_calibration_training.DATA_DIR', root):
                    report = train_calibration(session, progress=lambda _:None)
                temporal.assert_not_called()
                self.assertTrue(report['published'])
                self.assertFalse((root/'conditioned-video-forecast.json').exists())
                self.assertEqual(meta,json.loads((root/'conditioned-video-model.json').read_text()))
                self.assertEqual(old, json.loads((Path(report['training_directory'])/'previous-spatial.json').read_text()))


if __name__ == '__main__':
    unittest.main()
