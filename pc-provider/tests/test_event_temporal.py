import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch
import numpy as np
from opengazelink_pc.event_temporal import EventTemporalFilter, fit_landing
from opengazelink_pc.spatial_metrics import spatial_metrics, acceptance, eccentricity_degrees


class EventTemporalTests(unittest.TestCase):
    def test_correlated_fixation_noise_does_not_toggle_stability(self):
        for fps in (30, 60, 120, 240):
            rng = np.random.default_rng(182)
            f = EventTemporalFilter()
            f.set_stability_profile(dict(noise_prior_deg=.4, noise_floor_deg=.2))
            jitter = np.zeros(2)
            states = []
            for i in range(fps*4):
                jitter = .65*jitter + rng.normal(0, 8, 2)
                _, state = f.update([900,550]+jitter, i*1000/fps, (1920,1080),
                                    pixels_per_degree=40, horizon_ms=85)
                states.append(state)
            self.assertFalse(any(s['prediction_active'] for s in states), fps)
            self.assertLess(sum(not s['stability_active'] for s in states), len(states)*.03, fps)
            switches = sum(a['stability_active'] != b['stability_active'] for a,b in zip(states,states[1:]))
            self.assertLessEqual(switches, 2, fps)

    def test_pursuit_confirmation_and_quiet_hysteresis_release(self):
        for fps in (30,60,120,240):
            f = EventTemporalFilter()
            states = []
            for t in np.arange(0, 1000, 1000/fps):
                p = 700 + np.clip(t-200, 0, 400)*.4
                _, state = f.update((p,500),t,(1920,1080),pixels_per_degree=40,horizon_ms=85)
                states.append(state)
            self.assertTrue(any(s['mode']=='event_pursuit' for s in states))
            self.assertFalse(any(s['prediction_active'] for s in states))
            self.assertTrue(states[-1]['stability_active'])
            self.assertEqual(sum(a['stability_active'] != b['stability_active'] for a,b in zip(states,states[1:])),2)

    def test_head_motion_does_not_generate_landing_prediction(self):
        f = EventTemporalFilter()
        for i in range(30):
            a = i*.02
            rotation=np.array([[np.cos(a),0,np.sin(a)],[0,1,0],[-np.sin(a),0,np.cos(a)]])
            _,s=f.update((600+30*i,500),i*8.33,(1920,1080),pixels_per_degree=40,
                         horizon_ms=85,head_rotation=rotation)
            self.assertIsNone(s['landing_px'])

    def test_fixed_gaze_jitter_reduced_without_drift_at_multiple_rates(self):
        for fps in (30, 60, 120, 240):
            rng = np.random.default_rng(3)
            f = EventTemporalFilter()
            raw = rng.normal(0, 3, (fps*3, 2)) + [900, 550]
            out, states = [], []
            for i, p in enumerate(raw):
                q, s = f.update(p, i*1000/fps, (1920,1080), pixels_per_degree=40, horizon_ms=60)
                out.append(q); states.append(s)
            self.assertLess(np.std(np.asarray(out)[fps:],axis=0).mean(), np.std(raw[fps:],axis=0).mean()*.6)
            self.assertFalse(any(s['landing_px'] is not None for s in states))

    def test_endpoint_family_recovers_landing(self):
        t = np.arange(0,41,8.)
        points = np.c_[10*(1-np.exp(-(t/30)**3)), np.zeros(len(t))]
        fit = fit_landing(t, points, .04)
        self.assertIsNotNone(fit)
        self.assertLess(np.linalg.norm(fit[0]-[10,0]), .8)

    def test_eye_jump_gets_bounded_prediction_and_reacquires_observation(self):
        for fps in (60, 120, 240):
            f = EventTemporalFilter(); predicted = []; out = None
            for t in np.arange(0, 600, 1000/fps):
                p = 800+400*(1-np.exp(-(max(0,t-200)/30)**3))
                out, state = f.update((p,500),t,(1920,1080),pixels_per_degree=40,horizon_ms=85)
                if state['prediction_active']:
                    predicted.append(out[0])
            self.assertTrue(predicted, fps)
            self.assertLess(abs(out[0]-1200), 1)
            self.assertLess(max(predicted), 1300)

    def test_single_spike_is_not_landing_prediction(self):
        f = EventTemporalFilter()
        for i in range(40):
            _, s = f.update((1000 if i==20 else 600,500),i*8.33,(1920,1080),pixels_per_degree=40,horizon_ms=85)
            self.assertIsNone(s['landing_px'])

    def test_noisy_fixation_can_estimate_noise_despite_high_derivative(self):
        rng=np.random.default_rng(16); f=EventTemporalFilter(); predictions=0
        for i in range(1200):
            _,s=f.update(rng.normal(0,40,2)+[1000,600],i*1000/120,(1920,1080),
                         pixels_per_degree=40,horizon_ms=85)
            predictions += s['landing_px'] is not None
        self.assertGreater(s['noise_deg'],.6)
        self.assertLess(predictions,3)

    def test_gaps_and_clock_reversal_start_fresh(self):
        f = EventTemporalFilter()
        for t in (0,8,16,300,290):
            out,s = f.update((600+t,500),t,(1920,1080),pixels_per_degree=40,horizon_ms=85)
            if t in (300,290):
                self.assertEqual(s['sample_count'],1)
                self.assertEqual(out,(600+t,500))

    def test_zero_horizon_never_predicts_endpoint(self):
        f = EventTemporalFilter()
        for i in range(12):
            _, s = f.update((600+30*i,500),i*8.33,(1920,1080),pixels_per_degree=40,horizon_ms=0)
            self.assertIsNone(s['landing_px'])


class RegionMetricsTests(unittest.TestCase):
    def make(self, center_error, middle_error, edge_error):
        target=np.zeros((90,2))
        points=np.c_[np.repeat([center_error,middle_error,edge_error],30)/1000,np.zeros(90)]
        return spatial_metrics(points,target,np.ones(90),np.array([1001,1001]),
                               np.repeat([2.,10.,20.],30),[dict(trial_id=str(i%3)) for i in range(90)])

    def test_middle_gate_not_dominated_by_edges(self):
        baseline=self.make(10,50,100)
        candidate=self.make(10,40,1000)
        self.assertTrue(acceptance(candidate,baseline))
        self.assertEqual(candidate['by_region']['middle']['p75_px'],40)
        self.assertFalse(acceptance(self.make(10,51,1),baseline))

    def test_rails_do_not_become_point_truth_or_pass_empty_gate(self):
        m=spatial_metrics(np.zeros((30,2)),np.ones((30,2)),np.ones(30)*.25,[1920,1080],
                          np.ones(30)*10,[dict(trial_id=str(i%3)) for i in range(30)])
        self.assertEqual(m['frames'],0)
        self.assertFalse(acceptance(m,m))

    def test_real_angle_depends_on_eye_distance(self):
        near=eccentricity_degrees([[.75,.5]], [[[0,0,50],[0,0,50]]],[0,0,0],[60,34])
        far=eccentricity_degrees([[.75,.5]], [[[0,0,100],[0,0,100]]],[0,0,0],[60,34])
        self.assertGreater(near[0],far[0])


class EventEvaluationTests(unittest.TestCase):
    def test_stage_two_reports_separate_splits_without_training_or_publication(self):
        from opengazelink_pc.event_evaluation import evaluate_session
        from opengazelink_pc.config import ProviderConfig
        from dataclasses import asdict
        cfg=asdict(ProviderConfig())
        rows=[]; raw=[]; segment=[]
        for j, split in enumerate(('train','validation','test')):
            for t in range(0,4200,20):
                k=0 if t<1200 else 1 if t<2700 else 2
                start=(0,1200,2700)[k]
                target=((.5,.5),(.65,.5),(.35,.5))[k]
                rows.append(dict(input='eyes.npz',index=len(rows),trial_id=split+'-0',split=split,
                    target=target,weight=1. if t-start>=700 else 0.,constraint=None,
                    source_ms=j*5000+t,pc_ms=j*5000+t,trial_age_ms=t,dt_ms=20.))
                raw.append(target);segment.append(j)
        data=dict(raw=np.asarray(raw),segment=np.asarray(segment),frame_index=np.arange(len(rows)))
        meta={'variants':{'conditioned_video':{'module_sha256':'fixture'}}}
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            np.savez(root/'eyes.npz',center=np.array([[0,0,60],[0,0,60]]),rotation=np.array([np.eye(3),np.eye(3)]))
            with patch('opengazelink_pc.video_forecast_training.replay',return_value=(data,rows,meta,cfg)):
                report=evaluate_session(root,progress=lambda _:None)
            self.assertFalse(report['published'])
            self.assertFalse(report['parameters_changed'])
            self.assertEqual(set(report['by_split']),{'train','validation','test'})
            self.assertEqual(len(report['events']),6)
            self.assertTrue((Path(report['training_directory'])/'report.json').is_file())


if __name__=='__main__': unittest.main()
