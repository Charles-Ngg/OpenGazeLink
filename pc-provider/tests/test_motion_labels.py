import unittest
import numpy as np
from opengazelink_pc.motion_labels import label_motion


def rows_for(t, target, phase='pursuit'):
    return [dict(index=i,source_ms=float(ti),target=target[i].tolist(),phase=phase,
                 trial_id='train-motion',split='train',stimulus_supported=True,reset=i==0)
            for i,ti in enumerate(t)]


class MotionLabelTests(unittest.TestCase):
    def test_slow_pursuit_survives_frame_rate_and_static_prior(self):
        for fps in (30,60,120,240):
            t=np.arange(0,2000,1000/fps); p=np.c_[.2+t*.0001,np.full(len(t),.4)]
            labels=label_motion(p,rows_for(t,p),.0035)
            self.assertTrue(np.all(labels['phase'][t>500]==1))
            static=np.full_like(p,.5)
            labels=label_motion(static,rows_for(t,static,'anchor'),.0035)
            self.assertTrue(np.all(labels['phase']==0))

    def test_delayed_saccade_uses_eye_onset_and_rejects_returning_spike(self):
        boundaries=[]
        for fps in (30,60,120,240):
            t=np.arange(0,2000,1000/fps)
            target=np.c_[np.where(t<700,.2,.5),np.full(len(t),.4)]
            p=np.c_[.2+.3*np.clip((t-880)/60,0,1),np.full(len(t),.4)]
            labels=label_motion(p,rows_for(t,target,'jump'),.0035)
            self.assertEqual(len(labels['audit']['events']),1)
            event=labels['audit']['events'][0]
            boundaries.append((t[event['start']],t[event['end']]))
            self.assertGreater(t[event['start']],800)
            self.assertTrue(np.all(labels['phase'][t<800]==0))
            np.testing.assert_allclose(labels['landing'][labels['landing_valid']],np.tile([.5,.4],(labels['landing_valid'].sum(),1)),atol=.01)
            p=np.full_like(p,.4); p[len(t)//2,0]+=.3
            spike=label_motion(p,rows_for(t,np.full_like(p,.4),'anchor'),.0035)
            self.assertFalse(spike['landing_valid'].any())
        self.assertLess(np.ptp(np.array(boundaries)[:,0]),45)
        self.assertLess(np.ptp(np.array(boundaries)[:,1]),45)

    def test_test_data_cannot_change_train_lag_or_labels(self):
        t=np.arange(0,3000,1000/120); p=np.c_[.5+.2*np.sin(t/300),np.full(len(t),.5)]
        rows=rows_for(t,p); first=label_motion(p,rows,.003)
        other=[dict(r,index=len(rows)+i,trial_id='test',split='test',reset=i==0) for i,r in enumerate(rows)]
        both=label_motion(np.r_[p,p[::-1]],rows+other,.003)
        self.assertEqual(first['audit']['lag_model'],both['audit']['lag_model'])
        np.testing.assert_array_equal(first['phase'],both['phase'][:len(t)])

    def test_train_alignment_recovers_known_delay(self):
        t=np.arange(0,9000,1000/120)
        target=np.c_[.5+.2*np.cos(t/250),.5+.2*np.sin(t/250)]
        observed=np.c_[.5+.2*np.cos((t-120)/250),.5+.2*np.sin((t-120)/250)]
        labels=label_motion(observed,rows_for(t,target),.0035)
        fit=labels['audit']['lag_model']
        self.assertTrue(fit['reliable'])
        self.assertAlmostEqual(fit['lag_ms'],120,delta=10)

    def test_gap_cannot_create_an_event(self):
        t=np.r_[np.arange(0,800,1000/120),np.arange(1200,2000,1000/120)]
        p=np.c_[np.where(t<1000,.2,.6),np.full(len(t),.4)]
        labels=label_motion(p,rows_for(t,p,'jump'),.0035)
        self.assertFalse(labels['landing_valid'].any())
