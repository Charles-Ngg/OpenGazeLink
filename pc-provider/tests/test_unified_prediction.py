import unittest
from types import SimpleNamespace
import numpy as np
import torch
from opengazelink_pc.unified_prediction import UnifiedHistory, UnifiedNetwork, UnifiedPrediction, EventTrajectoryNetwork, CONTEXT_DIM
from opengazelink_pc.conditioned_eye_model import ConditionedEyeModel, EmbeddedJointForecast


class UnifiedPredictionTests(unittest.TestCase):
    def test_dual_scale_preserves_high_rate_evidence_and_long_experts(self):
        slow=UnifiedHistory(25.,True)
        dual=UnifiedHistory(25.,True,1000/120)
        for i in range(49):
            t=i*1000/120
            raw=np.array([.3+t*.0002,.4+(.02 if i==47 else 0)])
            a=slow.update(t,raw,raw,np.zeros(404),np.zeros(64))
            b=dual.update(t,raw,raw,np.zeros(404),np.zeros(64))
        np.testing.assert_allclose(a[1],b[1],atol=1e-6)
        trajectory=b[0][64:148].reshape(12,7)
        self.assertAlmostEqual(-11/120,float(trajectory[0,4]),places=6)
        self.assertGreater(float(trajectory[-2,1]),.019)
        self.assertEqual(a[2],b[2])
        self.assertEqual(0.,UnifiedPrediction(None,{}).history.recent_interval_ms)
        self.assertAlmostEqual(1000/120,UnifiedPrediction(None,{'history_recent_interval_ms':1000/120}).history.recent_interval_ms)

    def test_event_model_landing_is_finite_bounded_and_exportable(self):
        history=UnifiedHistory(25.,True,1000/120)
        for i in range(49):
            t=i*1000/120
            raw=np.array([.3+t*.0002,.4])
            context,experts,_=history.update(t,raw,raw,np.zeros(404),np.zeros(64))
        model=EventTrajectoryNetwork(np.zeros(CONTEXT_DIM),np.ones(CONTEXT_DIM)).eval()
        ctx=torch.from_numpy(np.tile(context,(3,1)))
        exp=torch.from_numpy(np.tile(experts,(3,1,1)))
        h=torch.tensor([[0.],[.085],[.25]])
        scripted=torch.jit.trace(model,(ctx[:1],exp[:1],h[:1]),strict=False)
        for a,b in zip(model(ctx,exp,h),scripted(ctx,exp,h)):torch.testing.assert_close(a,b)
        _,_,velocity,endpoint,remaining=model.state(ctx,exp)
        torch.testing.assert_close(velocity,torch.tensor([[.2,0.]]).expand(3,2),atol=1e-5,rtol=1e-5)
        self.assertTrue(bool(((remaining>=.008)&(remaining<=.12)).all()))
        self.assertTrue(bool((endpoint[:,0]>0).all()))
        stationary=torch.zeros_like(ctx); stationary[:,64:148].reshape(-1,12,7)[:,:,6]=1
        result=model(stationary,torch.zeros_like(exp),h)[0]
        torch.testing.assert_close(result,torch.zeros_like(result))

    def test_source_time_history_has_same_coverage_and_motion_at_30_and_120_hz(self):
        outputs=[]
        for rate in (30,120):
            history=UnifiedHistory(sample_interval_ms=25.)
            for t in np.linspace(0,400,round(.4*rate)+1):
                raw=np.array([.3+t*.0002,.4])
                output=history.update(t,raw,raw,np.zeros(404),np.zeros(64))
            outputs.append(output)
            self.assertEqual(12,output[2])
            trajectory=output[0][64:148].reshape(12,7)
            self.assertAlmostEqual(-.275,float(trajectory[0,4]),places=6)
            np.testing.assert_allclose(output[1][2:5,2:],np.tile([.2,0],(3,1)),atol=1e-5)
        np.testing.assert_allclose(outputs[0][0],outputs[1][0],atol=2e-5)

    def test_old_artifacts_keep_frame_history_and_new_artifacts_use_source_time(self):
        self.assertEqual(0.,UnifiedPrediction(None,{}).history.sample_interval_ms)
        self.assertEqual(25.,UnifiedPrediction(None,{"history_sample_interval_ms":25.}).history.sample_interval_ms)

    def test_stationary_expert_denoises_with_external_filter_disabled(self):
        rng=np.random.default_rng(192)
        history=UnifiedHistory(25.,intrinsic_stability=True)
        errors=[];raw_errors=[]
        for i in range(240):
            raw=np.full(2,.5)+rng.normal(0,.004,2)
            _,experts,n=history.update(i*1000/120,raw,raw,np.zeros(404),np.zeros(64))
            if n==12:
                errors.append(np.linalg.norm(raw+experts[1,:2]-.5))
                raw_errors.append(np.linalg.norm(raw-.5))
        self.assertLess(np.mean(errors),np.mean(raw_errors)*.65)

    def test_irregular_motion_fits_are_causal_and_translation_invariant(self):
        histories=[UnifiedHistory(),UnifiedHistory()]
        for t in (0,21,57,93,124,166,201,236):
            outputs=[]
            for history,offset in zip(histories,(0.,.2)):
                raw=np.array([.3+t*.0002,.4])+offset
                outputs.append(history.update(t,raw,raw-.01,np.zeros(404),np.zeros(64)))
            np.testing.assert_allclose(outputs[0][0],outputs[1][0],atol=2e-5)
        experts=outputs[0][1]
        np.testing.assert_allclose(experts[2:5,2:],np.tile([.2,0],(3,1)),atol=1e-5)
        _,_,n=histories[0].update(500,[.5,.5],[.5,.5],np.zeros(404),np.zeros(64))
        self.assertEqual(n,1)
        with self.assertRaises(ValueError):
            histories[0].update(float("nan"),[.5,.5],[.5,.5],np.zeros(404),np.zeros(64))
        self.assertEqual(len(histories[0].samples),0)

    def test_no_filter_and_no_hidden_ablations_remove_those_inputs(self):
        torch.manual_seed(2)
        model=UnifiedNetwork(np.zeros(CONTEXT_DIM),np.ones(CONTEXT_DIM),False,False,0.).eval()
        context=torch.randn(5,CONTEXT_DIM);experts=torch.randn(5,6,4)*.01;h=torch.rand(5,1)*.25
        expected=model(context,experts,h)
        context[:,:64]+=10;experts[:,1]+=10
        actual=model(context,experts,h)
        for a,b in zip(expected,actual):torch.testing.assert_close(a,b)

    def test_export_matches_eager_and_does_not_predict_a_new_direction_at_rest(self):
        torch.manual_seed(3)
        model=UnifiedNetwork(np.zeros(CONTEXT_DIM),np.ones(CONTEXT_DIM),True,False,0.).eval()
        context=torch.randn(7,CONTEXT_DIM);experts=torch.zeros(7,6,4);h=torch.linspace(0,.25,7)[:,None]
        scripted=torch.jit.trace(model,(context[:1],experts[:1],h[:1]),strict=False)
        for a,b in zip(model(context,experts,h),scripted(context,experts,h)):torch.testing.assert_close(a,b)
        torch.testing.assert_close(scripted(context,experts,h)[0],torch.zeros(7,2))

    def test_runtime_disabled_and_zero_lead_keep_stable(self):
        model=UnifiedNetwork(np.zeros(CONTEXT_DIM),np.ones(CONTEXT_DIM)).eval()
        runtime=UnifiedPrediction(model,{})
        for i in range(5):
            point,info=runtime.update([100,100],[90,90],i*33,np.zeros(404),np.zeros(64),[1920,1080],85,0)
            self.assertEqual(point,(90,90))

    def test_automatic_horizon_covers_real_capture_age_and_clamps_without_reset(self):
        class Forecast:
            metadata={"schema":"opengazelink-unified-prediction-v3","filter_config":{}}
            resets=0
            def reset(self):self.resets+=1
            def update(self,*args,**kwargs):
                return (1,2),{"horizon_ms":min(250,args[6])}
        wrapper=ConditionedEyeModel.__new__(ConditionedEyeModel)
        wrapper.forecast=Forecast();wrapper._previous=torch.zeros(1,404);wrapper._hidden=torch.zeros(1,64);wrapper.reset_probability=0.
        config=SimpleNamespace(extrapolation_horizon_ms=85,prediction_auto_horizon_enabled=True,screen_width=1920,screen_height=1080,extrapolation_max_lead_fraction=.12)
        for horizon in (175,200,270):
            _,info=wrapper.forecast_point([0,0],[0,0],1000,config,{"horizon_ms_proxy":horizon})
            self.assertEqual(info["horizon_ms"],min(250,horizon))
            self.assertEqual(wrapper.forecast.resets,0)
        _,info=wrapper.forecast_point([0,0],[0,0],1000,config,{"horizon_ms_proxy":501})
        self.assertEqual(info["mode"],"prediction_stale_frame")
        self.assertEqual(wrapper.forecast.resets,1)

    def test_embedded_joint_prediction_uses_automatic_horizon_and_caps_at_training_limit(self):
        forecast=EmbeddedJointForecast({"schema":"opengazelink-joint-video-v1",
            "filter_config":{},"max_horizon_ms":100.})
        forecast.set_motion([1.,0.,0.,0.])
        wrapper=ConditionedEyeModel.__new__(ConditionedEyeModel)
        wrapper.forecast=forecast;wrapper._previous=torch.zeros(1,404);wrapper._hidden=torch.zeros(1,64);wrapper.reset_probability=0.
        config=SimpleNamespace(extrapolation_horizon_ms=85,prediction_auto_horizon_enabled=True,
            screen_width=101,screen_height=101,extrapolation_max_lead_fraction=1.)
        point,info=wrapper.forecast_point([0.,0.],[0.,0.],1000,config,{"horizon_ms_proxy":120.})
        self.assertEqual(100.,info["horizon_ms"])
        self.assertEqual(120.,info["requested_horizon_ms"])
        np.testing.assert_allclose(point,[10.,0.],atol=1e-6)

    def test_saved_output_filter_change_does_not_disable_trained_prediction_filter(self):
        class Forecast:
            metadata={"schema":"opengazelink-unified-prediction-v3","filter_config":{
                "one_euro_enabled":True,"one_euro_min_cutoff":1.5,
                "one_euro_beta":2.5,"one_euro_derivative_cutoff":1.0,
            }}
            resets=0
            calls=0
            def reset(self):self.resets+=1
            def update(self,*args,**kwargs):
                self.calls+=1
                return (3,4),{"mode":"prediction_fixation","horizon_ms":args[6]}
        wrapper=ConditionedEyeModel.__new__(ConditionedEyeModel)
        wrapper.forecast=Forecast();wrapper._previous=torch.zeros(1,404);wrapper._hidden=torch.zeros(1,64);wrapper.reset_probability=0.
        config=SimpleNamespace(extrapolation_horizon_ms=85,prediction_auto_horizon_enabled=False,
            screen_width=1920,screen_height=1080,extrapolation_max_lead_fraction=.12,
            one_euro_enabled=False,one_euro_min_cutoff=1.5,one_euro_beta=2.5,one_euro_derivative_cutoff=1.0)
        point,info=wrapper.forecast_point([1,2],[1,2],1000,config,{})
        self.assertEqual((3,4),point)
        self.assertEqual("prediction_fixation",info["mode"])
        self.assertEqual(1,wrapper.forecast.calls)
        self.assertTrue(wrapper.forecast_filter_config()["one_euro_enabled"])

if __name__=="__main__":unittest.main()
