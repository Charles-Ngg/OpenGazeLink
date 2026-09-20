import json
import hashlib
import time
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import torch

from opengazelink_pc.prediction import MotionHistory,MotionPrediction
from opengazelink_pc.prediction_network import PredictionNetwork
from opengazelink_pc.prediction_dataset import make_examples,reconstruct,interpolation_confidence
from opengazelink_pc.prediction_training import runtime_delta,motion_beats_raw
from opengazelink_pc.prediction_timing import PredictionClock
from opengazelink_pc.video_session import VideoSession
from opengazelink_pc.config import ProviderConfig


class MotionPredictionTests(unittest.TestCase):
    def test_fast_edge_interpolation_is_uncertain_but_observed_endpoints_are_not(self):
        self.assertEqual(.25,interpolation_confidence(np.array([.2,.5]),np.array([.7,.5]),.5,.003))
        self.assertEqual(1.,interpolation_confidence(np.array([.2,.5]),np.array([.7,.5]),1.,.003))
        self.assertEqual(1.,interpolation_confidence(np.array([.2,.5]),np.array([.21,.5]),.5,.003))

    def test_slow_irregular_constant_velocity_and_jump_keep_evidence(self):
        history=MotionHistory()
        for t in (0,22,55,89,122,160):
            seq,state,n=history.update(t,[.3+t*.00002,.4],[.29+t*.00002,.4],np.zeros(404))
        np.testing.assert_allclose(state[:2],[.02,0],atol=1e-5)
        seq,state,n=history.update(193,[.55,.4],[.4,.4],np.zeros(404))
        self.assertEqual(7,n)
        self.assertGreater(seq[-1,406],1)
        seq,state,n=history.update(500,[.55,.4],[.4,.4],np.zeros(404))
        self.assertEqual(1,n)
        self.assertEqual(1,int(seq[:,409].sum()))

    def test_runtime_matches_batched_validation_including_confidence_and_cap(self):
        torch.manual_seed(4)
        model=PredictionNetwork(np.zeros(411),np.ones(411)).eval()
        scripted=torch.jit.trace(model,(torch.zeros(1,12,411),torch.zeros(1,6),torch.ones(1,1)*.085),strict=False)
        predictor=MotionPrediction(scripted,{})
        history=MotionHistory()
        wh=np.array([1920,1080])
        for i in range(15):
            raw=np.array([.3+i*.005,.4]);stable=raw-[.01,0]
            seq,state,n=history.update(i*33.,raw,stable,np.zeros(404))
            point,info=predictor.update(raw*(wh-1),stable*(wh-1),i*33.,np.zeros(404),np.zeros(64),wh,85,.02)
            if n>=4:
                with torch.inference_mode():
                    delta,sigma,logits,*_=scripted(torch.tensor(seq[None]),torch.tensor(state[None]),torch.tensor([[.085]]))
                expected=runtime_delta(delta.numpy(),sigma.numpy(),None,np.array([[.085]]),raw[None],stable[None],wh,.02,logits.softmax(-1).numpy())
                np.testing.assert_allclose(point,(raw+expected[0])*(wh-1),atol=1e-4)
                self.assertLessEqual(info["lead_distance_px"],.02*np.linalg.norm(wh-1)+1e-6)
        point,info=predictor.update(raw*(wh-1),stable*(wh-1),500,np.zeros(404),np.zeros(64),wh,0)
        np.testing.assert_allclose(point,stable*(wh-1))

    def test_centered_reconstruction_has_no_linear_phase_lag_or_cross_jump_fit(self):
        times=np.arange(20)*33.
        points=np.c_[times*.0001,np.ones(20)*.5]
        points[10:,0]+=.3
        actual=reconstruct(points,times,np.zeros(20),.001)
        np.testing.assert_allclose(actual,points,atol=1e-10)

    def test_reconstruction_supports_source_clock_restarts(self):
        times=np.tile(np.arange(9)*8.333,2)
        segments=np.repeat([0,1],9)
        points=np.full((18,2),.4)
        points[9:]=.7
        points[4,0]+=.01
        points[13,0]+=.01
        actual=reconstruct(points,times,segments,.003)
        self.assertLess(abs(actual[4,0]-.4),.003)
        self.assertLess(abs(actual[13,0]-.7),.003)

    def test_reconstruction_window_is_time_based_at_120_fps(self):
        times=np.arange(49)*1000/120
        points=np.c_[np.ones(49)*.4,np.ones(49)*.5]
        points[24,0]+=.03
        reconstructed=reconstruct(points,times,np.zeros(49),.003,window_ms=100.)
        # A 100 ms neighborhood contains roughly 25 frames at 120 FPS, so one
        # noisy observation cannot survive as it did in the old seven-frame window.
        self.assertLess(abs(reconstructed[24,0]-.4),.005)

    def test_reconstruction_preserves_fast_burst_of_small_120hz_steps(self):
        for hz in (60,120):
            times=np.arange(round(.5*hz)+1)*1000/hz
            points=np.c_[.3+.15*np.clip((times-200)/50,0,1),np.full(len(times),.5)]
            actual=reconstruct(points,times,np.zeros(len(times)),.003)
            np.testing.assert_allclose(actual,points,atol=1e-7)
            self.assertLess(abs(actual[np.argmin(abs(times-200)),0]-.3),1e-7)

    def test_examples_do_not_leak_future_or_trial_data_into_inputs(self):
        rows=[];points=[]
        for split in ("train","validation","test"):
            for i in range(100):
                index=len(rows)
                rows.append(dict(index=index,source_ms=index*33.,split=split,trial_id=split+"-a",block=split+"-a",
                    weight=1 if i<25 else 0,target=[.4,.5],stimulus_supported=True))
                points.append([.3+i*.002,.5])
        points=np.array(points,np.float32)
        values=dict(raw=points,stable=points,instantaneous=points.copy(),segment=np.zeros(300),eye_features=np.zeros((300,404),np.float32))
        frames,data,audit=make_examples(values,rows)
        for i,b in zip(data["frame"],data["future_frame"]):
            self.assertEqual(rows[i]["trial_id"],rows[b]["trial_id"])
        self.assertGreater(audit["examples"],1000)
        values["instantaneous"][200:,0]+=.2
        other,changed,_=make_examples(values,rows)
        np.testing.assert_array_equal(frames["sequences"],other["sequences"])
        np.testing.assert_allclose(data["delta"][data["split"]==0],changed["delta"][changed["split"]==0])

    def test_clock_reports_proxy_age_and_restarts_offset(self):
        clock=PredictionClock()
        timing=dict(phone_sensor_time_ns=100e6,phone_send_time_ns=110e6,pc_first_packet_monotonic_ns=1010e6)
        result=clock.observe(100,timing,1030,16)
        self.assertEqual(30,result["frame_age_ms_proxy"])
        self.assertEqual(46,result["horizon_ms_proxy"])
        self.assertTrue(result["unknown_transport_floor"])
        self.assertFalse(result["is_sensor_to_photon_measurement"])
        result=clock.observe(1,dict(phone_sensor_time_ns=1e6,phone_send_time_ns=11e6,pc_first_packet_monotonic_ns=2011e6),2030,16)
        self.assertEqual(1,result["clock_epoch"])
        self.assertEqual(29,result["frame_age_ms_proxy"])
        self.assertIsNone(clock.observe(2,{},2100)["horizon_ms_proxy"])

    def test_prediction_session_trains_only_predictor(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);manifest=root/"replay.jsonl";manifest.write_text("")
            (root/"base-model").mkdir()
            session=VideoSession.__new__(VideoSession)
            session.purpose="prediction";session.state="training";session.path=root;session.metadata={}
            session.registry=SimpleNamespace(clear=lambda:None)
            session._persist=lambda:None
            with patch("opengazelink_pc.video_replay.prepare_replay",return_value={"frames":1,"valid_frames":1,"manifest":str(manifest)}), \
                 patch("opengazelink_pc.unified_prediction_training.train_unified",return_value={"published":False}) as train, \
                 patch("opengazelink_pc.video_training.train_session") as current:
                session._train()
        train.assert_called_once();current.assert_not_called()
        self.assertEqual("prediction_not_accepted",session.phase)
        self.assertEqual("complete",session.state)

    def test_raw_control_is_required_for_motion_but_not_fixation(self):
        raw={"groups":{"fixation":{"mean_px":10},"pursuit":{"mean_px":20},"saccade_proxy":{"mean_px":30}}}
        candidate={"groups":{"fixation":{"mean_px":100},"pursuit":{"mean_px":19},"saccade_proxy":{"mean_px":29}}}
        self.assertTrue(motion_beats_raw(candidate,raw))
        candidate["groups"]["pursuit"]["mean_px"]=21
        self.assertFalse(motion_beats_raw(candidate,raw))

    def test_prediction_capture_snapshots_model_and_retains_full_trial_telemetry(self):
        class Camera:
            def camera_model(self):return {}
            def read_latest(self,sequence,timeout_s):
                time.sleep(.005)
                return True,np.zeros((8,8,3),np.uint8),time.monotonic()*1000,sequence+1
            def latest_frame_timing(self,sequence):return {"source_capture_monotonic_ns":time.monotonic_ns()}
        class Backend:
            def predict(self,*args):return None
            def close(self):pass
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);config=ProviderConfig()
            (root/"base.pt").write_bytes(b"snapshot-test")
            meta={"variants":{"conditioned_video":{"feature_dim":404,"module_file":"base.pt","module_sha256":hashlib.sha256(b"snapshot-test").hexdigest()}},
                "screen":{"width":config.screen_width,"height":config.screen_height},"screen_diagonal_inches":config.screen_diagonal_inches,
                "camera_position_screen_cm":list(config.camera_position_screen_cm)}
            (root/"conditioned-video-model.json").write_text(json.dumps(meta))
            with patch("opengazelink_pc.video_session.DATA_DIR",root):
                session=VideoSession(Camera(),config,None,root=root/"captures",backend=Backend(),purpose="prediction")
            try:
                now=time.monotonic()*1000
                event=dict(pc_ms=now,browser_ms=50,x=.5,y=.5,sync_rtt_ms=1,phase="pursuit",block="train-a",trial_id="train-a",split="train",trial_age_ms=50,display_frame_id=3,motion_profile="stop_reverse")
                session.add_events({"batch_id":"one","events":[event],"plan":[{"complete":"plan"}]})
                session.add_events({"batch_id":"one","events":[event]})
                self.assertEqual(1,session.events)
                event.update(pc_ms=now+1,split="test")
                with self.assertRaises(ValueError):session.add_events({"batch_id":"two","events":[event]})
                time.sleep(.03)
            finally:
                session.cancel()
            self.assertEqual(b"snapshot-test",(session.path/"base-model/base.pt").read_bytes())
            frames=[json.loads(line) for line in (session.path/"frames.jsonl").read_text().splitlines()]
            self.assertTrue(frames)
            self.assertIn("prediction_timing",frames[0])
            self.assertEqual("stop_reverse",json.loads((session.path/"stimulus.jsonl").read_text())["motion_profile"])


if __name__=="__main__":
    unittest.main()
