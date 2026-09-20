import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np

from opengazelink_pc.config import ProviderConfig
from opengazelink_pc.video_session import VideoSession
from opengazelink_pc.video_dataset import align_frames
from opengazelink_pc.unified_capture import coverage_report, validate_plan
from opengazelink_pc.unified_prediction_training import (balanced_draw, selection_objective, quality_scores,
                                                       preserves_dynamic_accuracy, dynamic_accuracy_regressions)
from opengazelink_pc.prediction_dataset import event_timing
from opengazelink_pc.unified_calibration_training import train_calibration


def step(identity="train-a", duration=1800, profile="anchor"):
    return dict(trial_id=identity, block=identity, split=identity.split("-")[0], duration=duration, motion_profile=profile)


class UnifiedCalibrationTests(unittest.TestCase):
    def test_pause_and_discard_reset_alignment_and_settling(self):
        events=[]
        for ms in range(1000, 6001, 20):
            segment=1 if ms<2600 else 2
            events.append(dict(pc_ms=ms, x=.5, y=.5, phase="pause" if 2600<=ms<3500 else "anchor",
                               block="train-a", trial_id="train-a", split="train", capture_segment=segment,
                               visible=True, sync_rtt_ms=1))
        frames=[dict(index=i,source_ms=ms,pc_read_ms=ms,valid=True,input="a.npz",
                     timing={"source_capture_monotonic_ns":ms*1e6}) for i,ms in enumerate(range(1000,5980,40))]
        rows=align_frames(frames,events,discarded_segments=[1])
        self.assertTrue(all(not r["valid"] and not r["weight"] for r in rows if r["pc_ms"]<2600))
        self.assertTrue(all(not r["stimulus_supported"] for r in rows if 2600<=r["pc_ms"]<3500))
        resumed=next(r for r in rows if r["pc_ms"]>=3500)
        self.assertTrue(resumed["reset"])
        self.assertEqual(0,resumed["weight"])
        self.assertTrue(all(not r["weight"] for r in rows if 3500<=r["pc_ms"]<4200))
        self.assertTrue(coverage_report(rows,[step()])["ready"])

    def test_partial_attempts_cannot_be_stitched_to_pass_coverage(self):
        rows=[dict(trial_id="train-a",capture_segment=segment,valid=True,stimulus_supported=True,
                   pc_ms=segment*10000+i*30,weight=1) for segment in (1,2) for i in range(28)]
        report=coverage_report(rows,[step()])
        self.assertFalse(report["ready"])
        self.assertEqual(["train-a"],report["missing_trials"])
        # One genuinely complete trial passes; an unobserved region stays missing.
        rows.extend(dict(trial_id="train-a",capture_segment=3,valid=True,stimulus_supported=True,
                         pc_ms=30000+i*30,weight=1) for i in range(55))
        report=coverage_report(rows,[step(),step("test-missing")])
        self.assertEqual(["test-missing"],report["missing_trials"])

    def test_pause_stops_capture_and_resume_preserves_session(self):
        class Camera:
            def camera_model(self):return {}
            def latest_frame_timing(self, sequence):return {"source_capture_monotonic_ns":time.monotonic_ns()}
            def read_latest(self,sequence,timeout_s):
                time.sleep(.01)
                return True,np.zeros((8,8,3),np.uint8),time.monotonic()*1000,sequence+1
        class Backend:
            def predict(self,*args):return None
            def close(self):pass
        with tempfile.TemporaryDirectory() as directory:
            session=VideoSession(Camera(),ProviderConfig(),None,root=Path(directory),backend=Backend())
            try:
                deadline=time.monotonic()+2
                while session.frames<2 and time.monotonic()<deadline:time.sleep(.01)
                session.pause(1)
                time.sleep(.05)
                count=session.frames
                time.sleep(.12)
                self.assertEqual(count,session.frames)
                self.assertTrue(session.status()["active"])
                self.assertTrue((session.path/"raw-camera").exists())
                session._heartbeat=time.monotonic()-100
                session.resume()
                deadline=time.monotonic()+2
                while session.frames<=count and time.monotonic()<deadline:time.sleep(.01)
                self.assertGreater(session.frames,count)
                self.assertEqual("collecting",session.state)
            finally:session.cancel()
            self.assertEqual([1],json.loads((session.path/"session.json").read_text())["discarded_segments"])

    def test_camera_stall_fails_capture_instead_of_accepting_empty_trials(self):
        class Camera:
            def camera_model(self):return {}
            def latest_frame_timing(self, sequence):return {"source_capture_monotonic_ns":time.monotonic_ns()}
            def reported_mode(self):return {"error":"decoder disconnected"}
            def read_latest(self,sequence,timeout_s):
                time.sleep(.01)
                if sequence < 0:return True,np.zeros((8,8,3),np.uint8),time.monotonic()*1000,1
                return False,None,0,sequence
        class Backend:
            record_diagnostics=False
            last_diagnostics={}
            def predict(self,*args):return None
            def close(self):pass
        with tempfile.TemporaryDirectory() as directory, \
             patch("opengazelink_pc.video_session.FRAME_STALL_TIMEOUT_S",.05):
            session=VideoSession(Camera(),ProviderConfig(),None,root=Path(directory),backend=Backend())
            deadline=time.monotonic()+2
            while session.state!="failed" and time.monotonic()<deadline:time.sleep(.01)
            self.assertEqual("failed",session.state)
            self.assertIn("decoder disconnected",session.error)
            now=time.monotonic()*1000
            with self.assertRaises(RuntimeError):
                session.add_events({"events":[dict(pc_ms=now,browser_ms=now,x=.5,y=.5,sync_rtt_ms=1,
                    phase="anchor",block="train-a",visible=True)]})

    def test_training_sampler_balances_trials_and_never_draws_holdout(self):
        # 100x imbalance in recorded duration must not become 100x learning mass.
        data={"phase":np.zeros(1020,int),"trial":np.array(["long"]*1000+["short"]*10+["test"]*10),
              "frame":np.arange(1020),"stimulus_target":np.full((1020,2),.5)}
        draw=balanced_draw(data,np.arange(1020)<1010,np.random.default_rng(7),per_phase=2000)
        self.assertFalse(np.any(draw>=1010))
        self.assertEqual(1000,int(np.sum(draw<1000)))
        self.assertEqual(1000,int(np.sum(draw>=1000)))

    def test_absent_saccade_proxy_does_not_crash_objective(self):
        score={"mean_proxy_px":10,"p95_proxy_px":20,"groups":{"pursuit":{"mean_px":10}}}
        self.assertAlmostEqual(1.,selection_objective(score,score))

    def test_better_average_cannot_hide_operating_horizon_regression(self):
        baseline = dict(mean_proxy_px=100., p95_proxy_px=300., groups={},
                        horizons={'85': dict(samples=100, mean_proxy_px=80., p95_proxy_px=200., groups={})})
        candidate = dict(mean_proxy_px=95., p95_proxy_px=290., groups={},
                         horizons={'85': dict(samples=100, mean_proxy_px=81., p95_proxy_px=210., groups={})})
        self.assertFalse(preserves_dynamic_accuracy(candidate, baseline))
        reasons = dynamic_accuracy_regressions(candidate, baseline)
        self.assertEqual({'85ms'}, {r['scope'] for r in reasons})
        self.assertEqual(2, len(reasons))
        candidate['horizons']['85'].update(mean_proxy_px=79., p95_proxy_px=202.)
        self.assertTrue(preserves_dynamic_accuracy(candidate, baseline))

    def test_horizon_pursuit_regression_is_visible_even_when_total_improves(self):
        baseline = dict(mean_proxy_px=100., p95_proxy_px=300., groups={},
                        horizons={'85': dict(samples=100, mean_proxy_px=80., p95_proxy_px=200.,
                                             groups={'pursuit': dict(mean_px=30., p95_px=90.)})})
        candidate = dict(mean_proxy_px=95., p95_proxy_px=290., groups={},
                         horizons={'85': dict(samples=100, mean_proxy_px=79., p95_proxy_px=195.,
                                              groups={'pursuit': dict(mean_px=31., p95_px=90.)})})
        reasons = dynamic_accuracy_regressions(candidate, baseline)
        self.assertEqual(1, len(reasons))
        self.assertEqual('85ms/pursuit', reasons[0]['scope'])

    def test_sampler_keeps_exact_phase_and_horizon_budgets_with_sparse_cells(self):
        records = [(phase, h, trial, cell) for phase in range(3)
                   for h in (0., .085, .25) for trial in range(1 + phase * 3)
                   for cell in range(1 + trial)]
        phase, horizon, trial, cell = map(np.array, zip(*records))
        data = dict(phase=phase, horizon=horizon[:, None], trial=trial,
                    frame=np.arange(len(records)), phase_conf=np.ones(len(records)),
                    stimulus_target=np.c_[(cell % 3 + .5) / 3, (cell // 3 + .5) / 3])
        draw = balanced_draw(data, np.ones(len(records), bool), np.random.default_rng(7), per_phase=30)
        self.assertEqual(90, len(draw))
        for p in range(3):
            for h in (0., .085, .25):
                self.assertEqual(10, int(((phase[draw] == p) & (horizon[draw] == h)).sum()))
        # Even with fewer draws than cells, counts must not grow to cover every cell.
        draw = balanced_draw(data, np.ones(len(records), bool), np.random.default_rng(7), per_phase=2)
        self.assertEqual(6, len(draw))

    def test_event_timing_uses_source_time_and_never_crosses_segments(self):
        phase = np.array([0, 2, 2, 0, 0, 2, 2])
        times = np.array([0, 8, 42, 50, 60, 0, 9])
        segments = np.array([0, 0, 0, 0, 0, 1, 1])
        age, upcoming = event_timing(phase, times, segments)
        np.testing.assert_array_equal(age, [-1, 0, 34, -1, -1, 0, 9])
        self.assertEqual(8, upcoming[0])
        self.assertTrue(np.isinf(upcoming[1:]).all())

    def test_event_scores_partition_every_85ms_sample_without_changing_total(self):
        n = 5
        data = dict(frame=np.arange(n), horizon=np.full((n, 1), .085),
                    phase=np.array([2, 2, 0, 1, 0]), delta=np.arange(10).reshape(n, 2) / 100,
                    anchor=np.zeros(n, bool), current_anchor=np.zeros(n, bool))
        frames = dict(event_age_ms=np.array([0, 30, -1, -1, -1]),
                      next_event_ms=np.array([100, 100, 85, 86, np.inf]))
        pred = np.zeros((n, 2))
        score = quality_scores(pred, data, frames, np.ones(n, bool), np.array([100, 80]))
        self.assertEqual({'16', '33', '50', '67', '85', '100', '125', '150', '175', '200', '225', '250'},
                         set(score['horizons']))
        groups = score['event_timing_85ms']
        self.assertEqual([1, 1, 1, 2], [g['samples'] for g in groups.values()])
        aggregate = sum(g['samples'] * g['mean_proxy_px'] for g in groups.values()) / n
        self.assertAlmostEqual(score['mean_proxy_px'], aggregate)

    def test_diagnostic_feature_histories_are_causal_across_clock_resets(self):
        from tools.audit_prediction_observability import histories
        points = np.array([[0, 0], [1, 2], [90, 90], [5, 6], [8, 9]], dtype=float)
        times = np.array([0, 10, 20, 0, 10])
        segments = np.array([0, 0, 0, 1, 1])
        ids = np.array([1, 3])
        result = histories(points, times, segments, ids, np.array([0, 5, 25]))
        np.testing.assert_allclose(result[0], [[1, 2], [.5, 1], [0, 0]])
        np.testing.assert_allclose(result[1], [[5, 6], [5, 6], [5, 6]])
        points[2] = -900
        np.testing.assert_array_equal(result, histories(points, times, segments, ids, np.array([0, 5, 25])))

    def test_unified_session_dispatches_staged_training(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);manifest=root/"replay.jsonl";manifest.write_text("")
            session=VideoSession.__new__(VideoSession)
            session.path=root;session.purpose="unified";session.state="training";session.metadata={}
            session.registry=SimpleNamespace(clear=lambda:None);session._persist=lambda:None
            with patch("opengazelink_pc.video_replay.prepare_replay",return_value={"frames":1,"valid_frames":1,"manifest":str(manifest)}), \
                 patch("opengazelink_pc.unified_calibration_training.train_calibration",return_value={"published":True}) as train:
                session._train()
        train.assert_called_once()
        self.assertEqual("unified_complete",session.phase)

    def test_plan_contract_rejects_overlapping_or_mismatched_splits(self):
        plan=[step(f"{split}-{i}", profile="anchor" if i<5 else "line") for split in ("train","validation","test") for i in range(11)]
        validate_plan(plan)
        plan[1]=plan[0]
        with self.assertRaises(ValueError):validate_plan(plan)

    def test_event_plan_contract_rejects_invalid_points_and_times(self):
        plan=[]
        for split in ("train","validation","test"):
            for i in range(3 if split != "train" else 6):
                identity=f"{split}-event-{i}"
                plan.append(dict(trial_id=identity,block=identity,split=split,
                    calibration_stage="events_v1",motion_profile="jump",duration=4200,
                    points=[[.5,.5],[.7,.5],[.3,.5]],changes=[1200,2700]))
        validate_plan(plan)
        plan[0]["points"][1][0]=1.2
        with self.assertRaises(ValueError):validate_plan(plan)
        plan[0]["points"][1][0]=.7
        plan[0]["changes"]=[1200,4000]
        with self.assertRaises(ValueError):validate_plan(plan)

    def test_head_varied_spatial_plan_requires_distinct_crossed_pairs(self):
        plan=[]
        counts={"train":4,"validation":2,"test":2}
        poses=(("yaw","head_left","head_right"),("pitch","head_up","head_down"))
        for split,pairs in counts.items():
            split_offset={"train":0.,"validation":.01,"test":.02}[split]
            for pair_index in range(pairs):
                a=[.1+.04*pair_index,.2+.03*pair_index+split_offset]
                b=[.9-.04*pair_index,.8-.03*pair_index+split_offset]
                axis,start,end=poses[pair_index%2]
                for rail_index in (0,1):
                    identity=f"{split}-spatial-{len([s for s in plan if s['split']==split])}"
                    offset=[0,.05] if axis=="yaw" else [.05,0]
                    point=[v+offset[i]*rail_index for i,v in enumerate(a)]
                    finish=[v+offset[i]*rail_index for i,v in enumerate(b)]
                    plan.append(dict(trial_id=identity,block=identity,split=split,
                        calibration_stage="spatial_v1",plan_version=5,motion_profile="line",duration=5200,
                        point=point,end=finish,pair_id=f"{split}-pair-{pair_index}",head_axis=axis,
                        head_cue_family="relative",head_mode="follow" if rail_index==0 else "counter",
                        head_start=start if rail_index==0 else end,head_end=end if rail_index==0 else start))
        validate_plan(plan)
        plan[1]["head_start"],plan[1]["head_end"] = plan[1]["head_end"],plan[1]["head_start"]
        with self.assertRaises(ValueError):validate_plan(plan)

    def test_pipeline_reuses_incumbent_when_spatial_candidate_not_better(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);session=root/"session";session.mkdir()
            model={"variants":{"conditioned_video":{"module_file":"old.pt","module_sha256":"unchanged"}}}
            (root/"conditioned-video-model.json").write_text(json.dumps(model))
            (root/"old.pt").write_bytes(b"old-spatial")
            spatial={"published":False,"publication_reason":"keep_existing_model","training_directory":str(root)}
            with patch("opengazelink_pc.unified_calibration_training.DATA_DIR",root), \
                 patch("opengazelink_pc.video_training.train_session",return_value=spatial) as train, \
                 patch("opengazelink_pc.unified_prediction_training.train_unified",return_value={"accepted":False,"published":False}) as prediction:
                result=train_calibration(session,progress=lambda _:None)
            train.assert_called_once()
            self.assertFalse(result["published"])
            self.assertEqual("frozen_spatial_trajectory",result["prediction"]["mode"])
            self.assertFalse(train.call_args.kwargs["publish"])
            self.assertFalse(train.call_args.kwargs["train_temporal"])
            self.assertEqual(60,prediction.call_args.kwargs["epochs"])
            self.assertEqual(model,json.loads((root/"conditioned-video-model.json").read_text()))

    def test_temporal_publication_can_succeed_without_replacing_spatial_model(self):
        for publish in (False,True):
            with self.subTest(publish=publish), tempfile.TemporaryDirectory() as folder:
                root=Path(folder);session=root/'session';session.mkdir()
                model={'variants':{'conditioned_video':{'module_file':'old.pt','module_sha256':'unchanged'}}}
                active=root/'conditioned-video-model.json';active.write_text(json.dumps(model))
                (root/'old.pt').write_bytes(b'old-spatial')
                spatial={'published':False,'publication_reason':'keep_existing_model','training_directory':str(root)}
                with patch('opengazelink_pc.unified_calibration_training.DATA_DIR',root), \
                     patch('opengazelink_pc.video_training.train_session',return_value=spatial), \
                     patch('opengazelink_pc.unified_prediction_training.train_unified',return_value={'accepted':True,'published':False}) as train, \
                     patch('opengazelink_pc.unified_prediction_training.publish_unified') as release:
                    result=train_calibration(session,prediction_epochs=17,publish=publish,progress=lambda _:None)
                self.assertEqual(17,train.call_args.kwargs['epochs'])
                self.assertEqual(publish,result['prediction']['published'])
                self.assertEqual(publish,release.called)
                if publish:self.assertEqual(active,release.call_args.args[1])
                self.assertEqual(model,json.loads(active.read_text()))


if __name__=="__main__":unittest.main()
