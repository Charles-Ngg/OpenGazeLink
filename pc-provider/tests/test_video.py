from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import time
import unittest
from types import SimpleNamespace
from urllib.request import urlopen

import numpy as np
import torch
from torch import nn

from opengazelink_pc.config import ProviderConfig
from opengazelink_pc.video_dataset import align_frames, roundtrip_capture_time
from opengazelink_pc.prediction_timing import PredictionClock
from opengazelink_pc.video_network import VideoAdapter, VideoInference
from opengazelink_pc.video_session import SCHEMA, VideoSession
from opengazelink_pc.video_training import (camera_profile, candidate_is_safe, coordinate_loss,
    balance_joint_future_pairs, joint_future_metrics, joint_future_pairs, joint_motion_teacher,
    joint_selection_score, train_session)
from opengazelink_pc.video_training import joint_event_teacher, joint_motion_delta
from opengazelink_pc.video_archive import RawVideoArchive
from opengazelink_pc.control_server import ControlServer


class FakeBase(nn.Module):
    def forward(self, images, geometry):
        return torch.tensor([[0., 0., 1.], [0., 0., 1.]]), torch.tensor([.5, .5])


def event(ms, **kwargs):
    return dict(pc_ms=ms, x=.5, y=.5, phase="anchor", block="train-anchor", visible=True, sync_rtt_ms=2., **kwargs)


def frame(ms, index=0, valid=True):
    return dict(index=index, source_ms=ms, pc_read_ms=ms, valid=valid, input="input.npz",
                timing={"source_capture_monotonic_ns": ms * 1e6})


class VideoTests(unittest.TestCase):
    def test_event_runtime_matches_evaluation_at_high_uncertainty(self):
        from opengazelink_pc.conditioned_eye_model import EmbeddedJointForecast
        forecast = EmbeddedJointForecast({"max_horizon_ms":100.})
        motion = np.array([1.,.2,0.,0.,-4.,2.,4.,.1,.02,.04,.1])
        forecast.set_motion(motion)
        origin = np.array([500.,500.]); screen = np.array([3840.,2160.])
        point, _ = forecast.update(origin, origin, 1000., None, None, screen, 85., .3)
        np.testing.assert_allclose(np.array(point)-origin, joint_motion_delta(motion,.085)*(screen-1),atol=1e-8)

    def test_event_motion_delta_uses_phase_landing_and_uncertainty(self):
        motion=np.array([1.,0.,0.,0.,-4.,2.,4.,.1,.0,.04,.01],np.float32)
        delta=joint_motion_delta(motion,.085)
        self.assertGreater(float(delta[0]),.02)
        self.assertLess(float(np.linalg.norm(delta)),.2)

    def test_detached_motion_state_keeps_future_head_from_recurrent_gradient(self):
        torch.manual_seed(44)
        adapter=VideoAdapter(148,detach_motion_state=True)
        with torch.no_grad():
            adapter.forecast[-1].weight.fill_(.01)
        features=torch.randn(1,148,requires_grad=True)
        hidden=torch.randn(1,64,requires_grad=True)
        previous=torch.zeros(1,148)
        _,_,_,_,motion=adapter(features,torch.tensor([[8.333]]),torch.zeros(1,1),hidden,previous)
        motion.sum().backward()
        self.assertIsNone(hidden.grad)
        self.assertIsNotNone(features.grad)

    def test_phone_camera_profile_distinguishes_capture_geometry(self):
        first = camera_profile({"width":720, "height":1280, "fx":905., "fy":905.,
            "sourceMetadata":{"cameraId":"0", "effectiveStreamCrop":{"width":4096,"height":2304}}})
        second = camera_profile({"width":720, "height":960, "fx":679., "fy":679.,
            "sourceMetadata":{"cameraId":"0", "effectiveStreamCrop":{"width":4096,"height":3072}}})
        self.assertNotEqual(first, second)

    def test_video_export_retains_fit_time_geometry_reference(self):
        class GeometryBase(nn.Module):
            def forward(self, images, geometry):
                direction = torch.stack((geometry[:, 67], torch.zeros(2), torch.ones(2)), 1)
                return nn.functional.normalize(direction, dim=1), torch.tensor([.5, .5])
        model = VideoInference(
            torch.jit.script(GeometryBase()), VideoAdapter(), np.zeros(70), np.ones(70),
            geometry_reference_indices=[67], geometry_reference_values=[0.25],
        ).eval()
        output = model(torch.zeros(2,2,36,64,dtype=torch.uint8), torch.full((2,70), 9.),
            torch.ones(1,1)*33, torch.ones(1,1), torch.zeros(1,64), torch.zeros(1,148))
        torch.testing.assert_close(output[0][:, 0], torch.full((2,), 0.25 / np.sqrt(1.0625)))

    def test_offline_roundtrip_matches_online_and_rejects_stale_or_invalid_probe(self):
        timing=dict(phone_sensor_time_ns=1_000_000_000,phone_send_time_ns=1_060_000_000,
                    pc_first_packet_monotonic_ns=2_003_000_000,phone_to_pc_offset_ns=940_000_000,
                    clock_probe_uncertainty_ms=2.,clock_probe_age_ms=100.)
        online=PredictionClock().observe(1000.,timing,2010.)
        self.assertAlmostEqual(roundtrip_capture_time(timing),online['source_pc_ms_proxy'])
        frames=[dict(source_ms=1000.,pc_read_ms=2003.,valid=True,input='a.npz',timing=timing)]
        rows=align_frames(frames,[event(ms) for ms in range(1900,2101,20)])
        self.assertEqual(rows[0]['clock'],'phone_roundtrip_alignment')
        self.assertEqual(rows[0]['clock_uncertainty_ms'],2.)
        for updates in [dict(clock_probe_age_ms=5001),dict(phone_to_pc_offset_ns=float('nan')),
                        dict(clock_probe_uncertainty_ms=21)]:
            altered=dict(timing,**updates)
            self.assertIsNone(roundtrip_capture_time(altered))
            rows=align_frames([dict(frames[0],timing=altered)],[event(ms) for ms in range(1900,2101,20)])
            self.assertEqual(rows[0]['clock'],'phone_minimum_transit_proxy')

    def test_selection_allows_bounded_tail_tradeoff_but_rejects_regressions(self):
        baseline = {"median_px": 186., "mean_px": 415., "p95_px": 2308.}
        self.assertTrue(candidate_is_safe(
            {"median_px": 85., "mean_px": 390., "p95_px": 2477.}, baseline, 186.,
        ))
        self.assertFalse(candidate_is_safe(
            {"median_px": 85., "mean_px": 416., "p95_px": 2477.}, baseline, 186.,
        ))
        self.assertFalse(candidate_is_safe(
            {"median_px": 85., "mean_px": 390., "p95_px": 2600.}, baseline, 186.,
        ))
        self.assertFalse(candidate_is_safe(
            {"median_px": 190., "mean_px": 390., "p95_px": 2200.}, baseline, 186.,
        ))

    def test_capture_keeps_invalid_frames_and_cancellation_preserves_files(self):
        class Camera:
            def read_latest(self, sequence, timeout_s):
                time.sleep(.005)
                return True, np.zeros((8,8,3), np.uint8), time.monotonic()*1000, sequence + 1
            def latest_frame_timing(self, sequence):
                return {"source_capture_monotonic_ns": time.monotonic_ns()}
            def camera_model(self):
                return {}
        class Backend:
            def predict(self, *args):
                return None
            def close(self):
                pass
        with tempfile.TemporaryDirectory() as folder:
            session = VideoSession(Camera(), ProviderConfig(), None, root=Path(folder), backend=Backend())
            now = time.monotonic()*1000
            sample = dict(pc_ms=now,browser_ms=123.,x=.5,y=.5,sync_rtt_ms=2.,phase="anchor",block="train",visible=True)
            session.add_events({"events":[sample]})
            with self.assertRaises(ValueError):
                session.add_events({"events":[sample]})
            deadline = time.monotonic()+2
            while session.frames < 2 and time.monotonic()<deadline:
                time.sleep(.01)
            session.cancel()
            self.assertEqual("cancelled",session.state)
            rows=(session.path/"frames.jsonl").read_text().splitlines()
            self.assertGreaterEqual(len(rows),2)
            self.assertTrue(all(json.loads(row)["valid"] for row in rows))
            self.assertTrue((session.path/"stimulus.jsonl").is_file())
            self.assertTrue((session.path/"raw-camera").exists())
            decisions=[json.loads(line) for line in (session.path/"event-batches.jsonl").read_text().splitlines()]
            self.assertEqual(2,len(decisions))
            self.assertTrue(decisions[1]["decision"].startswith("rejected"))

    def test_capture_does_not_run_mediapipe_or_write_derived_inputs(self):
        class Camera:
            def read_latest(self, sequence, timeout_s):
                time.sleep(.02)
                return True, np.zeros((720,1280,3), np.uint8), time.monotonic()*1000, sequence + 1
            def latest_frame_timing(self, sequence):
                return {"source_capture_monotonic_ns": time.monotonic_ns()}
            def camera_model(self):
                return {"fx":900.,"fy":900.,"cx":640.,"cy":360.}
        class Backend:
            def predict(self,*args): raise AssertionError("MediaPipe must not run during capture")
            def close(self):pass
        with tempfile.TemporaryDirectory() as folder:
            session=VideoSession(Camera(),ProviderConfig(),None,root=Path(folder),backend=Backend())
            deadline=time.monotonic()+2
            while session.frames<1 and time.monotonic()<deadline:time.sleep(.01)
            session.cancel()
            self.assertTrue((session.path/"raw-camera").exists())
            frame_row=json.loads((session.path/"frames.jsonl").read_text().splitlines()[0])
            self.assertTrue(frame_row["capture_only"])
            self.assertEqual("raw-camera", frame_row["input"])
            self.assertFalse(list((session.path/"inputs").glob("*.npz")))

    def test_raw_archive_preserves_exact_bytes_and_reports_overload(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/"raw"
            archive=RawVideoArchive(path,max_pending_bytes=1024)
            payload=bytes(range(256))
            archive.submit("udp_packet",{"pc_receive_ms":123.},payload)
            archive.submit("udp_packet",{},bytes(2048))
            archive.close()
            self.assertEqual(payload,(path/"data.bin").read_bytes())
            self.assertTrue(archive.error)
            self.assertTrue(json.loads((path/"archive.json").read_text(encoding="utf-8"))["error"])

    def test_raw_replay_recovers_frames_rejected_by_live_decoder(self):
        from opengazelink_pc.camera import HEADER, MAGIC, FORMAT_NV21
        from tools.replay_video_archive import recover
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)
            archive=RawVideoArchive(path/"raw")
            yuv=bytes([128]*24)
            for seq in (5,3):
                # The older frame would be rejected by the live latest-frame policy.
                packet=HEADER.pack(MAGIC,1,HEADER.size,seq,0,1,4,4,FORMAT_NV21,0,seq*100,seq*100+5,len(yuv))+yuv
                archive.submit("udp_packet",{"pc_receive_ms":1000.+seq},packet)
            archive.submit("udp_packet",{"pc_receive_ms":1009.},b"malformed")
            archive.close()
            result=recover(path/"raw",path/"recovered")
            self.assertEqual(2,result["recovered_frames"])
            self.assertEqual(2,len(list((path/"recovered").glob("*.png"))))
            self.assertIn("non_frame_packet_retained",(path/"recovered/recovery.jsonl").read_text())

    def test_rail_labels_tolerate_timing_along_rail_but_constrain_normal(self):
        events=[event(ms) for ms in range(1000,3001,20)]
        for e in events:
            e.update(phase="pursuit",x=.1+(e["pc_ms"]-1000)/2500,y=.3,
                     rail={"a":[.1,.3],"b":[.9,.3]},motion_age_ms=e["pc_ms"]-1000)
        row=align_frames([frame(2000)],events)[0]
        self.assertEqual(.25,row["weight"])
        constraint=row["constraint"]
        self.assertIsNotNone(constraint)
        label=torch.tensor([row["target"]])
        inside=torch.tensor([[(constraint["lower"]+constraint["upper"])/2,.3]])
        self.assertEqual(0.,coordinate_loss(inside,label,constraint).item())
        off=inside+torch.tensor([[0.,.2]])
        self.assertGreater(coordinate_loss(off,label,constraint).item(),.1)
        events[48]["visible"]=False
        rejected=align_frames([frame(2000)],events)[0]
        self.assertIsNone(rejected["constraint"])
        self.assertEqual(0.,rejected["weight"])

    def test_phone_clock_restart_uses_separate_clock_offset(self):
        frames = [frame(10,0),frame(43,1),frame(1,2)]
        for f,pc in zip(frames,[1800,1833,2100]):
            f["timing"]={"phone_sensor_time_ns":f["source_ms"]*1e6,
                         "phone_send_time_ns":(f["source_ms"]+5)*1e6,
                         "pc_first_packet_monotonic_ns":(pc+5)*1e6}
        rows=align_frames(frames,[event(ms) for ms in range(1000,2300,20)])
        self.assertEqual([1800.,1833.,2100.],[row["pc_ms"] for row in rows])
        self.assertTrue(rows[-1]["reset"])

    def test_control_server_serves_the_video_entry_script(self):
        server = ControlServer(SimpleNamespace(config=ProviderConfig(control_port=0)))
        server.start(open_browser=False)
        try:
            port = server.server.server_address[1]
            with urlopen(f"http://127.0.0.1:{port}/video.js") as response:
                self.assertEqual(200, response.status)
                self.assertIn(b"/api/calibration/unified/start", response.read())
        finally:
            server.close()

    def test_transitions_are_unlabelled_and_do_not_force_memory_reset(self):
        events = [event(ms) for ms in range(1000, 2201, 20)]
        for item in events[45:]:
            item.update(x=.9, block="new-anchor")
        rows = align_frames([frame(1800, 0), frame(1900, 1), frame(1930, 2)], events)
        self.assertEqual(1., rows[0]["weight"])
        self.assertEqual(0., rows[1]["weight"])
        self.assertFalse(rows[1]["reset"])
        self.assertFalse(rows[2]["reset"])

    def test_blink_and_gap_reset_following_valid_frame(self):
        events = [event(ms) for ms in range(1000, 3001, 20)]
        rows = align_frames([frame(1800, 0), frame(1833, 1, False), frame(1866, 2), frame(2300, 3)], events)
        self.assertTrue(all(row["reset"] for row in rows))
        self.assertEqual(0., rows[1]["weight"])

    def test_missing_display_telemetry_does_not_extrapolate_labels(self):
        rows = align_frames([frame(2000)], [event(1000), event(2500)])
        self.assertEqual(0., rows[0]["weight"])

    def test_pursuit_requires_explicit_lag_and_never_crosses_blocks(self):
        events = [event(ms) for ms in range(1000, 2201, 20)]
        for item in events:
            item.update(phase="pursuit", x=(item["pc_ms"] - 1000) / 1500)
        self.assertEqual(0., align_frames([frame(1800)], events)[0]["weight"])
        row = align_frames([frame(1800)], events, pursuit_lag_ms=100)[0]
        self.assertAlmostEqual(700 / 1500, row["target"][0])
        self.assertEqual(.1, row["weight"])

    def test_adapter_is_causal_and_explicit_reset_removes_history(self):
        torch.manual_seed(2)
        adapter = VideoAdapter().eval()
        nn.init.normal_(adapter.residual.weight)
        features = torch.randn(1, 148)
        dt, reset = torch.ones(1, 1) * 33, torch.ones(1, 1)
        a = adapter(features, dt, reset, torch.randn(1, 64), torch.randn(1, 148))
        b = adapter(features, dt, reset, torch.zeros(1, 64), torch.zeros(1, 148))
        torch.testing.assert_close(a[0], b[0])
        torch.testing.assert_close(a[1], b[1])

    def test_adapter_signed_frame_delta_reaches_shared_motion_state(self):
        torch.manual_seed(17)
        adapter=VideoAdapter().eval()
        current=torch.zeros(1,148)
        previous_positive=torch.zeros(1,148);previous_positive[0,0]=-.01
        previous_negative=torch.zeros(1,148);previous_negative[0,0]=.01
        args=(current,torch.ones(1,1)*8.333,torch.zeros(1,1),torch.zeros(1,64))
        positive=adapter(*args,previous_positive)
        negative=adapter(*args,previous_negative)
        self.assertFalse(torch.allclose(positive[1],negative[1]))

    def test_joint_future_pairs_never_cross_trial_or_split(self):
        rows=[]
        for i in range(8):
            rows.append(dict(index=i,source_ms=i*33.,reset=i in (0,4),split="train" if i<4 else "validation",
                             trial_id="a" if i<4 else "b",capture_segment=0))
        pairs=joint_future_pairs(rows,np.ones(8),horizons=(67.,))
        self.assertTrue(pairs[0])
        self.assertFalse(pairs[3])
        self.assertTrue(all(rows[a]["split"]==rows[b]["split"] and rows[a]["trial_id"]==rows[b]["trial_id"]
                            for a,values in pairs.items() for b,_,_ in values))

    def test_joint_future_pairs_keep_unlabelled_motion_and_balance_classes(self):
        rows=[dict(index=i,source_ms=i*33.,reset=i==0,split="train",trial_id="a",capture_segment=0)
              for i in range(10)]
        pairs=joint_future_pairs(rows,np.zeros(10),horizons=(100.,))
        self.assertTrue(pairs[0])
        teacher=np.zeros((10,2),np.float32)
        teacher[5:,0]=np.arange(5,dtype=np.float32)*.05
        balanced,audit=balance_joint_future_pairs(pairs,teacher,np.asarray([101.,101.]),np.arange(10))
        self.assertGreater(audit["100"]["stationary"],0)
        self.assertGreater(audit["100"]["moving"],0)
        weights=[weight for values in balanced.values() for _,_,weight in values]
        self.assertGreater(max(weights),min(weights))

    def test_joint_motion_teacher_does_not_smooth_across_trial_boundary(self):
        rows=[dict(index=i,source_ms=i*8.333,reset=i in (0,5),split="train",
                   trial_id="a" if i<5 else "b",capture_segment=0) for i in range(10)]
        points=np.asarray([[.1+i*.01,.5] for i in range(5)]+[[.8+i*.01,.5] for i in range(5)],np.float32)
        teacher,_=joint_motion_teacher(points,rows,np.ones(10))
        self.assertLess(teacher[4,0],.2)
        self.assertGreater(teacher[5,0],.7)

    def test_joint_future_metric_uses_future_label_not_current_label(self):
        # The source point is x=.15 and the future point is x=.40.  A velocity
        # of 2.5 normalized units/s must land exactly on the t+100 ms target.
        current=np.asarray([[.15,.10]],dtype=np.float32)
        motion=np.asarray([[2.5,0.,0.,0.]],dtype=np.float32)
        target=np.asarray([[.15,.10],[.40,.10]],dtype=np.float32)
        metrics=joint_future_metrics(current,motion,np.asarray([0]),{0:[(1,.1,1.)]},target,np.asarray([101.,101.]))
        self.assertAlmostEqual(0.,metrics["mean_px"],places=5)
        self.assertAlmostEqual(25.,metrics["hold_mean_px"],places=5)
        self.assertTrue(metrics["improves_hold"])
        self.assertAlmostEqual(0.,metrics["displacement_mean_px"],places=5)
        self.assertTrue(metrics["displacement_improves_zero"])
        self.assertAlmostEqual(0.,metrics["by_horizon_ms"]["100"]["future_mean_px"],places=5)
        self.assertTrue(metrics["moving_improves_hold"])
        self.assertEqual("y_hat(t+h) versus frozen_replay_teacher(t+h)",metrics["evaluation"])

    def test_joint_score_allows_small_current_tradeoff_for_future_gain(self):
        baseline={"median_px":100.,"mean_px":120.,"p95_px":250.}
        current={"median_px":101.,"mean_px":121.,"p95_px":252.}
        future={"samples":100,"mean_px":70.,"hold_mean_px":100.,"improves_hold":True}
        self.assertLess(joint_selection_score(current,baseline,future),1.)
        current["median_px"]=104.
        self.assertEqual(float("inf"),joint_selection_score(current,baseline,future))

    def test_scripted_adapter_matches_multiple_stream_steps(self):
        model = VideoInference(torch.jit.script(FakeBase()), VideoAdapter(), np.zeros(70), np.ones(70)).eval()
        args = (torch.zeros(2, 2, 36, 64, dtype=torch.uint8), torch.zeros(2,70), torch.ones(1,1)*33,
                torch.ones(1,1), torch.zeros(1,64), torch.zeros(1,148))
        script = torch.jit.trace(model, args, strict=False)
        h, p = torch.zeros(1, 64), torch.zeros(1, 148)
        hs, ps = h.clone(), p.clone()
        for i in range(8):
            image, geometry = torch.zeros(2, 2, 36, 64, dtype=torch.uint8), torch.randn(2, 70)
            dt, reset = torch.ones(1, 1) * 33, torch.tensor([[float(i in (0, 4))]])
            a = model(image, geometry, dt, reset, h, p)
            b = script(image, geometry, dt, reset, hs, ps)
            for left, right in zip(a, b):
                torch.testing.assert_close(left, right)
            h, p, hs, ps = a[2], a[3], b[2], b[3]

    def test_staged_training_export_on_synthetic_continuous_session(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            config = ProviderConfig()
            config.camera_offset_y_cm = 0.
            config.geometry_configured = True
            (path / "session.json").write_text(json.dumps({"schema":SCHEMA, "config":asdict(config)}))
            torch.jit.script(FakeBase()).save(str(path / "base.pt"))
            metadata = {"schema":"opengazelink-conditioned-eye-v1", "preprocessing":{"landmarker_backend":"tasks"},
                        "variants":{"conditioned_binocular":{"raw_inputs":True, "module_file":"base.pt",
                        "normalization":{"mean":[0.]*70, "scale":[1.]*70}}}}
            (path / "base.json").write_text(json.dumps(metadata))
            images = np.zeros((2, 2, 2, 36, 64), np.uint8)
            np.savez_compressed(path / "input.npz", images=images, head=np.zeros((2,10)), points=np.zeros((2,2,52)),
                                crop=np.zeros((2,8)), rotation=np.tile(np.diag([-1.,1.,-1.]),(2,1,1)),
                                center=np.array([[0.,0.,50.],[0.,0.,50.]]))
            events = [event(ms) for ms in range(1000, 10001, 20)]
            for item in events:
                if item["pc_ms"] >= 6000:
                    item["block"] = "validation-anchor"
                if item["pc_ms"] >= 8500:
                    item["block"] = "test-anchor"
                if 2500<=item["pc_ms"]<3500:
                    item.update(block="train-rail",phase="pursuit",x=.1+(item["pc_ms"]-2500)*.0008,
                                rail={"a":[.1,.5],"b":[.9,.5]},motion_age_ms=item["pc_ms"]-2500)
            frames = [frame(ms, i) for i, ms in enumerate(range(1000, 9950, 40))]
            for name, rows in (("frames", frames), ("stimulus",events)):
                (path / f"{name}.jsonl").write_text("".join(json.dumps(row)+"\n" for row in rows))
            report = train_session(path, base_path=path/"base.json", epochs=1, publish=False)
            self.assertFalse(report["published"])
            self.assertEqual(1, len(report["history"]))
            self.assertTrue((path/"conditioned-video-model.json").is_file())
            self.assertIn(report["selected_stage"], ("personal_spatial", "joint"))
            self.assertGreater(report["rail_constraint_frames"],0)
            self.assertIn("independent_test",report)
            run=Path(report["training_directory"])
            with np.load(run/"training-inputs.npz") as arrays:
                self.assertFalse(np.any(arrays["train"] & arrays["test"]))
                self.assertFalse(np.any(arrays["selection"] & arrays["test"]))
            self.assertTrue((run/"alignment.jsonl").is_file())
            self.assertTrue((run/"joint-001.pt").is_file())
            frozen_report=train_session(path,base_path=path/"base.json",epochs=1,publish=False,train_temporal=False)
            self.assertEqual([],frozen_report['history'])
            self.assertFalse(frozen_report['temporal_training_enabled'])
            exported=json.loads((path/'conditioned-video-model.json').read_text(encoding='utf8'))
            self.assertFalse(exported['variants']['conditioned_video']['joint_prediction'])
            self.assertEqual(2,len(list((path/"training-runs").iterdir())))
            self.assertTrue((run/"joint-001.pt").is_file())


if __name__ == "__main__":
    unittest.main()
