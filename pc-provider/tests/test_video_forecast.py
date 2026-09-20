import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
from torch import nn

from opengazelink_pc.config import ProviderConfig
from opengazelink_pc.video_forecast import ForecastHistory, LearnedForecast
from opengazelink_pc.video_forecast_network import ForecastNetwork
from opengazelink_pc.video_forecast_training import future_pairs


class BiasedHead(nn.Module):
    def forward(self, features, horizon):
        return torch.ones(features.shape[0], 2) * .04


class VideoForecastTest(unittest.TestCase):
    def update(self, predictor, x, time, horizon=85., reset=False):
        return predictor.update((x,400.),(x,400.),time,np.zeros(404),np.zeros(64),(1920,1080),horizon,.02,reset)

    def test_fixation_remains_exactly_stationary_even_with_biased_head(self):
        predictor=LearnedForecast(BiasedHead(),{})
        for i in range(20):
            point,state=self.update(predictor,500.,i*33.)
            self.assertEqual((500.,400.),point)
            self.assertEqual(0.,state["lead_distance_px"])

    def test_jump_gap_and_explicit_reset_remove_old_predictor_history(self):
        for stamp,x,forced in ((133.,900.,False),(400.,560.,False),(133.,560.,True)):
            predictor=LearnedForecast(BiasedHead(),{})
            for i in range(4):
                self.update(predictor,500.+i*15.,i*33.)
            point,state=self.update(predictor,x,stamp,reset=forced)
            self.assertEqual((x,400.),point)
            self.assertEqual(1,state["sample_count"])
            self.assertEqual("forecast_reset",state["mode"])

    def test_reversal_zero_horizon_and_lead_cap(self):
        predictor=LearnedForecast(BiasedHead(),{})
        for i in range(4):
            point,state=self.update(predictor,500.+i*15.,i*33.,horizon=300.)
        self.assertEqual(100.,state["horizon_ms"])
        self.assertLessEqual(state["lead_distance_px"],.02*np.hypot(1919,1079)+1e-7)
        point,state=self.update(predictor,530.,132.)
        self.assertEqual((530.,400.),point)
        point,state=self.update(predictor,545.,165.,horizon=0.)
        self.assertEqual((545.,400.),point)

    def test_future_label_windows_never_cross_reset_split_or_target_block(self):
        rows=[{"source_ms":i*33.,"block":"a" if i<6 else "b"} for i in range(12)]
        segment=[0]*4+[1]*8
        pairs=future_pairs(rows,segment,67.)
        self.assertTrue(pairs)
        for i,a,b,f in pairs:
            self.assertEqual(segment[i],segment[b])
            self.assertEqual(rows[i]["block"],rows[b]["block"])
            self.assertLess(i,b)
            self.assertAlmostEqual(rows[i]["source_ms"]+67.,rows[a]["source_ms"]*(1-f)+rows[b]["source_ms"]*f)

    def test_timestamp_restart_does_not_create_future_targets_from_another_epoch(self):
        rows=[{"source_ms":t,"block":"a"} for t in (100.,133.,166.,0.,33.,66.,99.,132.)]
        for i,a,b,f in future_pairs(rows,[0,0,0,1,1,1,1,1],67.):
            self.assertGreaterEqual(i,3)
            self.assertGreaterEqual(a,3)

    def test_torchscript_and_zero_horizon_match(self):
        torch.manual_seed(1)
        net=ForecastNetwork(np.zeros(483),np.ones(483)).eval()
        nn.init.normal_(net.net[-1].weight)
        net.motion_gain.data.fill_(.5)
        x=torch.randn(8,483)
        x[:,12:15]=1.
        horizon=torch.ones(8,1)*.85
        script=torch.jit.trace(net,(x[:1],horizon[:1]))
        torch.testing.assert_close(net(x,horizon),script(x,horizon))
        torch.testing.assert_close(net(x,torch.zeros_like(horizon)),torch.zeros(8,2))

    def test_model_mismatch_does_not_load_a_stale_forecast(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"conditioned-video-model.json"
            path.with_name("conditioned-video-forecast.json").write_text(json.dumps({
                "schema":"opengazelink-video-forecast-v1","accepted":True,"base_sha256":"old"}))
            self.assertIsNone(LearnedForecast.load_for(path,{"module_sha256":"new"}))

    def test_legacy_compensation_defaults_off(self):
        self.assertFalse(ProviderConfig().extrapolation_enabled)


if __name__=="__main__":
    unittest.main()
