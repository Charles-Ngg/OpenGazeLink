import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch

from opengazelink_pc.video_forecast_training import replay


class SixOutputVideo(torch.nn.Module):
    def forward(self, images, geometry, dt, reset, hidden, previous):
        return (torch.tensor([[0.,0.,1.],[0.,0.,1.]]), torch.tensor([.5,.5]),
                hidden, previous, torch.zeros(1,1), torch.zeros(1,4))


class ForecastReplayTests(unittest.TestCase):
    def test_replay_writes_artifacts_and_supports_embedded_motion_output(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); output=root/'replay'; output.mkdir()
            cfg=dict(screen_width=1920,screen_height=1080,screen_diagonal_inches=27.,
                     camera_offset_x_cm=0.,camera_offset_y_cm=0.,camera_offset_z_cm=0.,
                     one_euro_enabled=False,one_euro_min_cutoff=1.5,one_euro_beta=2.5,
                     one_euro_derivative_cutoff=1.)
            (root/'session.json').write_text(json.dumps({'config':cfg}))
            module=root/'base.pt';module.write_bytes(b'mocked-module')
            meta=dict(screen={'width':1920,'height':1080},screen_diagonal_inches=27.,
                      camera_position_screen_cm=[0.,0.,0.],variants={'conditioned_video':{
                          'feature_dim':404,'module_file':module.name,
                          'module_sha256':hashlib.sha256(module.read_bytes()).hexdigest()}})
            metadata=root/'model.json';metadata.write_text(json.dumps(meta))
            np.savez(root/'input.npz',images=np.zeros((2,2,2,36,64),np.uint8),
                     head=np.zeros((2,10)),points=np.zeros((2,2,52)),crop=np.zeros((2,8)),
                     rotation=np.tile(np.diag([-1.,1.,-1.]),(2,1,1)),center=np.array([[0.,0.,50.]]*2))
            rows=[dict(index=i,source_ms=i*8.333,dt_ms=8.333,valid=True,input='input.npz',
                       block='train-anchor',reset=i==0) for i in range(8)]
            with patch('opengazelink_pc.video_forecast_training.read_jsonl',return_value=[]), \
                 patch('opengazelink_pc.video_forecast_training.align_frames',return_value=rows), \
                 patch('torch.jit.load',return_value=SixOutputVideo()):
                values,actual,_,_=replay(root,metadata,output,instantaneous=True)
            self.assertEqual((8,2),values['instantaneous'].shape)
            np.testing.assert_allclose(values['raw'],values['instantaneous'])
            self.assertTrue((output/'replay.npz').is_file())
            self.assertEqual(8,len((output/'alignment.jsonl').read_text().splitlines()))


if __name__=='__main__':unittest.main()
