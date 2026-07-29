from __future__ import annotations

import inspect
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from opengazelink_pc.shared_eye_appearance import (
    CNN_INPUT_CHANNELS,
    CNN_MODEL_HEIGHT,
    CNN_MODEL_WIDTH,
)
from opengazelink_pc.shared_eye_models import (
    _tiny_cnn_class,
    _trace_tiny_cnn,
    require_torch,
)


class CnnExportTests(unittest.TestCase):
    def test_trace_export_does_not_require_python_source(self) -> None:
        torch, nn = require_torch()
        model = _tiny_cnn_class(torch, nn)().cpu().eval()

        with mock.patch.object(inspect, "getsource", side_effect=OSError("source unavailable")):
            exported = _trace_tiny_cnn(torch, model)

        values = torch.from_numpy(np.random.default_rng(42).normal(
            size=(2, CNN_INPUT_CHANNELS, CNN_MODEL_HEIGHT, CNN_MODEL_WIDTH),
        ).astype(np.float32))
        gain = torch.ones((1, 8, 1, 1), dtype=torch.float32)
        bias = torch.zeros((1, 8, 1, 1), dtype=torch.float32)
        expected = model(values, gain, bias).detach().numpy()

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "cnn.pt"
            exported.save(str(path))
            loaded = torch.jit.load(str(path), map_location="cpu").eval()
            actual = loaded(values, gain, bias).detach().numpy()

        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
