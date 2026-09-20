"""Exercise packaged inference/training dependencies without a camera or user data."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import traceback


def run(report_path: Path) -> int:
    from . import __version__
    report = {"version": __version__, "ok": False, "checks": {}}
    try:
        import av
        import numpy as np
        import torch
        from .paths import RESOURCE_ROOT
        from .landmarker import FaceLandmarkerProvider
        from .personal_binocular_training import PersonalBinocularNet, PersonalBinocularInference, _geometry_mask

        torch.set_num_threads(2)
        assets = RESOURCE_ROOT / "models" / "public-conditioned"
        metadata = json.loads((assets / "result.json").read_text(encoding="utf-8"))
        model = PersonalBinocularNet()
        missing, unexpected = model.load_state_dict(
            torch.load(assets / "best.pt", map_location="cpu", weights_only=True), strict=False)
        expected = {"uncertainty.0.weight", "uncertainty.0.bias", "uncertainty.2.weight", "uncertainty.2.bias"}
        assert set(missing) == expected and not unexpected, (missing, unexpected)
        report["checks"]["public_checkpoint_sha256"] = hashlib.sha256((assets / "best.pt").read_bytes()).hexdigest()
        # An actual optimizer step catches frozen-build dependencies that imports
        # and a control-page launch alone cannot exercise.
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
        directions, uncertainty = model(torch.zeros(2, 2, 36, 64), torch.zeros(2, 70))
        (directions.square().mean() + uncertainty.square().mean()).backward()
        optimizer.step()
        report["checks"]["training_step"] = True
        normalization = metadata["normalization"]
        inference = PersonalBinocularInference(model.eval(), normalization["mean"], normalization["scale"], _geometry_mask())
        inputs = (torch.zeros(2, 2, 36, 64, dtype=torch.uint8), torch.zeros(2, 70))
        with tempfile.TemporaryDirectory(prefix="opengazelink-self-check-") as directory:
            exported = Path(directory) / "model.pt"
            with torch.no_grad():
                traced = torch.jit.trace(inference, inputs)
                traced.save(str(exported))
                actual = torch.jit.load(str(exported))(*inputs)
                reference = inference(*inputs)
                for a, b in zip(actual, reference):
                    assert torch.isfinite(a).all() and torch.allclose(a, b, atol=1e-6)
        report["checks"]["torchscript_export_reload"] = True
        landmarker = FaceLandmarkerProvider()
        try:
            landmarker.detect_bgr(np.zeros((128, 128, 3), dtype=np.uint8), 1)
        finally:
            landmarker.close()
        report["checks"]["face_landmarker"] = True
        codec = av.CodecContext.create("h264", "r")
        codec.open()
        report["checks"]["h264_decoder"] = True
        for name in ("index.html", "app.js", "styles.css", "i18n.js", "unified-plan.js", "video.js"):
            assert (RESOURCE_ROOT / "web" / name).is_file(), name
        # Import the complete control/training graph, including provenance users.
        from . import app_host, video_replay, unified_calibration_training
        report["checks"]["control_and_training_imports"] = True
        from .config import ProviderConfig
        from .control_server import ControlApplication, ControlServer
        from .paths import USER_ROOT
        from urllib.request import urlopen
        config = ProviderConfig(udp_bind="127.0.0.1", udp_port=0, control_port=0,
                                discovery_port=0)
        application = ControlApplication(config, USER_ROOT / "config.json")
        server = None
        try:
            server = ControlServer(application)
            server.start(open_browser=False)
            base = "http://127.0.0.1:" + str(server.server.server_address[1])
            with urlopen(base + "/api/status", timeout=10) as response:
                status = json.load(response)
            assert not status["config"]["geometry_configured"]
            assert not any(m.get("ready") for m in status["artifacts"]["models"].values())
            with urlopen(base + "/", timeout=10) as response:
                assert b"combinedDot" in response.read()
            report["checks"]["fresh_control_http"] = True
            report["checks"]["isolated_udp_tcp_receiver"] = True
        finally:
            if server is not None:
                server.close()
            application.close()
        report["ok"] = True
    except Exception:
        report["error"] = traceback.format_exc()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0 if report["ok"] else 1
