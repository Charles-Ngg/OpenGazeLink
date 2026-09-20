"""One capture; spatial calibration followed by frozen-base temporal fitting."""
import hashlib
import json
from pathlib import Path
import shutil
from datetime import datetime, timezone
from .paths import DATA_DIR
from .video_session import write_json


def train_calibration(session_path, *, progress=print, cancelled=lambda: False,
                      spatial_epochs=120, prediction_epochs=60, publish=True,
                      force_publish=False):
    from .video_training import train_session
    from .unified_prediction_training import train_unified, publish_unified
    session_path = Path(session_path)
    session_file = session_path / "session.json"
    capture = json.loads(session_file.read_text(encoding="utf-8")) if session_file.exists() else {}
    plan = capture.get("plan", [])
    spatial_only = bool(plan) and all(step.get("calibration_stage") == "spatial_v1" for step in plan)
    # Retain force_publish as a backwards-compatible CLI keyword. User spatial
    # calibration always publishes a successfully trained/exported model now.
    output = session_path / "unified-runs" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S-%fZ")
    output.mkdir(parents=True)
    active = DATA_DIR / "conditioned-video-model.json"
    prior = active.read_bytes() if active.exists() else None
    from .paths import RESOURCE_ROOT
    sources = {}
    for name in ("unified_calibration_training.py", "unified_capture.py", "calibration_split.py", "video_dataset.py"):
        source = Path(__file__).with_name(name)
        if not source.is_file():
            source = RESOURCE_ROOT / "provenance" / "opengazelink_pc" / name
        if source.is_file():
            shutil.copy2(source, output / name)
            sources[name] = hashlib.sha256(source.read_bytes()).hexdigest()
    replay_report = {}
    replay_path = session_path / "replay-report.json"
    if replay_path.exists():
        replay_report = json.loads(replay_path.read_text(encoding="utf-8"))
    report = {"source_sha256": sources, "published": False, "training_directory": str(output),
              "spatial_only": spatial_only,
              "replay": replay_report,
              "strategy": "personal spatial calibration then frozen-base source-time trajectory denoising and forecasting; whole-trial train/selection/test split"}
    try:
        progress("unified_spatial")
        spatial = train_session(session_path, epochs=spatial_epochs, progress=progress, cancelled=cancelled,
                                publish=False, train_temporal=False)
        report["spatial"] = spatial
        if spatial_only:
            selected = True
        else:
            from .spatial_metrics import acceptance
            final_test = spatial.get("independent_test", {})
            reference = final_test.get("incumbent") or final_test.get("baseline")
            selected = spatial["publication_reason"] == "eligible" and acceptance(final_test.get("candidate", {}), reference, improve=False)
        report["acceptance_policy"] = ("publish after test-set epoch selection; no accuracy gate"
                                       if spatial_only else
                                       "middle 5-15 degrees with all-anchor fallback; exact anchors in >=2 trials; "
                                       "median/P75/P95; legacy v5 captures require measured PnP motion compliance")
        model = Path(spatial["training_directory"]) / "conditioned-video-model.json" if selected else active
        if not model.exists():
            report["reason"] = "spatial export missing; capture retained"
            return report
        # Forecast gradients must not move the calibrated spatial coordinate
        # system. Train a head on exactly the frozen module used at runtime.
        frozen = output / "spatial"
        frozen.mkdir()
        meta = json.loads(model.read_text(encoding="utf-8"))
        module = model.with_name(meta["variants"]["conditioned_video"]["module_file"])
        shutil.copy2(model, frozen / model.name)
        shutil.copy2(module, frozen / module.name)
        joint_enabled = bool(meta["variants"]["conditioned_video"].get("joint_prediction"))
        report["prediction"] = {
            "accepted": joint_enabled,
            "published": False,
            "mode": "embedded_joint_head" if joint_enabled else "unavailable",
            "validation": spatial.get("joint_future_validation"),
            "independent_test": (spatial.get("independent_test") or {}).get("future"),
        }
        prediction_output = output / "prediction"
        if spatial_only:
            report["strategy"] = "stage one: training set fitting, test set epoch selection, no accuracy release gate"
            report["prediction"] = {"accepted":False,"published":False,"mode":"disabled_spatial_stage"}
        elif not joint_enabled:
            progress("unified_prediction")
            report["prediction"] = train_unified(
                session_path, frozen / model.name, prediction_output,
                epochs=prediction_epochs, progress=progress, cancelled=cancelled,
            )
            report["prediction"]["mode"] = "frozen_spatial_trajectory"
        report["spatial_selected"] = selected
        report["base_source"] = spatial.get("base_source")
        report["incumbent_is_comparator_only"] = False
        if cancelled():
            raise RuntimeError("统一训练已取消，数据与候选保留")
        if publish:
            if (active.read_bytes() if active.exists() else None) != prior:
                raise RuntimeError("训练期间当前模型发生变化，候选已保留，请重新检查")
            if selected:
                if prior is not None:
                    (output / "previous-spatial.json").write_bytes(prior)
                DATA_DIR.mkdir(parents=True, exist_ok=True)
                destination = active.with_name(module.name)
                shutil.copy2(module, destination)
                if hashlib.sha256(destination.read_bytes()).hexdigest() != meta["variants"]["conditioned_video"]["module_sha256"]:
                    raise ValueError("spatial publication checksum mismatch")
                write_json(active, meta)
                report["published"] = True
                spatial["published"] = True
            if selected and joint_enabled:
                report["prediction"]["published"] = True
            forecast = active.with_name("conditioned-video-forecast.json")
            if not joint_enabled and report["prediction"]["accepted"]:
                publish_unified(prediction_output, active)
                report["prediction"]["published"] = True
                report["published"] = True
            elif selected and forecast.exists():
                shutil.copy2(forecast, output / "previous-forecast.json")
                forecast.unlink()
            report["reason"] = ("validated temporal predictor published against frozen spatial model"
                                if report["prediction"]["published"] else
                                "stage one spatial model published; prediction training skipped" if selected and spatial_only else
                                "spatial model published; temporal candidate did not pass validation" if selected else
                                "candidates retained; installed models kept")
        return report
    except Exception as error:
        report["error"] = str(error)
        raise
    finally:
        write_json(output / "report.json", report)
