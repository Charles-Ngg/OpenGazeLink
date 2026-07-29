from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from opengazelink_pc.shared_eye_models import SharedTinyCnnModel, require_torch


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrain shared-eye tiny CNN models from saved datasets")
    parser.add_argument("--dataset", type=Path, default=Path("data/shared-eye-angle-calibration.json"))
    parser.add_argument("--cnn-model", type=Path, default=Path("data/shared-eye-cnn-model.json"))
    parser.add_argument("--cnn-module", type=Path, default=Path("data/shared-eye-cnn-module.pt"))
    parser.add_argument("--legacy-dataset", type=Path, default=Path("data/shared-eye-legacy-angle-calibration.json"))
    parser.add_argument("--legacy-cnn-model", type=Path, default=Path("data/shared-eye-legacy-cnn-model.json"))
    parser.add_argument("--legacy-cnn-module", type=Path, default=Path("data/shared-eye-legacy-cnn-module.pt"))
    args = parser.parse_args()
    require_torch()
    specs = {
        "tasks": (args.dataset, args.cnn_model, args.cnn_module),
        "legacy": (
            args.legacy_dataset, args.legacy_cnn_model, args.legacy_cnn_module,
        ),
    }
    diagnostics = {}
    outputs = {}
    for backend, (dataset_path, cnn_path, module_path) in specs.items():
        if backend == "legacy" and not dataset_path.exists():
            continue
        dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
        error_path = dataset_path.with_name(f"shared-eye-{backend}-training-error.json")
        try:
            print(f"Training {backend} shared-eye tiny CNN model...", flush=True)
            _, cnn_diagnostics = SharedTinyCnnModel.fit_dataset(dataset, cnn_path, module_path)
            if error_path.exists():
                error_path.unlink()
        except Exception as error:
            error_path.write_text(json.dumps({
                "schema": "eyetracing-shared-eye-training-error-v1",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "dataset": str(dataset_path.resolve()),
                "landmarker_backend": backend,
                "error_type": type(error).__name__,
                "error": str(error),
            }, ensure_ascii=False, indent=2), encoding="utf-8")
            raise
        diagnostics[backend] = cnn_diagnostics
        outputs[backend] = {
            "dataset": str(dataset_path.resolve()),
            "cnn_model": str(cnn_path.resolve()), "cnn_module": str(module_path.resolve()),
        }
    print(json.dumps({
        "ok": True,
        "models": outputs,
        "diagnostics": diagnostics,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
