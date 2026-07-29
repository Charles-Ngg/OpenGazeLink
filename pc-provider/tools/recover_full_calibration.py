from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _load_dataset(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    samples = list(payload.get("samples") or [])
    if not samples:
        raise ValueError(f"{path} contains no samples")
    for calibration_pass in payload.get("passes") or []:
        calibration_pass["cancelled"] = False
    return payload


def _latest_complete_capture(data_dir: Path) -> Path:
    candidates = []
    for directory in (data_dir / "calibration-frames").glob("*"):
        if not directory.is_dir():
            continue
        if all((directory / f"partial-{name}-dataset.json").is_file()
               for name in ("tasks", "legacy")):
            candidates.append(directory)
    if not candidates:
        raise FileNotFoundError("no capture contains both partial datasets")
    return max(candidates, key=lambda value: value.stat().st_mtime)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Recover a completed OpenGazeLink capture after training/export failure",
    )
    parser.add_argument("--user-dir", type=Path, required=True)
    parser.add_argument("--capture-dir", type=Path)
    args = parser.parse_args()

    user_dir = args.user_dir.expanduser().resolve()
    os.environ["OPENGAZELINK_USER_DIR"] = str(user_dir)

    from opengazelink_pc.calibration_session import _commit_artifacts
    from opengazelink_pc.model_registry import DATASET_PATHS, MODEL_PATHS
    from opengazelink_pc.shared_eye_models import SharedTinyCnnModel

    data_dir = user_dir / "data"
    capture_dir = (
        args.capture_dir.expanduser().resolve()
        if args.capture_dir else _latest_complete_capture(data_dir)
    )
    datasets = {
        name: _load_dataset(capture_dir / f"partial-{name}-dataset.json")
        for name in ("tasks", "legacy")
    }
    staging = capture_dir / (
        "recovered-artifacts-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    )
    staging.mkdir(parents=True, exist_ok=False)

    artifacts: list[tuple[Path, Path]] = []
    diagnostics = {}
    try:
        for name in ("tasks", "legacy"):
            print(
                f"Training {name} CNN from {len(datasets[name].get('samples') or [])} samples...",
                flush=True,
            )
            dataset_target = DATASET_PATHS[name]
            metadata_target, module_target = MODEL_PATHS[name]
            dataset_stage = staging / dataset_target.name
            metadata_stage = staging / metadata_target.name
            module_stage = staging / module_target.name
            dataset_stage.write_text(
                json.dumps(datasets[name], ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            _, diagnostics[name] = SharedTinyCnnModel.fit_dataset(
                datasets[name], metadata_stage, module_stage,
            )
            SharedTinyCnnModel.load(metadata_stage, module_stage)
            print(f"Validated {name} CNN export.", flush=True)
            artifacts.extend((
                (dataset_stage, dataset_target),
                (metadata_stage, metadata_target),
                (module_stage, module_target),
            ))

        backup = capture_dir / (
            "recovery-backup-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        )
        _commit_artifacts(artifacts, backup)
        shutil.rmtree(staging, ignore_errors=True)

        completed = {
            "schema": "opengazelink-calibration-recovery-v1",
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "capture_dir": str(capture_dir),
            "samples": {name: len(dataset.get("samples") or [])
                        for name, dataset in datasets.items()},
            "diagnostics": diagnostics,
        }
        (capture_dir / "recovery-completed.json").write_text(
            json.dumps(completed, ensure_ascii=False, indent=2), encoding="utf-8",
        )
        print(json.dumps(completed["samples"], ensure_ascii=False))
        return 0
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


if __name__ == "__main__":
    sys.exit(main())
