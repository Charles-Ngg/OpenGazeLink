from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from opengazelink_pc.landmarker import NORMALIZED_EYE_LANDMARKER_BACKENDS
from opengazelink_pc.normalized_eye import NormalizedEyeBackend
from opengazelink_pc.shared_eye_models import SHARED_DATASET_SCHEMA
from opengazelink_pc.calibration_session import ROOT, project_relative_path, sample_payload


def _save(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _resolve_raw_frame(source: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    project_path = ROOT / path
    if project_path.exists():
        return project_path
    return source.parent / path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recompute a shared-eye dataset from its saved lossless calibration frames",
    )
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--landmarker", choices=NORMALIZED_EYE_LANDMARKER_BACKENDS, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    source = json.loads(args.source.read_text(encoding="utf-8"))
    if source.get("schema") != SHARED_DATASET_SCHEMA:
        raise ValueError("source is not a shared-eye calibration dataset")
    samples = list(source.get("samples") or [])
    if not samples:
        raise ValueError("source dataset has no samples")
    missing = [index for index, sample in enumerate(samples) if not sample.get("raw_frame")]
    if missing:
        raise ValueError(
            f"source dataset predates raw-frame capture; {len(missing)} samples cannot be recomputed"
        )

    output = {key: value for key, value in source.items() if key not in ("samples", "passes", "created_at")}
    output["created_at"] = datetime.now(timezone.utc).isoformat()
    output["samples"] = []
    output["passes"] = [{
        "id": f"offline-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S-%fZ')}",
        "condition": "offline_landmarker_recompute",
        "source_dataset": str(args.source.resolve()),
        "source_samples": len(samples),
        "landmarker_backend": args.landmarker,
    }]
    normalization = dict(output.get("normalization") or {})
    normalization.update({
        "face_landmarker": (
            "MediaPipe legacy Face Mesh" if args.landmarker == "legacy"
            else "MediaPipe Tasks Face Landmarker"
        ),
        "landmarker_backend": args.landmarker,
        "landmarker_delegate": "cpu",
    })
    output["normalization"] = normalization

    backend = NormalizedEyeBackend(args.landmarker)
    try:
        for index, sample in enumerate(samples):
            raw_path = _resolve_raw_frame(args.source, sample["raw_frame"])
            frame = cv2.imread(str(raw_path), cv2.IMREAD_COLOR)
            if frame is None:
                raise FileNotFoundError(f"could not read raw calibration frame: {raw_path}")
            observation = backend.predict(
                frame, float(sample["t_ms"]), dict(sample["camera_model"]),
            )
            if observation is None:
                raise RuntimeError(f"{args.landmarker} produced no valid observation for sample {index}")
            target = {
                "x": float(sample["target"][0]), "y": float(sample["target"][1]),
                "row": int(sample["row"]), "column": int(sample["column"]),
                "grid_index": int(sample["grid_index"]),
            }
            output["samples"].append(sample_payload(
                observation, target, np.asarray(sample["target_camera_cm"], dtype=np.float64),
                str(sample.get("condition") or "offline_recompute"),
                str(sample.get("pass_id") or output["passes"][0]["id"]),
                project_relative_path(raw_path),
            ))
            if (index + 1) % 25 == 0 or index + 1 == len(samples):
                print(f"Recomputed {index + 1}/{len(samples)} samples", flush=True)
    finally:
        backend.close()

    output["passes"][0]["completed_samples"] = len(output["samples"])
    output["passes"][0]["finished_at"] = datetime.now(timezone.utc).isoformat()
    _save(args.output, output)
    print(json.dumps({
        "ok": True, "source": str(args.source.resolve()),
        "output": str(args.output.resolve()), "samples": len(output["samples"]),
        "landmarker_backend": args.landmarker,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
