"""Compact legacy VIDEO sessions to the processed-data retention policy."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from opengazelink_pc.paths import DATA_DIR

DEFAULT_ROOT = Path(os.environ.get("LOCALAPPDATA", str(DATA_DIR.parent.parent))) / "OpenGazeLink/data/video-sessions"


REMOVABLE_AUDIT_FILES = ("stimulus-requests.jsonl", "stimulus-decisions.jsonl")
REPEATED_FRAME_FIELDS = ("camera_model",)


def _inside(root: Path, target: Path) -> Path:
    resolved = target.resolve()
    resolved.relative_to(root.resolve())
    return resolved


def _write_npz(path: Path, arrays: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".compact.tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    with np.load(temporary, allow_pickle=False) as checked:
        if set(checked.files) != set(arrays):
            raise RuntimeError(f"compacted input verification failed: {path}")
    temporary.replace(path)


def _compact_jsonl(path: Path, session: Path, *, migrate_landmarks: bool) -> dict:
    before = path.stat().st_size
    temporary = path.with_suffix(path.suffix + ".compact.tmp")
    if temporary.exists():
        # Recover from an explicitly interrupted earlier compaction. The path is
        # validated against this exact session before removing only the temp.
        _inside(session, temporary).unlink()
    rows = matrices = 0
    landmark_frames, landmark_values = [], []
    try:
        with path.open(encoding="utf-8") as source, temporary.open("x", encoding="utf-8", newline="\n") as output:
            for line in source:
                if not line.strip():
                    continue
                record = json.loads(line)
                diagnostics = record.get("diagnostics")
                faces = diagnostics.pop("landmarks", None) if isinstance(diagnostics, dict) else None
                if migrate_landmarks and faces:
                    landmarks = np.asarray(faces[0], dtype=np.float32)
                    if landmarks.ndim == 2 and landmarks.shape[1] == 3:
                        record["mediapipe_landmarks_index"] = len(landmark_values)
                        landmark_frames.append(int(record.get("index", rows)))
                        landmark_values.append(landmarks)
                        matrices += 1
                for field in REPEATED_FRAME_FIELDS:
                    record.pop(field, None)
                output.write(json.dumps(record, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n")
                rows += 1
        # Parse every output row before replacing the source.
        with temporary.open(encoding="utf-8") as checked:
            verified = sum(1 for line in checked if line.strip() and isinstance(json.loads(line), dict))
        if verified != rows:
            raise RuntimeError(f"compacted JSONL verification failed: {path}")
        if migrate_landmarks and landmark_values:
            # One compressed matrix archive per session avoids tens of thousands
            # of tiny NPZ rewrites while preserving every processed landmark.
            landmark_path = session / "mediapipe-landmarks.npz"
            _write_npz(landmark_path, {
                "frame_index": np.asarray(landmark_frames, dtype=np.int64),
                "landmarks": np.stack(landmark_values).astype(np.float32),
            })
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return {"path": str(path), "rows": rows, "matrices_migrated": matrices,
            "bytes_before": before, "bytes_after": path.stat().st_size}


def inventory(root: Path) -> dict:
    root = root.resolve()
    raw = [path for path in root.glob("*/raw-camera") if path.is_dir()]
    audits = [path for name in REMOVABLE_AUDIT_FILES for path in root.glob(f"*/{name}") if path.is_file()]
    raw_files = [file for directory in raw for file in directory.rglob("*") if file.is_file()]
    jsonl = list(root.glob("*/frames.jsonl")) + list(root.glob("*/training-runs/*/alignment.jsonl"))
    return {
        "root": str(root), "raw_directories": len(raw),
        "raw_bytes": sum(file.stat().st_size for file in raw_files),
        "duplicate_audit_files": len(audits),
        "duplicate_audit_bytes": sum(file.stat().st_size for file in audits),
        "compactable_jsonl_files": len(jsonl),
        "compactable_jsonl_bytes": sum(file.stat().st_size for file in jsonl),
    }


def compact(root: Path) -> dict:
    root = root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    reports = []
    for frames in sorted(root.glob("*/frames.jsonl")):
        session = _inside(root, frames.parent)
        reports.append(_compact_jsonl(_inside(root, frames), session, migrate_landmarks=True))
    for alignment in sorted(root.glob("*/training-runs/*/alignment.jsonl")):
        session = _inside(root, alignment.parents[2])
        reports.append(_compact_jsonl(_inside(root, alignment), session, migrate_landmarks=False))

    removed_bytes = removed_files = removed_directories = 0
    for name in REMOVABLE_AUDIT_FILES:
        for path in sorted(root.glob(f"*/{name}")):
            target = _inside(root, path)
            removed_bytes += target.stat().st_size
            target.unlink()
            removed_files += 1
    for path in sorted(root.glob("*/raw-camera")):
        target = _inside(root, path)
        files = [item for item in target.rglob("*") if item.is_file()]
        removed_bytes += sum(item.stat().st_size for item in files)
        removed_files += len(files)
        shutil.rmtree(target)
        removed_directories += 1

    for metadata_path in sorted(root.glob("*/session.json")):
        metadata_path = _inside(root, metadata_path)
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata.pop("raw_archive", None)
        metadata["retention"] = "legacy session compacted to processed eye tensors, MediaPipe matrices and lightweight timing/labels; full frames removed"
        metadata["storage_compacted_at"] = datetime.now(timezone.utc).isoformat()
        temporary = metadata_path.with_suffix(".json.compact.tmp")
        temporary.write_text(json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        json.loads(temporary.read_text(encoding="utf-8"))
        temporary.replace(metadata_path)

    return {
        "root": str(root), "jsonl": reports,
        "matrices_migrated": sum(item["matrices_migrated"] for item in reports),
        "jsonl_bytes_before": sum(item["bytes_before"] for item in reports),
        "jsonl_bytes_after": sum(item["bytes_after"] for item in reports),
        "removed_bytes": removed_bytes, "removed_files": removed_files,
        "removed_directories": removed_directories,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--apply", action="store_true", help="perform verified rewrites and delete redundant full-frame/audit data")
    args = parser.parse_args()
    print(json.dumps(compact(args.root) if args.apply else inventory(args.root), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
