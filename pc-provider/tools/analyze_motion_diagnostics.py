from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from opengazelink_pc.motion_analysis import analyze_motion_recording


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate gaze predictors against future phone timestamps.")
    parser.add_argument("recording", type=Path)
    parser.add_argument("--horizon-ms", type=float, default=80.0)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    result = analyze_motion_recording(arguments.recording, arguments.horizon_ms)
    output = arguments.output or arguments.recording.with_name(
        f"{arguments.recording.stem}-analysis-{int(arguments.horizon_ms)}ms.json"
    )
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "output": str(output.resolve()),
        "recommended_state_gated_method": result["recommended_state_gated_method"],
        "best_saccade_holdout_p90_method": result["best_saccade_holdout_p90_method"],
        "methods": {
            name: stats["saccade_holdout"] for name, stats in result["methods"].items()
        },
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
