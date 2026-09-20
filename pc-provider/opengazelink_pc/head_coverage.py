"""Measured pose coverage and equal trial/cell/pose mass, never cue-as-input."""
from collections import Counter
import numpy as np

MIN_AXIS_P05_P95_DEG = 8.0
MIN_OCCUPIED_POSE_BINS = 4
MIN_CELLS_WITH_MULTIPLE_POSES = 4
MIN_TRIAL_POSE_CHANGE_DEG = 6.0


def _planned_movement_compliance(yaw, pitch, rows, plan):
    planned = {step["trial_id"]: step for step in (plan or []) if step.get("head_start")}
    trials = {}
    for trial_id, step in planned.items():
        values = yaw if step.get("head_axis") == "yaw" else pitch
        ids = [i for i, row in enumerate(rows) if row["trial_id"] == trial_id and row["weight"] > 0]
        start_ids = [i for i in ids if row_progress(rows[i]) <= .2]
        end_ids = [i for i in ids if row_progress(rows[i]) >= .8]
        start = float(np.median(values[start_ids])) if start_ids else None
        end = float(np.median(values[end_ids])) if end_ids else None
        delta = end - start if start is not None and end is not None else None
        trials[trial_id] = {
            "split": step["split"], "pair_id": step["pair_id"], "axis": step["head_axis"],
            "head_start": step["head_start"], "head_end": step["head_end"],
            "start_frames": len(start_ids), "end_frames": len(end_ids),
            "start_median_deg": start, "end_median_deg": end, "delta_deg": delta,
            "completed": len(start_ids) >= 4 and len(end_ids) >= 4
                         and delta is not None and abs(delta) >= MIN_TRIAL_POSE_CHANGE_DEG,
        }
    by_split = {}
    for split in ("train", "validation", "test"):
        local = {key: value for key, value in trials.items() if value["split"] == split}
        pair_items = {}
        for pair_id in {value["pair_id"] for value in local.values()}:
            members = [value for value in local.values() if value["pair_id"] == pair_id]
            deltas = [value["delta_deg"] for value in members]
            pair_items[pair_id] = {
                "trials": len(members), "deltas_deg": deltas,
                "opposed": (len(members) == 2 and all(value["completed"] for value in members)
                            and deltas[0] * deltas[1] < 0),
            }
        axes = {}
        for axis, first, opposite in (("yaw", "head_left", "head_right"),
                                      ("pitch", "head_up", "head_down")):
            forward = [value["delta_deg"] for value in local.values()
                       if value["axis"] == axis and value["head_start"] == first and value["completed"]]
            reverse = [value["delta_deg"] for value in local.values()
                       if value["axis"] == axis and value["head_start"] == opposite and value["completed"]]
            contrasted = bool(forward and reverse)
            axes[axis] = {
                "opposite_instructions_tested": contrasted,
                "opposite_measured_directions": (float(np.median(forward)) * float(np.median(reverse)) < 0
                                                  if contrasted else None),
            }
        by_split[split] = {
            "trials": local, "pairs": pair_items, "axes": axes,
            "adequate": (bool(local) and all(value["opposed"] for value in pair_items.values())
                         and all(value["opposite_instructions_tested"]
                                 and value["opposite_measured_directions"] for value in axes.values())),
        }
    return {"minimum_trial_start_end_change_deg": MIN_TRIAL_POSE_CHANGE_DEG, "by_split": by_split}


def row_progress(row):
    return float(row.get("drag_progress", 0.))


def pose_balance(rotations, rows, train_ids, plan=None):
    rotation = np.asarray(rotations)[:, 0]
    u, _, vt = np.linalg.svd(rotation[train_ids].mean(0))
    reference = u @ np.diag([1., 1., np.linalg.det(u @ vt)]) @ vt
    relative = rotation @ reference.T
    yaw = np.degrees(np.arctan2(relative[:, 0, 2], relative[:, 2, 2]))
    pitch = np.degrees(np.arctan2(-relative[:, 1, 2], np.hypot(relative[:, 0, 2], relative[:, 2, 2])))
    bins = [(int(np.digitize(y, [-4., 4.])), int(np.digitize(p, [-4., 4.]))) for y, p in zip(yaw, pitch)]
    regional = bool(plan) and all(step.get("calibration_stage") == "spatial_v1"
                                 and step.get("plan_version") == 10 for step in plan)
    grid = 3 if regional else 4
    cells = [tuple(np.clip((np.asarray(row["target"]) * grid).astype(int), 0, grid-1)) for row in rows]
    keys = [(row["trial_id"], cells[i], bins[i]) for i, row in enumerate(rows)]
    counts = Counter(keys[i] for i in train_ids)
    # V10 gives each occupied screen region equal training mass first, then
    # balances distinct trial/observed-pose strata within it. A slow drag or
    # repeated corner hold must not dominate the useful center (or vice versa).
    strata = Counter(key[1] if regional else key[0] for key in counts)
    weights = np.zeros(len(rows), np.float32)
    for i in train_ids:
        weights[i] = 1 / (counts[keys[i]] * strata[keys[i][1 if regional else 0]])
    weights *= len(train_ids) / max(1e-9, weights.sum())
    weights = np.minimum(weights, 4.)
    weights *= len(train_ids) / max(1e-9, weights.sum())
    splits = {}
    for split in ("train", "validation", "test"):
        ids = [i for i, row in enumerate(rows) if row["split"] == split and row["weight"] > 0]
        by_cell = {}
        for i in ids:
            by_cell.setdefault(str(cells[i]), set()).add(bins[i])
        yaw_range = np.percentile(yaw[ids], [5, 95]).tolist() if ids else None
        pitch_range = np.percentile(pitch[ids], [5, 95]).tolist() if ids else None
        occupied = len({bins[i] for i in ids})
        multi_pose_cells = sum(len(v) >= 2 for v in by_cell.values())
        yaw_span = yaw_range[1] - yaw_range[0] if yaw_range else 0.
        pitch_span = pitch_range[1] - pitch_range[0] if pitch_range else 0.
        splits[split] = dict(
            frames=len(ids), yaw_p05_p95_deg=yaw_range, pitch_p05_p95_deg=pitch_range,
            yaw_span_deg=yaw_span, pitch_span_deg=pitch_span,
            occupied_pose_bins=occupied, cells_with_multiple_poses=multi_pose_cells, cells=len(by_cell),
            adequate=(yaw_span >= MIN_AXIS_P05_P95_DEG and pitch_span >= MIN_AXIS_P05_P95_DEG
                      and occupied >= MIN_OCCUPIED_POSE_BINS
                      and multi_pose_cells >= MIN_CELLS_WITH_MULTIPLE_POSES),
        )
    compliance = _planned_movement_compliance(yaw, pitch, rows, plan)
    if compliance["by_split"]["train"]["trials"]:
        splits["train"]["adequate"] = (splits["train"]["adequate"]
                                               and compliance["by_split"]["train"]["adequate"])
    region_audit = {}
    if regional:
        for y in range(3):
            for x in range(3):
                ids = [i for i in train_ids if cells[i] == (x, y)]
                region_audit[f"{x},{y}"] = dict(frames=len(ids), balance_mass=float(weights[ids].sum()))
    return weights, dict(reference="PnP rotation relative to training mean; prompts are not measured pose labels",
                         spatial_region_balance=region_audit,
                         reference_rotation=reference.tolist(), by_split=splits,
                         planned_movement_compliance=compliance,
                         requirements={"yaw_p05_p95_span_deg": MIN_AXIS_P05_P95_DEG,
                                       "pitch_p05_p95_span_deg": MIN_AXIS_P05_P95_DEG,
                                       "occupied_pose_bins": MIN_OCCUPIED_POSE_BINS,
                                       "cells_with_multiple_poses": MIN_CELLS_WITH_MULTIPLE_POSES,
                                       "per_trial_start_end_change_deg": MIN_TRIAL_POSE_CHANGE_DEG,
                                       "paired_distinct_rails_opposite_measured_directions": True,
                                       "opposite_instructions_opposite_measured_directions": True},
                         weighting=("equal 3x3 screen regions then trial/observed-pose strata; bounded frame weights; label confidence unchanged"
                                    if regional else "equal trial then spatial-cell/observed-pose strata; bounded frame weights"))
