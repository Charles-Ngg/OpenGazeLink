"""User calibration has train/test sets; legacy research captures remain readable."""
import numpy as np


def spatial_stage(plan):
    return bool(plan) and all(s.get("calibration_stage") == "spatial_v1" for s in plan)


def split_masks(rows, plan):
    test = np.array([r["block"].startswith("test-") for r in rows], dtype=bool)
    validation = np.array([r["block"].startswith("validation-") for r in rows], dtype=bool)
    if spatial_stage(plan):
        # Old validation trajectories become training trajectories; the entire
        # test trajectories are held out for epoch selection, never gradients.
        return ~test, test, np.zeros(len(rows), dtype=bool)
    return ~(test | validation), validation, test


def selection_score(prediction, rows, wh):
    """Weak-label loss for epoch ranking, not an accuracy acceptance gate."""
    # Use perpendicular rail error while moving; only settled anchors provide
    # point labels. There is no reliable along-rail gaze ground truth.
    target = np.asarray([r["target"] for r in rows])
    residual = np.asarray(prediction) - target
    weights = np.asarray([r["weight"] if r.get("constraint") or r["weight"] > .5 else 0 for r in rows])
    for i, row in enumerate(rows):
        constraint = row.get("constraint")
        if constraint:
            normal = np.asarray(constraint["normal"])
            tangent = np.asarray(constraint["tangent"])
            along = np.dot(prediction[i], tangent)
            outside = max(0, constraint["lower"] - along, along - constraint["upper"])
            residual[i] = normal * (np.dot(prediction[i], normal) - constraint["normal_target"]) + tangent * outside * .15 ** .5
    error = np.sum((residual * (np.asarray(wh) - 1)) ** 2, axis=1)
    if not np.isfinite(error).all() or weights.sum() <= 0:
        return float("inf")
    return float(np.average(error, weights=weights))
