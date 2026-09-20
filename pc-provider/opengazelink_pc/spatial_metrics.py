"""Target-defined eccentricity buckets; weak rails are never point truth."""
import numpy as np

REGIONS = (("center", 0., 5.), ("middle", 5., 15.), ("edge", 15., 180.))


def eccentricity_degrees(target, centers, origin, size):
    target, centers = np.asarray(target), np.asarray(centers)
    eye = centers.mean(axis=1) if centers.ndim == 3 else centers
    point = np.tile(origin, (len(target), 1)).astype(float)
    point[:, :2] += (target - .5) * size * [-1., 1.]
    ray, central = point - eye, np.asarray(origin) - eye
    cosine = (ray * central).sum(1) / np.maximum(1e-12, np.linalg.norm(ray, axis=1) * np.linalg.norm(central, axis=1))
    return np.degrees(np.arccos(np.clip(cosine, -1., 1.)))


def summary(error):
    error = np.asarray(error)
    if not len(error):
        return dict(frames=0, median_px=None, p75_px=None, p95_px=None, mean_px=None)
    # Invalid model outputs must fail acceptance, not silently disappear.
    if not np.isfinite(error).all():
        return dict(frames=len(error), median_px=None, p75_px=None, p95_px=None, mean_px=None,
                    invalid_predictions=int((~np.isfinite(error)).sum()))
    return dict(frames=len(error), median_px=float(np.median(error)),
                p75_px=float(np.percentile(error, 75)), p95_px=float(np.percentile(error, 95)),
                mean_px=float(error.mean()))


def spatial_metrics(prediction, target, weight, wh, eccentricity=None, rows=None):
    prediction, target, weight = map(np.asarray, (prediction, target, weight))
    mask = weight > .5
    if rows is not None:
        mask &= np.array([r.get("constraint") is None for r in rows])
    error = np.linalg.norm((prediction - target) * (np.asarray(wh) - 1), axis=1)
    result = summary(error[mask])
    result["trials"] = len({rows[i]["trial_id"] for i in np.flatnonzero(mask)}) if rows is not None else 0
    result["reference"] = "settled stimulus anchor assumption; not measured eye-tracker truth"
    # A whole-screen median can hide opposite signed errors at the two edges.
    # Keep top/bottom separate, and never turn rail intervals into point truth.
    result["by_screen_band"] = {}
    residual_y = (prediction[:, 1] - target[:, 1]) * (np.asarray(wh)[1] - 1)
    for name, band in (("top", target[:, 1] < .1),
                       ("middle", (target[:, 1] >= .1) & (target[:, 1] <= .9)),
                       ("bottom", target[:, 1] > .9)):
        selected = mask & band
        values = residual_y[selected]
        region = summary(error[selected])
        finite = len(values) > 0 and np.isfinite(values).all()
        region.update(signed_y_median_px=float(np.median(values)) if finite else None,
                      absolute_y_median_px=float(np.median(np.abs(values))) if finite else None,
                      trials=len({rows[i]["trial_id"] for i in np.flatnonzero(selected)}) if rows is not None else 0)
        result["by_screen_band"][name] = region
    if eccentricity is not None:
        eccentricity = np.asarray(eccentricity)
        result["by_region"] = {}
        for name, lo, hi in REGIONS:
            selected = mask & (eccentricity >= lo) & (eccentricity < hi)
            region = summary(error[selected])
            region["trials"] = len({rows[i]["trial_id"] for i in np.flatnonzero(selected)}) if rows is not None else 0
            region["angle_range_deg"] = [lo, hi]
            result["by_region"][name] = region
        result["primary_region"] = "middle"
        result["primary_fallback"] = "all held-out exact anchors when middle has fewer than 20 frames or 2 trials"
        result["region_basis"] = "target eccentricity relative to screen center from per-frame PnP eye center"
    return result


def primary(metrics):
    middle = metrics.get("by_region", {}).get("middle")
    if (middle and middle.get("median_px") is not None and middle.get("p95_px") is not None
            and middle.get("frames", 0) >= 20 and middle.get("trials", 0) >= 2):
        return middle
    return metrics


def supported(metrics):
    value = primary(metrics)
    populated = value.get("median_px") is not None and value.get("p95_px") is not None
    if "by_region" not in metrics:
        return populated
    return (populated and value.get("frames", 0) >= 20
            and value.get("trials", 0) >= 2)


def acceptance(candidate, reference, *, improve=True):
    if not reference or not supported(candidate) or not supported(reference):
        return False
    a, b = primary(candidate), primary(reference)
    if a["mean_px"] > b["mean_px"] + 1e-3:
        return False
    return bool((a["median_px"] < b["median_px"] - 1e-4 if improve else a["median_px"] <= b["median_px"] + 1e-3)
                and a.get("p75_px", a["mean_px"]) <= b.get("p75_px", b["mean_px"]) * 1.05 + 1e-3
                and a["p95_px"] <= b["p95_px"] * 1.10 + 1e-3)
