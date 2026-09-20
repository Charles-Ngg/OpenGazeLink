"""Independent short-event replay; no CNN optimization or publication."""
from datetime import datetime, timezone
from pathlib import Path
import json
import numpy as np
from .event_temporal import EventTemporalFilter
from .one_euro import OneEuroFilter2D
from .spatial_metrics import spatial_metrics, eccentricity_degrees, summary
from .normalized_eye import screen_camera_origin
from .video_session import write_json


def future_reference(raw, times, segments, trials, horizon_ms):
    """Offline reference only; never interpolate across a trial, gap or restart."""
    raw, times, segments, trials = map(np.asarray, (raw, times, segments, trials))
    result = np.full(raw.shape, np.nan, dtype=float)
    if not len(raw) or not np.isfinite(horizon_ms) or horizon_ms < 0:
        return result
    boundaries = ((segments[1:] != segments[:-1]) | (trials[1:] != trials[:-1])
                  | (np.diff(times) <= 0) | (np.diff(times) > 100))
    starts = np.r_[0, np.flatnonzero(boundaries) + 1]
    for start, end in zip(starts, np.r_[starts[1:], len(times)]):
        t = times[start:end]
        requested = t + horizon_ms
        supported = requested <= t[-1]
        ids = np.arange(start, end)[supported]
        for axis in (0, 1):
            result[ids, axis] = np.interp(requested[supported], t, raw[start:end, axis])
    return result


def is_prediction(state):
    return bool(state.get("prediction_active", state.get("mode") == "event_saccade_landing_prediction"))


def timing_summary(values):
    values = np.asarray(values, float)
    return dict(samples=len(values), median_ms=float(np.median(values)) if len(values) else None,
                p95_ms=float(np.percentile(values, 95)) if len(values) else None)


def target_windows(rows, ids):
    """Group actual target holds without merging restarts; support old caches."""
    groups = {}
    for i in ids:
        row = rows[i]
        age = row.get("trial_age_ms")
        if age is None:
            continue
        onset = row.get("target_onset_trial_ms")
        if "target_onset_trial_ms" in row and onset is None:
            continue  # unsupported display/clock, not an old fixed-plan cache
        if onset is None:
            # Legacy cache compatibility only. New alignments retain displayed
            # change times even when replaying an old recording.
            onset = 0 if age < 1200 else 1200 if age < 2700 else 2700
        key = (row["trial_id"], row.get("capture_segment", 0), round(onset, 3))
        groups.setdefault(key, []).append(int(i))
    return groups


def evaluate_session(session_path, *, progress=print, cancelled=lambda: False):
    from .video_forecast_training import replay
    session_path = Path(session_path)
    output = session_path / "event-runs" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S-%fZ")
    output.mkdir(parents=True)
    progress("event_replay")
    data, rows, metadata, cfg = replay(session_path,
        session_path / "base-model" / "conditioned-video-model.json", output, progress, cancelled)
    session_file = session_path / "session.json"
    plan = json.loads(session_file.read_text(encoding="utf-8")).get("plan", []) if session_file.exists() else []
    planned = {step["trial_id"]: step for step in plan}
    wh = np.array([cfg["screen_width"], cfg["screen_height"]], float)
    screen_cm = cfg["screen_diagonal_inches"] * 2.54 * wh / np.linalg.norm(wh)
    origin = screen_camera_origin(int(wh[0]), int(wh[1]), cfg["screen_diagonal_inches"],
        [cfg["camera_offset_x_cm"], cfg["camera_offset_y_cm"], cfg["camera_offset_z_cm"]])
    from .stability_profile import estimate_profile, validate_profile
    raw = data["raw"] * (wh - 1)
    centers, rotations, scales = [], [], []
    for i, row in enumerate(rows):
        if i % 120 == 0 and cancelled():
            raise RuntimeError("事件检查已取消，采集保留")
        with np.load(session_path / row["input"], allow_pickle=False) as stored:
            center = stored["center"].copy()
            rotations.append(stored["rotation"][0].copy())
        centers.append(center)
        depth = abs(float(center[:, 2].mean()) - float(origin[2]))
        scales.append(np.linalg.norm(wh) / np.linalg.norm(screen_cm) * max(1., depth) * np.pi / 180)
    profile = estimate_profile(raw, [r["source_ms"] for r in rows], scales, rows)
    event = EventTemporalFilter(cfg["one_euro_min_cutoff"], cfg["one_euro_beta"], cfg["one_euro_derivative_cutoff"])
    event.set_stability_profile(profile)
    default_event = EventTemporalFilter(cfg["one_euro_min_cutoff"], cfg["one_euro_beta"], cfg["one_euro_derivative_cutoff"])
    euro = OneEuroFilter2D(cfg["one_euro_min_cutoff"], cfg["one_euro_beta"], cfg["one_euro_derivative_cutoff"])
    outputs, stable, states, defaults, default_states = [], [], [], [], []
    for i, row in enumerate(rows):
        if i % 120 == 0 and cancelled():
            raise RuntimeError("事件检查已取消，采集保留")
        if (i == 0 or data["segment"][i] != data["segment"][i-1]
                or row["trial_id"] != rows[i-1]["trial_id"]):
            event.reset(); default_event.reset(); euro.reset()
        scale, rotation = scales[i], rotations[i]
        # A configured horizon is an explicit replay assumption; recorded live
        # frame age includes offline replay scheduling and cannot be reused.
        point, state = event.update(raw[i], row["source_ms"], wh, pixels_per_degree=scale,
            head_rotation=rotation, horizon_ms=cfg["extrapolation_horizon_ms"],
            max_lead_fraction=cfg["extrapolation_max_lead_fraction"], smooth=True)
        outputs.append(point); states.append(state)
        default_point, default_state = default_event.update(raw[i], row["source_ms"], wh, pixels_per_degree=scale,
            head_rotation=rotation, horizon_ms=cfg["extrapolation_horizon_ms"],
            max_lead_fraction=cfg["extrapolation_max_lead_fraction"], smooth=True)
        defaults.append(default_point); default_states.append(default_state)
        stable.append(np.asarray(euro.update(tuple(raw[i] / (wh-1)), row["source_ms"]/1000)) * (wh-1)
                      if cfg["one_euro_enabled"] else raw[i])
    output_points = np.asarray(outputs)
    variants = dict(raw=raw, one_euro=np.asarray(stable), event=output_points)
    future = future_reference(raw, [r["source_ms"] for r in rows], data["segment"],
                              [r["trial_id"] for r in rows], cfg["extrapolation_horizon_ms"])
    targets = np.array([r["target"] for r in rows])
    weight = np.array([r["weight"] for r in rows])
    ecc = eccentricity_degrees(targets, np.asarray(centers), origin, screen_cm)
    decision = validate_profile(profile, np.asarray(defaults), output_points, targets * (wh-1),
                                rows, default_states, states)
    decision["parameters"] = profile
    report = dict(published=False, mode="event_evaluation", training_directory=str(output),
        stability_calibration=decision, stability_published=False,
        reference="settled stimulus is an assumed fixation target; early eye position has no independent truth",
        horizon_ms_assumed=cfg["extrapolation_horizon_ms"],
        model_sha256=metadata["variants"]["conditioned_video"]["module_sha256"],
        prediction_model="event_kinematic_v3",
        by_split={}, events=[], parameters_changed=False)
    for split in ("train", "validation", "test"):
        ids = np.flatnonzero([r["split"] == split for r in rows])
        local_rows = [rows[i] for i in ids]
        result = {name: spatial_metrics(points[ids]/(wh-1), targets[ids], weight[ids], wh, ecc[ids], local_rows)
                  for name, points in variants.items()}
        jitters = {name: [] for name in variants}
        late_predictions = 0
        windows = target_windows(rows, ids)
        attempts = sorted({(trial, segment) for trial, segment, _ in windows})
        stability_switches = 0
        settled_saccades = 0
        for trial, segment in attempts:
            starts = sorted(start for tr, seg, start in windows if (tr, seg) == (trial, segment))
            for window_index, start in enumerate(starts):
                window = windows[(trial, segment, start)]
                settled = np.array([i for i in window if
                    rows[i].get("target_age_ms", rows[i]["trial_age_ms"]-start) >= 700 and weight[i] > .5], dtype=int)
                if len(settled) < 4:
                    continue
                for name, points in variants.items():
                    residual = points[settled] - np.median(points[settled], axis=0)
                    jitters[name].append(float(np.sqrt(np.mean(np.sum(residual**2, axis=1)))))
                late_predictions += sum(is_prediction(states[i]) for i in settled)
                settled_saccades += sum(states[i]["mode"].startswith("event_saccade") for i in settled)
                stability_switches += sum(bool(states[a]["stability_active"]) != bool(states[b]["stability_active"])
                                          for a, b in zip(settled, settled[1:]) if 0 < rows[b]["source_ms"]-rows[a]["source_ms"] <= 100)
                if window_index == 0:
                    continue
                active = [i for i in window if rows[i].get("target_age_ms", rows[i]["trial_age_ms"]-start) < 700]
                predicted = [i for i in active if is_prediction(states[i])]
                target = targets[settled[0]] * (wh-1)
                record = dict(trial_id=trial, split=split, stimulus_jump_ms=start,
                    capture_segment=segment,
                    amplitude_band=planned.get(trial, {}).get("amplitude_band", "legacy"),
                    planned_amplitude_normalized=planned.get(trial, {}).get("amplitude_normalized"),
                    direction_axis=planned.get(trial, {}).get("direction_axis"),
                    predicted_frames=len(predicted),
                    braking_prediction_frames=sum(states[i].get("landing_px") is not None for i in predicted),
                    landing_target_error_px=summary(np.linalg.norm(output_points[predicted] - target, axis=1)),
                    reference="displayed destination assumption; no measured eye-onset or true in-flight trajectory")
                observed = [i for i in active if states[i]["mode"].startswith("event_saccade")]
                record["stimulus_to_detected_motion_ms"] = (
                    rows[observed[0]]["trial_age_ms"]-start if observed else None)
                record["timing_notice"] = ("Detection includes human response, estimator and decision delay. "
                    "Display clock is rAF submission, not measured photon onset; not physiological reaction time.")
                record["display_sync_uncertainty_ms"] = timing_summary([
                    rows[i]["display_sync_uncertainty_ms"] for i in active if rows[i].get("display_sync_uncertainty_ms") is not None])
                record["capture_clock_uncertainty_ms"] = timing_summary([
                    rows[i]["clock_uncertainty_ms"] for i in active if rows[i].get("clock_uncertainty_ms") is not None])
                supported = np.array([i for i in active if np.isfinite(future[i]).all()], dtype=int)
                record["all_window_future_proxy_error_px"] = {
                    name: summary(np.linalg.norm(points[supported] - future[supported], axis=1))
                    for name, points in variants.items()}
                record["future_proxy_notice"] = ("Every supported frame in the 700 ms stimulus-response window, "
                    "including abstentions; reference is later frozen spatial output, not eye-tracker truth.")
                # Acquisition is event-weighted and includes human response time.
                # Null means no sustained acquisition, never silently excluded.
                for name, points in variants.items():
                    first = None; acquired = None
                    for i in active + settled.tolist():
                        if np.linalg.norm(points[i]-target) <= scales[i]*2:
                            first = rows[i]["pc_ms"] if first is None else first
                            if rows[i]["pc_ms"] - first >= 40:
                                acquired = rows[i]["trial_age_ms"] - start - (rows[i]["pc_ms"]-first)
                                break
                        else:
                            first = None
                    record[name+"_stimulus_to_2deg_entry_ms"] = acquired
                report["events"].append(record)
        result["fixation_jitter_rms_px_per_event"] = {name: summary(v) for name, v in jitters.items()}
        result["predicted_frames_in_settled_fixations"] = late_predictions
        result["saccade_frames_in_settled_fixations"] = settled_saccades
        result["stability_switches_in_settled_fixations"] = stability_switches
        local_events = [e for e in report["events"] if e["split"] == split]
        result["by_amplitude_band"] = {}
        for band in sorted({e["amplitude_band"] for e in local_events}):
            selected = [e for e in local_events if e["amplitude_band"] == band]
            result["by_amplitude_band"][band] = dict(events=len(selected),
                detected_events=sum(e["stimulus_to_detected_motion_ms"] is not None for e in selected),
                predicted_events=sum(e["predicted_frames"] > 0 for e in selected),
                stimulus_to_detected_motion_ms=timing_summary([e["stimulus_to_detected_motion_ms"] for e in selected
                                                       if e["stimulus_to_detected_motion_ms"] is not None]))
        report["by_split"][split] = result
    report["source_interval_ms"] = dict(zip(("p50", "p95"), np.percentile([r["dt_ms"] for r in rows if r["dt_ms"]>0], [50,95]).tolist()))
    report["notice"] = "All three splits reported separately; fixed prediction algorithm; automatic stability noise/quiet-period estimated from training and gated by validation, never test. Entry time includes human response and spatial error; not system latency. Endpoint metrics include only predicted frames; all-window future proxy errors include abstentions. Both continuation and braking outputs count as predictions. Inspect coverage and settled prediction counts together."
    np.savez_compressed(output / "event-output.npz", raw=raw, one_euro=variants["one_euro"], event=output_points,
                        frame_index=data["frame_index"], modes=[s["mode"] for s in states],
                        prediction_active=[is_prediction(s) for s in states], default_event=np.asarray(defaults))
    write_json(output / "report.json", report)
    progress("event_evaluation_complete")
    return report
