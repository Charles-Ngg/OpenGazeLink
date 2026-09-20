"""Conservative alignment of stimulus telemetry with continuous eye observations."""
from __future__ import annotations

import json
import numpy as np

ALIGNMENT_POLICY = {"version": "rails-v4-event-timing", "anchor_settle_ms": 700,
                    "anchor_guard": "700 ms plus available display and capture clock uncertainty",
                    "rail_settle_ms": 350, "lag_interval_ms": [-80, 250],
                    "rail_weight": .25, "parallel_weight": .15,
                    "minimum_aperture": .18,
                    "phone_clock": "fresh roundtrip alignment, minimum-transit fallback"}


def roundtrip_capture_time(timing):
    """Match PredictionClock's source time, retaining bounded clock uncertainty.

    Old captures have no probes and keep their explicitly named proxy clock.
    """
    keys = ('phone_sensor_time_ns','phone_send_time_ns','pc_first_packet_monotonic_ns',
            'phone_to_pc_offset_ns','clock_probe_uncertainty_ms','clock_probe_age_ms')
    try:
        sensor,send,receive,offset,uncertainty,age = [float(timing[k]) for k in keys]
    except (KeyError, TypeError, ValueError):
        return None
    if (not np.isfinite([sensor,send,receive,offset,uncertainty,age]).all()
            or not 0 < sensor <= send or receive <= 0 or not 0 <= age <= 5000
            or not 0 <= uncertainty <= 20):
        return None
    transit = max(0.,(receive-send-offset)/1e6)
    return receive/1e6-(send-sensor)/1e6-transit


def rail_constraint(pc_ms, index, times, events):
    """Perpendicular target plus a tangential interval, not an exact moving label.

    The interval permits lag and anticipation. It is an explicit engineering
    assumption retained in the audit, not a measured physiological delay.
    """
    event = events[index]
    rail = event.get("rail")
    if not isinstance(rail, dict) or event.get("motion_age_ms", 0) < 350:
        return None
    a, b = np.asarray(rail.get("a"), dtype=float), np.asarray(rail.get("b"), dtype=float)
    if a.shape != (2,) or b.shape != (2,) or not np.isfinite([a, b]).all():
        return None
    direction = b - a
    length = np.linalg.norm(direction)
    if length < .1:
        return None
    tangent = direction / length
    lo, hi = pc_ms - 250, pc_ms + 80
    first, last = int(np.searchsorted(times, lo, side="right") - 1), int(np.searchsorted(times, hi, side="left"))
    if first < 0 or last >= len(events):
        return None
    segment = events[first:last+1]
    if (np.any(np.diff(times[first:last+1]) > 120) or any(
            e["block"] != event["block"] or e["phase"] != "pursuit" or not e.get("visible", True)
            or e.get("sync_rtt_ms", 999) > 40 or e.get("rail") != rail for e in segment)):
        return None
    positions = np.array([[e["x"], e["y"]] for e in segment])
    normal = np.array([-tangent[1], tangent[0]])
    if np.max(np.abs((positions - a) @ normal)) > .002:
        return None
    along = positions @ tangent
    return {"tangent": tangent.tolist(), "normal": normal.tolist(),
            "lower": float(along.min() - .005), "upper": float(along.max() + .005),
            "normal_target": float(a @ normal), "lag_interval_ms": [-80, 250]}


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def align_frames(frames, events, *, pursuit_lag_ms=None, discarded_segments=()):
    if not frames or len(events) < 2:
        raise ValueError("VIDEO session needs frames and stimulus telemetry")
    times = np.array([event["pc_ms"] for event in events])
    if not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError("stimulus timestamps must be strictly increasing")
    # Estimate a separate phone clock offset after each source-clock reset.
    # The minimum transit time remains unknown: this is an upper-bound estimate,
    # not a sensor-to-screen latency measurement.
    epochs, epoch, previous = [], 0, None
    offsets = {}
    for frame in frames:
        if previous is not None and frame["source_ms"] <= previous:
            epoch += 1
        previous = frame["source_ms"]
        epochs.append(epoch)
        timing = frame["timing"]
        send, receive = timing.get("phone_send_time_ns", 0), timing.get("pc_first_packet_monotonic_ns", 0)
        if send and receive:
            offsets[epoch] = min(offsets.get(epoch, float("inf")), (receive - send) / 1e6)
    changed_at, last = [], None
    onset = times[0]
    for event in events:
        signature = (event["phase"], event["block"], event["x"], event["y"], event.get("capture_segment", 0))
        if signature != last:
            onset = event["pc_ms"]
        changed_at.append(onset)
        last = signature
    aligned = []
    previous_frame, previous_valid, previous_epoch, previous_block = None, False, -1, None
    for frame, epoch in zip(frames, epochs):
        timing = frame["timing"]
        roundtrip_time = roundtrip_capture_time(timing)
        if timing.get("source_capture_monotonic_ns"):
            pc_ms = timing["source_capture_monotonic_ns"] / 1e6
            clock = "camera_read_completion_proxy"
        elif roundtrip_time is not None:
            pc_ms = roundtrip_time
            clock = "phone_roundtrip_alignment"
        elif epoch in offsets and timing.get("phone_sensor_time_ns"):
            pc_ms = timing["phone_sensor_time_ns"] / 1e6 + offsets[epoch]
            clock = "phone_minimum_transit_proxy"
        else:
            pc_ms, clock = frame["pc_read_ms"], "unsynchronized_read_proxy"
        index = int(np.searchsorted(times, pc_ms, side="right") - 1)
        event = events[max(0, min(index, len(events) - 1))]
        valid = bool(frame["valid"] and frame.get("input") and event.get("capture_segment", 0) not in discarded_segments)
        delta = frame["source_ms"] - previous_frame if previous_frame is not None else 0
        reset = (not previous_valid or not valid or epoch != previous_epoch
                 or not 0 < delta <= 250 or event.get("capture_segment", 0) != previous_block)
        label = [event["x"], event["y"]]
        weight = 0.
        constraint = None
        reasons = []
        supported = (0 <= index < len(events) - 1 and times[index + 1] - times[index] <= 120
                     and event["visible"] and event["phase"] != "pause" and event["sync_rtt_ms"] <= 40
                     and event.get("capture_segment", 0) not in discarded_segments
                     and event.get("capture_segment", 0) == events[index+1].get("capture_segment", 0)
                     and clock != "unsynchronized_read_proxy")
        if not valid:
            reasons.append("invalid_eye_input")
        if not supported:
            reasons.append("unsupported_display_or_clock")
        if supported and valid and event["phase"] in ("anchor", "jump"):
            # Require settling after the actual displayed change. Transition eye
            # frames remain in the sequence, but receive no coordinate label.
            uncertainty = float(event.get("sync_rtt_ms", 0)) / 2
            if clock == "phone_roundtrip_alignment":
                uncertainty += float(timing.get("clock_probe_uncertainty_ms", 0))
            if pc_ms - changed_at[index] >= 700 + uncertainty:
                weight = 1.
            else:
                reasons.append("anchor_settling")
        if supported and valid and event["phase"] == "pursuit":
            constraint = rail_constraint(pc_ms, index, times, events)
            if constraint is not None:
                weight = .25
            else:
                reasons.append("pursuit_without_stable_rail_constraint")
        if supported and valid and event["phase"] == "pursuit" and pursuit_lag_ms is not None:
            shifted = pc_ms - pursuit_lag_ms
            j = int(np.searchsorted(times, shifted, side="right") - 1)
            if 0 <= j < len(events) - 1:
                a, b = events[j], events[j + 1]
                if (a["phase"] == b["phase"] == "pursuit" and a["block"] == b["block"] == event["block"]
                        and times[j + 1] - times[j] <= 120 and a["visible"] and b["visible"]):
                    fraction = (shifted - times[j]) / (times[j + 1] - times[j])
                    label = ((1 - fraction) * np.array([a["x"], a["y"]]) + fraction * np.array([b["x"], b["y"]])).tolist()
                    weight = .1
                    constraint = None
        aligned.append(dict(frame, valid=bool(valid), pc_ms=pc_ms, clock=clock, target=label, weight=weight,
                            reset=bool(reset), dt_ms=max(0., min(delta, 250.)), block=event["block"], phase=event["phase"],
                            condition=event.get("condition", "normal"), constraint=constraint,
                            capture_segment=event.get("capture_segment", 0),
                            alignment_reasons=reasons, stimulus_index=int(index),
                            clock_epoch=epoch,
                            clock_offset_ms=(timing['phone_to_pc_offset_ns']/1e6 if clock=='phone_roundtrip_alignment' else offsets.get(epoch)),
                            clock_uncertainty_ms=(timing.get('clock_probe_uncertainty_ms') if clock=='phone_roundtrip_alignment' else None),
                            trial_id=event.get("trial_id", event["block"]),
                            split=event.get("split", "validation" if event["block"].startswith("validation-") else "test" if event["block"].startswith("test-") else "train"),
                            motion_profile=event.get("motion_profile", "legacy"),
                            trial_complete=bool(event.get("trial_complete", False)),
                            drag_progress=float(event.get("drag_progress", 0.)),
                            trial_age_ms=event.get("trial_age_ms"),
                            # Derived from actual submitted target changes, not
                            # the intended schedule or the arrival time of video.
                            target_age_ms=float(pc_ms - changed_at[index]) if supported else None,
                            target_onset_trial_ms=(float(event["trial_age_ms"]) -
                                (float(event["pc_ms"]) - changed_at[index]))
                                if supported and event.get("trial_age_ms") is not None else None,
                            display_sync_uncertainty_ms=float(event.get("sync_rtt_ms", 0)) / 2,
                            display_frame_id=event.get("display_frame_id"),
                            stimulus_supported=bool(supported),
                            source_dt_ms=delta, policy=ALIGNMENT_POLICY["version"]))
        previous_frame, previous_valid, previous_epoch, previous_block = frame["source_ms"], valid and supported, epoch, event.get("capture_segment", 0)
    return aligned
