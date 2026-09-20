"""Shared capture contract and quality checks for spatial and motion training."""
from collections import Counter
import math

PROTOCOL = "unified-calibration-v1"


def validate_plan(plan):
    if not isinstance(plan, list):
        raise ValueError("统一采集计划必须是小段列表")
    seen = set()
    stages = {step.get("calibration_stage", "unified") for step in plan}
    if len(stages) != 1 or not stages <= {"unified", "spatial_v1", "events_v1"}:
        raise ValueError("采集阶段不能混合")
    if "events_v1" in stages:
        if len(plan) not in (12, 48):
            raise ValueError("第二阶段需要12个小段（兼容第八版48段计划）")
    elif "spatial_v1" in stages:
        if not 12 <= len(plan) <= 180:
            raise ValueError("第一阶段直线校准需要 12–180 个小段")
    elif not 20 <= len(plan) <= 180:
        raise ValueError("统一采集计划需要 20–180 个小段")
    for step in plan:
        identity, split = step.get("trial_id"), step.get("split")
        if (split not in ("train", "validation", "test") or not isinstance(identity, str)
                or not identity.startswith(split + "-") or step.get("block") != identity or identity in seen):
            raise ValueError("采集计划的小段标识或数据划分无效")
        duration = float(step.get("duration", 0))
        if not math.isfinite(duration) or not 1200 <= duration <= 12000:
            raise ValueError("小段时长超出支持范围")
        seen.add(identity)
        if "spatial_v1" in stages:
            if step.get("motion_profile") != "line":
                raise ValueError("第一阶段只接受直线校准样本")
            for key in ("point", "end"):
                point = step.get(key)
                margin = 0 if step.get("plan_version") == 10 else .02
                if not isinstance(point, list) or len(point) != 2 or not all(isinstance(x, (int, float)) and math.isfinite(x) and margin <= x <= 1-margin for x in point):
                    raise ValueError("采样点必须位于屏幕内")
        elif "events_v1" in stages:
            if step.get("motion_profile") != "jump":
                raise ValueError("第二阶段只接受短眼跳小段")
            points = step.get("points")
            changes = step.get("changes")
            if (not isinstance(points, list) or len(points) != 3
                    or any(not isinstance(point, list) or len(point) != 2
                           or not all(isinstance(x, (int, float)) and math.isfinite(x) and .02 <= x <= .98
                                      for x in point) for point in points)):
                raise ValueError("眼跳采样点必须位于屏幕内")
            if (not isinstance(changes, list) or len(changes) != 2
                    or not all(isinstance(x, (int, float)) and math.isfinite(x) for x in changes)
                    or not 500 <= changes[0] < changes[1] <= duration - 500):
                raise ValueError("眼跳时刻无效")
    required_splits = ("train", "test") if "spatial_v1" in stages else ("train", "validation", "test")
    for split in required_splits:
        local = [step for step in plan if step["split"] == split]
        if "spatial_v1" in stages:
            if len(local) < 2:
                raise ValueError("第一阶段每份数据至少需要 2 条直线")
        elif "events_v1" in stages:
            if len(local) < 3 or any(s.get("motion_profile") != "jump" for s in local):
                raise ValueError("每份事件数据至少需要3个跳转小段")
        elif sum(step.get("motion_profile") == "anchor" for step in local) < 5 or sum(step.get("motion_profile") != "anchor" for step in local) < 6:
            raise ValueError("每份数据至少需要 5 段静止注视和 6 段动态追踪")
    if "spatial_v1" in stages:
        versions = [step.get("plan_version") for step in plan]
        if 10 in versions:
            _validate_edge_spatial_plan(plan)
        elif any(isinstance(version, (int, float)) and version >= 6 for version in versions):
            _validate_fixed_head_spatial_plan(plan)
        elif any(step.get("head_start") or (isinstance(version, (int, float)) and version >= 5)
                 for step, version in zip(plan, versions)):
            _validate_head_varied_spatial_plan(plan)
    if "events_v1" in stages and any(s.get("plan_version", 0) >= 8 for s in plan):
        version = plan[0].get("plan_version")
        if version not in (8, 9) or any(s.get("plan_version") != version for s in plan):
            raise ValueError("分距离眼跳计划必须使用一致的第八或第九版协议")
        bands = ("short", "medium", "long")
        if version == 8:
            if len(plan) != 48:
                raise ValueError("第八版分距离计划必须包含48段")
            for split, repeats in (("train", 2), ("validation", 1), ("test", 1)):
                counts = Counter((s.get("amplitude_band"), s.get("direction_axis")) for s in plan if s["split"] == split)
                if counts != Counter({(band, axis): repeats for band in bands for axis in range(4)}):
                    raise ValueError("每份数据必须均衡覆盖三种距离和四个方向轴")
        else:
            counts = Counter((s.get("amplitude_band"), s.get("direction_axis")) for s in plan)
            if len(plan) != 12 or counts != Counter({(band, axis): 1 for band in bands for axis in range(4)}):
                raise ValueError("第九版12段计划必须整体覆盖三种距离和四个方向轴")
            for split, repeats in (("train", 2), ("validation", 1), ("test", 1)):
                local = [s for s in plan if s["split"] == split]
                if Counter(s["amplitude_band"] for s in local) != Counter({band: repeats for band in bands}):
                    raise ValueError("第九版必须按6/3/3划分，每份数据均衡覆盖三种距离")
                if len({s["direction_axis"] for s in local}) < (4 if split == "train" else 3):
                    raise ValueError("第九版每份数据的方向覆盖不足")
        for step in plan:
            lo, hi = {"short": (.06, .10), "medium": (.18, .24), "long": (.38, .48)}[step["amplitude_band"]]
            if any(not lo-1e-8 <= math.dist(a, b) <= hi+1e-8 for a, b in zip(step["points"], step["points"][1:])):
                raise ValueError("实际跳距与声明的距离分层不符")
            a, b, back = step["points"]
            angle = math.atan2(b[1]-a[1], b[0]-a[0])
            axis_error = (angle-step["direction_axis"]*math.pi/4+math.pi/2) % math.pi-math.pi/2
            if abs(axis_error) > .080001 or math.dist(a, back) > 1e-8:
                raise ValueError("实际跳转必须沿声明方向往返")
            boundaries = [0, *step["changes"], step["duration"]]
            if any(b-a < 1100 for a, b in zip(boundaries, boundaries[1:])) or step.get("settle_guard_ms") != 700:
                raise ValueError("眼跳前后必须保留反应和稳定注视时间")
    return plan


def _validate_edge_spatial_plan(plan):
    """V10 covers perimeter and center without prescribed head poses."""
    if len(plan) != 12 or Counter(s["split"] for s in plan) != Counter(train=8, test=4):
        raise ValueError("第十版空间采集必须按8/4划分12条直线")
    width, height = plan[0].get("viewport_width", 0), plan[0].get("viewport_height", 0)
    if (not isinstance(width, (int, float)) or not isinstance(height, (int, float))
            or not math.isfinite(width) or not math.isfinite(height) or width < 320 or height < 240):
        raise ValueError("空间采集屏幕尺寸无效")
    x, y = 18 / width, 18 / height
    def canonical(a, b):
        return tuple(sorted((tuple(a), tuple(b))))
    expected = {
        "train": [((x,y),(1-x,y)), ((x,1-y),(1-x,1-y)),
                  ((x,y),(x,1-y)), ((1-x,y),(1-x,1-y)),
                  ((x,.4),(1-x,.4)), ((x,.6),(1-x,.6)),
                  ((.4,y),(.4,1-y)), ((.6,y),(.6,1-y))],
        "test": [((2*x,2*y),(1-2*x,1-2*y)), ((1-2*x,2*y),(2*x,1-2*y)),
                 ((x,.5),(1-x,.5)), ((.5,y),(.5,1-y))],
    }
    rates = {step.get("sample_rate_hz") for step in plan}
    if len(rates) != 1 or not rates <= {60, 120}:
        raise ValueError("空间采集必须统一使用60或120 Hz采样上限")
    for step in plan:
        if (step.get("plan_version") != 10
                or step.get("viewport_width") != width or step.get("viewport_height") != height
                or step.get("edge_margin_px") != 18 or step.get("spatial_balance") != "equal_3x3_regions"
                or any(step.get(k) for k in ("head_pose", "head_start", "head_end", "head_mode"))):
            raise ValueError("第十版空间采集协议或屏幕尺寸不一致")
    # Rounded comparison tolerates JSON float serialization, not missing corners.
    def rails(items):
        return Counter(tuple(round(v, 9) for point in canonical(a,b) for v in point) for a,b in items)
    for split in expected:
        if rails((s["point"],s["end"]) for s in plan if s["split"] == split) != rails(expected[split]):
            raise ValueError("空间轨迹必须完整覆盖四角、四边和中央")


def _validate_head_varied_spatial_plan(plan):
    """V5 crosses opposite head actions on distinct nearby rails."""
    if len(plan) != 16:
        raise ValueError("第五版第一阶段必须恰好包含 16 条直线")
    if any(not isinstance(step.get("plan_version"), (int, float)) or step["plan_version"] < 5 for step in plan):
        raise ValueError("头姿变化计划必须完整使用第五版协议")
    expected_counts = {"train": 8, "validation": 4, "test": 4}
    if Counter(step["split"] for step in plan) != Counter(expected_counts):
        raise ValueError("第五版第一阶段必须按 8/4/4 划分")
    directions = {
        "yaw": {"head_left", "head_right"},
        "pitch": {"head_up", "head_down"},
    }
    pairs = {}
    rails = set()
    for step in plan:
        pair_id, axis = step.get("pair_id"), step.get("head_axis")
        start, end = step.get("head_start"), step.get("head_end")
        if (not isinstance(pair_id, str) or not pair_id.startswith(step["split"] + "-pair-")
                or axis not in directions or {start, end} != directions[axis]):
            raise ValueError("第五版直线必须声明同轴且方向相反的起止头姿")
        canonical = tuple(sorted((tuple(step["point"]), tuple(step["end"]))))
        if canonical in rails:
            raise ValueError("第五版的 16 条轨道必须互不重复，不能往返重放")
        rails.add(canonical)
        pairs.setdefault(pair_id, []).append(step)
    for pair_id, pair in pairs.items():
        if len(pair) != 2 or pair[0]["split"] != pair[1]["split"]:
            raise ValueError(f"头姿轨迹对 {pair_id} 必须恰好包含同一划分的两条")
        first, second = pair
        vectors = [[step["end"][i] - step["point"][i] for i in range(2)] for step in pair]
        lengths = [math.hypot(*vector) for vector in vectors]
        parallel = sum(vectors[0][i] * vectors[1][i] for i in range(2)) / (lengths[0] * lengths[1])
        midpoints = [[(step["point"][i] + step["end"][i]) / 2 for i in range(2)] for step in pair]
        separation = math.dist(midpoints[0], midpoints[1])
        modes = {step.get("head_mode") for step in pair}
        family = first.get("head_cue_family")
        if (first["head_axis"] != second["head_axis"] or first["head_start"] != second["head_end"]
                or first["head_end"] != second["head_start"] or parallel < .98
                or not .025 <= separation <= .12 or family != second.get("head_cue_family")
                or family not in {"relative", "absolute"}
                or (family == "relative" and modes != {"follow", "counter"})
                or (family == "absolute" and modes != {"absolute"})):
            raise ValueError(f"头姿轨迹对 {pair_id} 必须是相邻不同轨道，并分配相反头动")
    for split, count in expected_counts.items():
        local = [step for step in plan if step["split"] == split]
        if Counter(step["head_axis"] for step in local) != Counter({"yaw": count // 2, "pitch": count // 2}):
            raise ValueError("每份数据必须等量覆盖左右转头与抬头低头")


def _validate_fixed_head_spatial_plan(plan):
    """V6 uses unique full-screen rails under four fixed head directions."""
    two_sets = all(step.get("plan_version", 0) >= 7 for step in plan)
    if len(plan) != (12 if two_sets else 16):
        raise ValueError("固定头姿第一阶段需要 12 条直线（旧版为16条）")
    if any(not isinstance(step.get("plan_version"), (int, float)) or step["plan_version"] < 6 for step in plan):
        raise ValueError("固定头姿计划必须完整使用第六版协议")
    expected_counts = {"train": 8, "test": 4} if two_sets else {"train": 8, "validation": 4, "test": 4}
    if Counter(step["split"] for step in plan) != Counter(expected_counts):
        raise ValueError("第一阶段数据划分与计划版本不匹配")
    poses = {"head_left", "head_right", "head_up", "head_down"}
    rails = set()
    orientations = {}
    for step in plan:
        pose = step.get("head_pose")
        if (pose not in poses or step.get("head_mode") != "fixed"
                or step.get("sample_rate_hz") != 60):
            raise ValueError("第六版直线必须指定四向固定头姿和 60 Hz 采样")
        if step.get("head_start") or step.get("head_end"):
            raise ValueError("固定头姿采样期间不能安排头部转动")
        a, b = step["point"], step["end"]
        dx, dy = b[0] - a[0], b[1] - a[1]
        horizontal = abs(dx) >= .86 and abs(dy) <= .03 and min(a[0], b[0]) <= .08 and max(a[0], b[0]) >= .92
        vertical = abs(dy) >= .86 and abs(dx) <= .03 and min(a[1], b[1]) <= .08 and max(a[1], b[1]) >= .92
        if not (horizontal or vertical):
            raise ValueError("第六版的每条轨道都必须从屏幕一侧延伸到另一侧")
        canonical = tuple(sorted((tuple(a), tuple(b))))
        if canonical in rails:
            raise ValueError("第六版的 16 条长轨道必须互不重复")
        rails.add(canonical)
        orientations.setdefault((step["split"], pose), set()).add("horizontal" if horizontal else "vertical")
    for split, count in expected_counts.items():
        local = [step for step in plan if step["split"] == split]
        expected_pose_count = count // 4
        if Counter(step["head_pose"] for step in local) != Counter({pose: expected_pose_count for pose in poses}):
            raise ValueError("每份数据必须均衡覆盖左转、右转、抬头和低头")
    for pose in poses:
        if orientations.get(("train", pose)) != {"horizontal", "vertical"}:
            raise ValueError("训练集的每种固定头姿都必须同时覆盖水平和垂直长轨道")


def _spatial_endpoint_coverage(attempt):
    """Measure observed stable time, without counting gaps or transition labels."""
    result = {}
    ordered = sorted(attempt, key=lambda r: r["pc_ms"])
    for name, at_endpoint in (("start", lambda p: p <= .1), ("end", lambda p: p >= .9)):
        count, duration, previous = 0, 0., None
        for row in ordered:
            eligible = (row["weight"] > .5 and row["phase"] == "anchor"
                        and at_endpoint(row.get("drag_progress", 0.)))
            if eligible:
                count += 1
                if previous is not None and not row.get("reset", False):
                    delta = row["pc_ms"] - previous["pc_ms"]
                    if (0 < delta <= 50.001
                            and row.get("clock_epoch") == previous.get("clock_epoch")):
                        duration += delta
            previous = row if eligible else None
        result[name] = dict(frames=count, observed_ms=duration)
    return result


def coverage_report(rows, plan, *, events=()):
    """Each planned trial must have useful observations; repetition cannot hide holes."""
    # Completion is a browser lifecycle event, not an eye-frame label. The UI
    # immediately advances after emitting it, so align_frames correctly rejects
    # its interval across capture segments (or no camera frame lands in it).
    # Keep this evidence separate from causal labels and bind it to one attempt.
    completions = {}
    for event in events:
        if (event.get("trial_complete") and event.get("visible", False)
                and event.get("phase") == "anchor" and event.get("sync_rtt_ms", 999) <= 40):
            key = (event.get("trial_id"), event.get("capture_segment", 0))
            completions.setdefault(key, []).append(event["pc_ms"])
    groups = {}
    for row in rows:
        if row["valid"] and row.get("stimulus_supported"):
            groups.setdefault(row["trial_id"], []).append(row)
    trials, missing, accepted_segments = [], [], []
    for step in plan:
        # A restart is a fresh contiguous attempt, never stitched to the old half.
        candidates = groups.get(step["trial_id"], [])
        attempts = {}
        for row in candidates:
            attempts.setdefault(row.get("capture_segment", 0), []).append(row)
        best = {"valid_frames": 0, "settled_frames": 0, "span_ms": 0., "accepted": False, "capture_segment": None}
        for segment, attempt in attempts.items():
            details = {}
            span = max(r["pc_ms"] for r in attempt) - min(r["pc_ms"] for r in attempt)
            settled = sum(r["weight"] > .5 for r in attempt)
            anchor = step.get("motion_profile") == "anchor"
            ok = len(attempt) >= (20 if anchor else 55) and span >= step["duration"] * .70 and (not anchor or settled >= 12)
            if step.get("calibration_stage") == "spatial_v1":
                # Waiting at the start is not a completed drag. Require observed
                # progress, useful rail labels, and settled endpoint observations.
                first_ms = min(r["pc_ms"] for r in attempt)
                last_ms = max(r["pc_ms"] for r in attempt)
                completed = any(r.get("trial_complete") for r in attempt) or any(
                    first_ms <= end_ms <= last_ms + 120
                    for end_ms in completions.get((step["trial_id"], segment), ()))
                if anchor:
                    ok = ok and completed
                else:
                    rail_frames = sum(r.get("constraint") is not None for r in attempt)
                    progress = [r.get("drag_progress", 0.) for r in attempt if r["phase"] == "pursuit"]
                    endpoints = _spatial_endpoint_coverage(attempt)
                    # 900 ms holds leave only ~200 ms after the 700 ms guard.
                    # At 30 FPS, a fixed 12-frame total rejects complete drags
                    # depending on camera/display phase. Require time at BOTH
                    # endpoints plus a small independent-observation floor.
                    checks = dict(completion=completed, rail=rail_frames >= 12,
                                  progress=len(progress) >= 20 and min(progress) <= .1 and max(progress) >= .9,
                                  start_settled=endpoints["start"]["frames"] >= 4 and endpoints["start"]["observed_ms"] >= 100 - 1e-3,
                                  end_settled=endpoints["end"]["frames"] >= 4 and endpoints["end"]["observed_ms"] >= 100 - 1e-3,
                                  settled_duration=sum(e["observed_ms"] for e in endpoints.values()) >= 250 - 1e-3)
                    ok = all(checks.values())
                    details = dict(endpoint_coverage=endpoints, rail_frames=rail_frames,
                                   rejection_reasons=[name for name, passed in checks.items() if not passed])
            if step.get("calibration_stage") == "events_v1":
                # A frozen tab or a long first fixation cannot replace later landings.
                boundaries = [0, *step.get("changes", [1200, 2700]), step["duration"] + 1]
                counts = [sum(r["weight"] > .5 and lo + 700 <= r.get("trial_age_ms", -1) < hi for r in attempt)
                          for lo, hi in zip(boundaries, boundaries[1:])]
                ok = ok and all(n >= 4 for n in counts)
            item = dict(valid_frames=len(attempt), settled_frames=settled, span_ms=span, accepted=ok, capture_segment=segment, **details)
            if (ok, span, len(attempt)) > (best["accepted"], best["span_ms"], best["valid_frames"]):
                best = item
        item = dict(trial_id=step["trial_id"], split=step["split"], profile=step.get("motion_profile"), **best)
        trials.append(item)
        if not best["accepted"]:
            missing.append(step["trial_id"])
        else:
            accepted_segments.append(best["capture_segment"])
    return {"ready": not missing, "missing_trials": missing, "trials": trials,
            "accepted_segments": accepted_segments, "completed": len(plan)-len(missing), "total": len(plan),
            "split_trials": dict(Counter(t["split"] for t in trials if t["accepted"])),
            "policy": "spatial line: completed drag, >=12 rail frames, >=20 pursuit frames spanning 0.1 to 0.9; each endpoint >=4 settled frames/100 ms, combined >=250 ms (gaps >50 ms excluded); other trials: >=70% planned span and profile frame minima"}
