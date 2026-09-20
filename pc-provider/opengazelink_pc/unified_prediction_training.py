"""Train shared-state prediction and compare filter/state ablations by trial."""
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import numpy as np
import torch
from torch.nn import functional as F
from .unified_prediction import SCHEMA, HORIZONS, UnifiedHistory, UnifiedNetwork, EventTrajectoryNetwork, cap_delta
from .prediction_dataset import make_examples
from .prediction_training import scores
from .video_forecast_training import replay, digest
from .video_session import write_json


def quality_scores(pred, data, frames, mask, wh):
    result = scores(pred, data, frames, mask, wh)
    result["horizons"] = {}
    for h in HORIZONS:
        if h <= 0:
            continue
        local = mask & np.isclose(data["horizon"][:, 0], h/1000)
        result["horizons"][str(int(h))] = scores(pred, data, frames, local, wh)
    if "event_age_ms" in frames and "next_event_ms" in frames:
        fi = data["frame"]
        age = frames["event_age_ms"][fi]
        until = frames["next_event_ms"][fi]
        local = mask & np.isclose(data["horizon"][:, 0], .085)
        regimes = {"event_first_25ms": (age >= 0) & (age < 25),
                   "event_after_25ms": age >= 25,
                   "future_event_within_horizon": (age < 0) & (until <= 85),
                   "no_event_in_horizon": (age < 0) & (until > 85)}
        result["event_timing_85ms"] = {name: scores(pred, data, frames, local & group, wh)
                                       for name, group in regimes.items()}
        result["event_timing_notice"] = "Offline proxy event times, not physiological onset truth or causal inputs."
    local = np.flatnonzero(mask & np.isclose(data["horizon"][:, 0], .085) & data["current_anchor"])
    if len(local):
        fi = data["frame"][local]
        point = frames["raw"][fi]+pred[local]
        err = np.linalg.norm((point-data["stimulus_target"][local])*(wh-1), axis=1)
        order = np.argsort(fi); fi, point = fi[order], point[order]
        adjacent = (np.diff(fi)==1) & (frames["segments"][fi[1:]]==frames["segments"][fi[:-1]])
        steps = np.linalg.norm(np.diff(point, axis=0)[adjacent]*(wh-1), axis=1)
        result["settled"] = {"samples":len(fi), "mean_px":float(err.mean()), "p95_px":float(np.percentile(err, 95)),
            "step_mean_px":float(steps.mean()) if len(steps) else 0., "step_p95_px":float(np.percentile(steps, 95)) if len(steps) else 0.}
    return result


def selection_objective(candidate, baseline):
    # Relative costs expose the motion/stability tradeoff, with no arbitrary
    # four-pixel publication cliff. Tail and jitter carry explicit costs.
    ratio = lambda a,b: a/max(b, 1.)
    g, b = candidate["groups"], baseline["groups"]
    cost = .25*ratio(candidate["mean_proxy_px"], baseline["mean_proxy_px"])
    total_weight = .35
    for group, amount in (("pursuit", .25), ("saccade_proxy", .20)):
        if group in g and group in b:
            cost += amount*ratio(g[group]["mean_px"], b[group]["mean_px"])
            total_weight += amount
    cost += .1*ratio(candidate["p95_proxy_px"], baseline["p95_proxy_px"])
    if "settled" in candidate and "settled" in baseline:
        for key, weight in (("mean_px", .07), ("p95_px", .03), ("step_mean_px", .1)):
            cost += weight*ratio(candidate["settled"][key], baseline["settled"][key])
            total_weight += weight
    return cost/total_weight


def dynamic_accuracy_regressions(candidate, baseline):
    """Apply the existing accuracy limits at each supported horizon as well."""
    regressions = []

    def compare(a, b, scope):
        pairs = [(a, b, scope, 'mean_proxy_px', 'p95_proxy_px')]
        for name in ('pursuit', 'saccade_proxy'):
            group_a, group_b = a.get('groups', {}).get(name), b.get('groups', {}).get(name)
            if group_a and group_b:
                pairs.append((group_a, group_b, scope + '/' + name, 'mean_px', 'p95_px'))
        for left, right, label, mean, tail in pairs:
            for metric, factor in ((mean, 1.), (tail, 1.02)):
                limit = right[metric] * factor
                if left[metric] > limit:
                    regressions.append(dict(scope=label, metric=metric, candidate=left[metric],
                                            baseline=right[metric], limit=limit))

    compare(candidate, baseline, 'all_horizons')
    for horizon, previous in baseline.get('horizons', {}).items():
        current = candidate.get('horizons', {}).get(horizon)
        if current and current.get('samples', 0) and previous.get('samples', 0):
            compare(current, previous, horizon + 'ms')
    return regressions


def preserves_dynamic_accuracy(candidate, baseline):
    """Long-horizon gains cannot conceal regressions at the operating horizon."""
    return not dynamic_accuracy_regressions(candidate, baseline)


def balanced_draw(data, mask, rng, per_phase=2400):
    """Equal phase/horizon budgets, then equal trials, spatial cells and frames.

    Training indices only. A long dwell or 13 correlated horizons cannot outweigh
    a short trajectory or a less populated screen region.
    """
    def allocations(total, size):
        counts = np.full(size, total // size, dtype=np.int64)
        counts[rng.permutation(size)[:total % size]] += 1
        return counts

    samples = []
    for phase in range(3):
        ids = np.flatnonzero(mask & (data["phase"] == phase) & (data.get("phase_conf", np.ones(len(mask))) >= .5))
        if not len(ids):
            continue
        horizons = np.unique(np.round(data["horizon"][ids, 0], 6)) if "horizon" in data else np.array([0.])
        for horizon, horizon_budget in zip(horizons, allocations(per_phase, len(horizons))):
            horizon_ids = ids if "horizon" not in data else ids[np.isclose(data["horizon"][ids, 0], horizon)]
            trials = np.unique(data["trial"][horizon_ids])
            for trial, trial_budget in zip(trials, allocations(horizon_budget, len(trials))):
                horizon_local = horizon_ids[data["trial"][horizon_ids] == trial]
                cells = np.clip((data["stimulus_target"][horizon_local]*3).astype(int), 0, 2)
                cell_ids = cells[:, 0]+3*cells[:, 1]
                unique = np.unique(cell_ids)
                for cell, count in zip(unique, allocations(trial_budget, len(unique))):
                    pool = horizon_local[cell_ids == cell]
                    frames = np.unique(data["frame"][pool])
                    for frame in rng.choice(frames, count, replace=True):
                        candidates = pool[data["frame"][pool] == frame]
                        samples.append(rng.choice(candidates))
    result = np.asarray(samples, dtype=np.int64)
    rng.shuffle(result)
    return result


def history_inputs(values, rows, segments, sample_interval_ms, intrinsic_stability=False, recent_interval_ms=0., recent_velocity_expert=False):
    history = UnifiedHistory(sample_interval_ms=sample_interval_ms, intrinsic_stability=intrinsic_stability,
                             recent_interval_ms=recent_interval_ms, recent_velocity_expert=recent_velocity_expert)
    contexts, experts, counts = [], [], []
    for i, row in enumerate(rows):
        c, e, n = history.update(row['source_ms'], values['raw'][i], values['stable'][i],
                                values['eye_features'][i], values['eye_hidden'][i],
                                i == 0 or segments[i] != segments[i-1])
        contexts.append(c); experts.append(e); counts.append(n)
    return np.asarray(contexts), np.asarray(experts), np.asarray(counts)


def train_unified(session_path, model_path, output=None, *, cache=None, epochs=60, progress=print, cancelled=lambda:False, conservative=True, publish_model_path=None, recent_interval_ms=0., event_model=False, consistency_weight=0., incumbent_model_path=None):
    if not 1 <= epochs <= 300:
        raise ValueError("epochs must be between 1 and 300")
    if not np.isfinite(recent_interval_ms) or recent_interval_ms < 0 or recent_interval_ms > 25:
        raise ValueError("recent interval must be between 0 and 25 ms")
    if not np.isfinite(consistency_weight) or not 0 <= consistency_weight <= 2:
        raise ValueError("consistency weight must be between 0 and 2")
    from .training_runtime import configure_training_threads
    training_threads = configure_training_threads()
    session_path, model_path = map(Path, (session_path, model_path))
    output = Path(output) if output else session_path/"prediction-runs"/datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S-%fZ")
    output.mkdir(parents=True, exist_ok=False)
    sources = {}
    from .paths import RESOURCE_ROOT
    for name in ("unified_prediction.py", "unified_prediction_training.py", "prediction_dataset.py", "motion_labels.py", "prediction_training.py", "video_forecast_training.py"):
        path = Path(__file__).with_name(name)
        if not path.is_file():
            path = RESOURCE_ROOT/"provenance"/"opengazelink_pc"/name
        if path.is_file():
            shutil.copy2(path, output/name); sources[name] = digest(path)
    run = {"sampling":"fixed phase/horizon/trial/3x3-cell/frame budgets using train only", "schema":SCHEMA, "session":str(session_path.resolve()), "source_sha256":sources, "epochs":epochs, "conservative":conservative,
           "split_notice":"Whole trial splits shared with spatial training; selection chooses model/epoch; final test checks only the selected candidate. Same-session weak labels, not external eye-tracker truth. Retained captures may have been inspected in earlier experiments."}
    run.update(history_recent_interval_ms=recent_interval_ms, event_model=event_model,
               consistency_weight=consistency_weight)
    write_json(output/"run.json", dict(run, state="running"))
    if cache:
        cache = Path(cache)
        meta = json.loads(model_path.read_text(encoding="utf-8"))
        cached_meta = json.loads((cache/"base-metadata.json").read_text(encoding="utf-8"))
        cached_session = Path(json.loads((cache/"run.json").read_text(encoding="utf-8"))["session"])
        if cached_session.resolve() != session_path.resolve() or meta != cached_meta:
            raise ValueError("replay cache does not match session and frozen base")
        if digest(model_path.with_name(meta["variants"]["conditioned_video"]["module_file"])) != meta["variants"]["conditioned_video"]["module_sha256"]:
            raise ValueError("base model checksum mismatch")
        values = dict(np.load(cache/"replay.npz", allow_pickle=False))
        rows = [json.loads(line) for line in (cache/"alignment.jsonl").read_text(encoding="utf-8").splitlines()]
        cfg = json.loads((session_path/"session.json").read_text(encoding="utf-8"))["config"]
        write_json(output/"cache.json", {"path":str(cache.resolve()), "replay_sha256":digest(cache/"replay.npz"),
                                      "alignment_sha256":digest(cache/"alignment.jsonl")})
    else:
        values, rows, meta, cfg = replay(session_path, model_path, output, progress, cancelled, instantaneous=True)
    frames, data, audit = make_examples(values, rows, horizons=HORIZONS)
    data["current_anchor"] = np.array([rows[i]["weight"]>.5 for i in data["frame"]]) & data["anchor"]
    context, experts, counts = history_inputs(values, rows, frames['segments'], 25., True, recent_interval_ms)
    supported = counts[data['frame']] >= 4
    data = {key:value[supported] for key,value in data.items()}
    audit['warmup_excluded_examples'] = int((~supported).sum())
    audit['usable_examples'] = int(supported.sum())
    np.savez_compressed(output/"motion-examples.npz", **data)
    np.savez_compressed(output/"motion-frames.npz", **frames, context=context, experts=experts)
    write_json(output/"label-audit.json", audit)
    masks = [data["split"]==i for i in range(3)]
    trials = [set(data["trial"][m]) for m in masks]
    if any(m.sum()<100 for m in masks) or any(trials[a]&trials[b] for a,b in ((0,1),(1,2),(0,2))):
        raise ValueError("insufficient or overlapping trial splits")
    fi = data["frame"]; wh = np.array([cfg["screen_width"],cfg["screen_height"]], np.float32)
    raw, stable = frames["raw"][fi], frames["stable"][fi]
    max_lead = cfg.get("extrapolation_max_lead_fraction", .12)
    ids = np.unique(fi[masks[0]])
    mean, scale = context[ids].mean(0), np.maximum(context[ids].std(0), .005)
    ctx, exp = torch.from_numpy(context), torch.from_numpy(experts)
    h = torch.tensor(data["horizon"], dtype=torch.float32)
    y = torch.tensor(data["delta"], dtype=torch.float32)
    # Use the same reconstructed current/future label that evaluation scores.
    # Copying stable-raw here teaches exactly zero denoising when filtering is
    # disabled, even though the offline teacher removes observation noise.
    target = y.clone()
    anchor = torch.from_numpy(data["current_anchor"])
    # Supervise adjacent outputs only within the same settled training target.
    # Comparing positions, not deltas, cancels the raw observation's noise.
    predecessor = np.full(len(fi),-1,np.int64)
    lookup = {(int(fi[i]),round(float(data['horizon'][i,0])*1000)):i
              for i in np.flatnonzero(masks[0] & data['current_anchor'])}
    for (frame,horizon),i in lookup.items():
        j = lookup.get((frame-1,horizon),-1)
        if (j>=0 and frames['segments'][frame]==frames['segments'][frame-1]
                and np.linalg.norm(data['stimulus_target'][i]-data['stimulus_target'][j])<.001):
            predecessor[i]=j
    phase = torch.tensor(data["phase"], dtype=torch.long)
    weight = torch.tensor(data["quality"]*data.get('interpolation_confidence',1.), dtype=torch.float32)
    baselines = {name:{split:quality_scores(pred,data,frames,m,wh) for split,m in zip(("train","selection","test"), masks[0:])}
                 for name,pred in (("stable",stable-raw),("raw",np.zeros_like(raw)))}
    results = {}; artifacts = {}
    variants = (("shared_filter",True,True),("shared_no_filter",False,True),("history_only",True,False),("history_no_filter",False,False))
    if conservative:
        variants = (("history_only",True,False),("history_no_filter",False,False))
    if event_model:
        variants = (("event_trajectory",True,False),)
    for name, use_filter, use_shared in variants:
        torch.manual_seed(9421)
        rng = np.random.default_rng(9421)
        model = (EventTrajectoryNetwork(mean, scale) if event_model else
                 UnifiedNetwork(mean, scale, use_filter, use_shared, residual_scale=0. if conservative else .025))
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.01)
        best, best_cost, best_epoch = None, float("inf"), 0
        records = []
        def evaluate(mask):
            prediction = np.zeros_like(raw)
            probabilities = np.zeros((len(raw), 3), np.float32)
            mixtures = np.zeros((len(raw), 6), np.float32)
            model.eval()
            with torch.inference_mode():
                for batch in np.array_split(np.flatnonzero(mask), max(1,int(np.ceil(mask.sum()/2048)))):
                    delta, logits, mixture = model(ctx[fi[batch]], exp[fi[batch]], h[batch])
                    prediction[batch] = cap_delta(delta.numpy(), wh-1, max_lead)
                    probabilities[batch] = logits.softmax(-1).numpy()
                    if mixtures.shape[1] != mixture.shape[1]:
                        mixtures = np.zeros((len(raw), mixture.shape[1]), np.float32)
                    mixtures[batch] = mixture.numpy()
            return prediction, probabilities, mixtures
        for epoch in range(epochs):
            if cancelled():
                raise RuntimeError("prediction training cancelled")
            draw = balanced_draw(data, masks[0], rng)
            rng.shuffle(draw); model.train(); losses=[]
            for batch in np.array_split(draw, max(1,int(np.ceil(len(draw)/256)))):
                if cancelled():
                    raise RuntimeError("prediction training cancelled")
                delta, logits, mixture = model(ctx[fi[batch]], exp[fi[batch]], h[batch])
                pixels = torch.tensor(wh-1)
                norm = torch.linalg.vector_norm(delta*pixels, dim=1, keepdim=True).clamp_min(1e-9)
                delta = delta*(max_lead*float(np.linalg.norm(wh-1))/norm).clamp(max=1)
                distance = F.smooth_l1_loss(delta, target[batch], beta=.008, reduction="none").mean(1)
                anchor_weight = 1+2*anchor[batch].float() if conservative else 1.
                effective_weight = weight[batch]*anchor_weight
                loss = (distance*effective_weight).sum()/effective_weight.sum().clamp_min(1e-8)+.003*(F.cross_entropy(logits, phase[batch], reduction="none")*torch.as_tensor(data["phase_conf"][batch],dtype=torch.float32)).mean()
                if consistency_weight:
                    valid = predecessor[batch]>=0
                    previous_batch = predecessor[batch][valid]
                    if len(previous_batch):
                        previous_delta = model(ctx[fi[previous_batch]],exp[fi[previous_batch]],h[previous_batch])[0]
                        previous_norm = torch.linalg.vector_norm(previous_delta*pixels,dim=1,keepdim=True).clamp_min(1e-9)
                        previous_delta = previous_delta*(max_lead*float(np.linalg.norm(wh-1))/previous_norm).clamp(max=1)
                        step = delta[valid]-previous_delta+torch.tensor(raw[batch][valid]-raw[previous_batch])
                        teacher_step = torch.tensor(data['target'][batch][valid]-data['target'][previous_batch])
                        loss = loss+consistency_weight*F.smooth_l1_loss(step,teacher_step,beta=.004)
                if event_model:
                    valid = data['landing_valid'][batch]
                    if valid.any():
                        _,_,_,endpoint,remaining = model.state(ctx[fi[batch]], exp[fi[batch]])
                        landing_target = torch.tensor(data['landing'][batch][valid],dtype=torch.float32)
                        remaining_target = torch.tensor(data['remaining'][batch][valid],dtype=torch.float32)
                        loss = loss+.02*F.smooth_l1_loss(endpoint[valid],landing_target,beta=.008)
                        loss = loss+.001*F.smooth_l1_loss(remaining[valid],remaining_target,beta=.02)
                optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                optimizer.step(); losses.append(float(loss.detach()))
            pred, _, _ = evaluate(masks[1])
            score = quality_scores(pred, data, frames, masks[1], wh)
            cost = selection_objective(score, baselines["stable"]["selection"])
            records.append({"epoch":epoch+1, "cost":cost, "loss":float(np.mean(losses)), "selection":score})
            if cost < best_cost:
                best, best_cost, best_epoch = copy.deepcopy(model.state_dict()), cost, epoch+1
            if (epoch+1)%10==0:
                progress(f"{name} epoch {epoch+1}/{epochs}: selection cost {cost:.4f}, best {best_cost:.4f}")
            if epoch+1 >= 20 and epoch+1-best_epoch >= 12:
                progress(f"{name}: selection stopped improving; stopped at epoch {epoch+1}")
                break
        model.load_state_dict(best); model.eval()
        pred, probabilities, mixtures = evaluate(masks[1])
        result = {"selected_epoch":best_epoch, "selection_cost":best_cost,
                  "selection":quality_scores(pred,data,frames,masks[1],wh), "history":records}
        results[name] = result
        torch.save({"model":model.state_dict(), "use_filter":use_filter, "use_shared":use_shared,
                    "residual_scale":getattr(model, 'residual_scale', 0.), "architecture":name}, output/f"{name}.pt")
        scripted = torch.jit.trace(model, (ctx[:2], exp[:2], h[:2]), strict=False)
        scripted.save(str(output/f"{name}-runtime.pt"))
        np.savez_compressed(output/f"{name}-predictions.npz", prediction=pred, phase=probabilities, mixture=mixtures)
        artifacts[name] = pred
        write_json(output/f"{name}-report.json", result)
        progress(f"{name} finished: epoch {best_epoch}, selection cost {best_cost:.4f}")
    chosen = min(results, key=lambda name:results[name]["selection_cost"])
    candidate = results[chosen]
    model = torch.jit.load(str(output/f"{chosen}-runtime.pt"), map_location="cpu").eval()
    pred, probabilities, mixtures = evaluate(masks[0] | masks[1] | masks[2])
    candidate["train"] = quality_scores(pred,data,frames,masks[0],wh)
    candidate["test"] = quality_scores(pred,data,frames,masks[2],wh)
    np.savez_compressed(output/f"{chosen}-predictions.npz", prediction=pred, phase=probabilities, mixture=mixtures)
    test_cost = selection_objective(candidate["test"], baselines["stable"]["test"])
    accepted = candidate["selection_cost"] < .95 and test_cost < .95
    # A compatible installed head is an additional comparator. Updating the
    # sampling recipe alone must not replace a better personal prediction head.
    from .paths import DATA_DIR
    incumbent_path = Path(incumbent_model_path) if incumbent_model_path else model_path.with_name("conditioned-video-forecast.json")
    if not incumbent_path.exists() and incumbent_model_path:
        raise ValueError("requested incumbent metadata does not exist")
    if not incumbent_path.exists():
        incumbent_path = DATA_DIR / "conditioned-video-forecast.json"
    incumbent_check = None
    incumbent_regressions = {}
    if incumbent_path.exists():
        previous = json.loads(incumbent_path.read_text(encoding="utf-8"))
        if previous.get("schema") == SCHEMA and previous.get("base_sha256") == meta["variants"]["conditioned_video"]["module_sha256"]:
            file = incumbent_path.with_name(previous["module_file"])
            filter_matches = all(previous.get("filter_config", {}).get(k) == cfg.get(k) for k in previous.get("filter_config", {}))
            if filter_matches and file.exists() and digest(file) == previous.get("module_sha256"):
                model = torch.jit.load(str(file), map_location="cpu").eval()
                # Preserve the incumbent's input ABI, including older models
                # trained with a fixed frame count instead of a time grid.
                prior_context, prior_experts, _ = history_inputs(
                    values, rows, frames['segments'], previous.get('history_sample_interval_ms', 0.),
                    previous.get('intrinsic_stability', False), previous.get('history_recent_interval_ms', 0.),
                    previous.get('recent_velocity_expert', False))
                saved_ctx, saved_exp = ctx, exp
                ctx, exp = torch.from_numpy(prior_context), torch.from_numpy(prior_experts)
                prior, _, _ = evaluate(masks[1] | masks[2])
                ctx, exp = saved_ctx, saved_exp
                incumbent_check = {split:quality_scores(prior,data,frames,m,wh) for split,m in zip(("selection","test"),masks[1:])}
                accepted = accepted and all(selection_objective(candidate[split], incumbent_check[split]) < 1. for split in ("selection","test"))
                incumbent_regressions = {split: dynamic_accuracy_regressions(candidate[split], incumbent_check[split])
                                         for split in ('selection', 'test')}
                accepted = accepted and not any(incumbent_regressions.values())
    # Keep fixed-target stability bounded even when average motion error improves.
    for split in ("selection", "test"):
        settled, reference = candidate[split].get("settled"), baselines["stable"][split].get("settled")
        if settled and reference:
            accepted = accepted and settled["mean_px"] <= reference["mean_px"]*1.10+2 and settled["step_mean_px"] <= reference["step_mean_px"]*1.75+2

    report = {"schema":SCHEMA, "accepted":accepted, "published":False, "chosen":chosen, "test_cost":test_cost,
              "incumbent":incumbent_check, "incumbent_regressions":incumbent_regressions,
              "baselines":baselines, "variants":{k:{a:b for a,b in v.items() if a!="history"} for k,v in results.items()},
              "label_audit":audit, **run}
    metadata = {"schema":SCHEMA, "accepted":accepted, "training_cpu_threads": training_threads,
                "base_sha256":meta["variants"]["conditioned_video"]["module_sha256"],
                "module_file":f"{chosen}-runtime.pt", "module_sha256":digest(output/f"{chosen}-runtime.pt"),
                "max_horizon_ms":250., "horizons_ms":HORIZONS, "training_directory":str(output.resolve()),
                "history_sample_interval_ms":25.,
                "history_recent_interval_ms":recent_interval_ms,
                "intrinsic_stability":True,
                "filter_config":{},
                "filter_policy":"intrinsic_causal_mean" if "no_filter" not in chosen else "fallback_only",
                "shared_temporal_state":chosen.startswith("shared"), "label_source":"offline_reconstructed_gaze_proxy"}
    write_json(output/"conditioned-video-forecast.json", metadata)
    write_json(output/"report.json", report)
    write_json(output/"run.json", dict(run, state="complete"))
    if publish_model_path is not None and accepted and not cancelled():
        publish_unified(output, publish_model_path)
        report["published"] = True
    return report


def publish_unified(output, model_path):
    """Publish only a validated module against the unchanged spatial model."""
    output, model_path = Path(output), Path(model_path)
    metadata = json.loads((output/"conditioned-video-forecast.json").read_text(encoding="utf-8"))
    report = json.loads((output/"report.json").read_text(encoding="utf-8"))
    active = json.loads(model_path.read_text(encoding="utf-8"))
    base = active["variants"]["conditioned_video"]
    if not metadata["accepted"] or not report["accepted"]:
        raise ValueError("unaccepted prediction candidate")
    if metadata["base_sha256"] != base["module_sha256"] or digest(model_path.with_name(base["module_file"])) != base["module_sha256"]:
        raise ValueError("spatial model changed since prediction training")
    module_path = output/metadata["module_file"]
    if digest(module_path) != metadata["module_sha256"]:
        raise ValueError("prediction module checksum mismatch")
    destination = model_path.with_name("conditioned-unified-prediction-"+metadata["module_sha256"][:16]+".pt")
    shutil.copy2(module_path, destination)
    target = model_path.with_name("conditioned-video-forecast.json")
    if target.exists():
        backup = output/"previous-forecast.json"
        if not backup.exists():shutil.copy2(target, backup)
    metadata = dict(metadata, module_file=destination.name)
    write_json(target, metadata)
    report["published"] = True
    report["published_metadata"] = str(target.resolve())
    write_json(output/"report.json", report)
