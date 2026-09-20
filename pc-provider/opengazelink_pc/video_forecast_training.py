"""Train a short-horizon head without changing the current-gaze model."""
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import torch
from torch.nn import functional as F

from .one_euro import OneEuroFilter2D
from .normalized_eye import screen_camera_origin
from .video_dataset import align_frames, read_jsonl
from .video_training import project
from .video_forecast import ForecastHistory, SCHEMA, HORIZONS
from .video_forecast_network import ForecastNetwork
from .video_session import write_json


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def future_pairs(rows, segment, horizon):
    """Bracket by source time, never across gaps, splits or stimulus blocks."""
    times = np.array([r["source_ms"] for r in rows])
    result = []
    for start in range(len(rows)):
        # Searching only within the contiguous segment also supports clock restarts.
        end = start + 1
        while end < len(rows) and segment[end] == segment[start] and times[end] < times[start] + horizon:
            end += 1
        if end >= len(rows) or segment[end] != segment[start] or end == start:
            continue
        left = end - 1
        if rows[start]["block"] != rows[end]["block"] or times[end] - times[left] > 100:
            continue
        fraction = (times[start] + horizon - times[left]) / (times[end] - times[left])
        if 0 <= fraction <= 1:
            result.append((start, left, end, float(fraction)))
    return result


def replay(session_path, model_path, output, progress=print, cancelled=lambda:False, *, instantaneous=False):
    meta = json.loads(model_path.read_text(encoding="utf-8"))
    item = meta["variants"]["conditioned_video"]
    if item.get("feature_dim") != 404:
        raise ValueError("forecast requires the CNN-connected 404-feature VIDEO model")
    model_file = model_path.with_name(item["module_file"])
    if digest(model_file) != item["module_sha256"]:
        raise ValueError("VIDEO module checksum mismatch")
    shutil.copy2(model_path, output / "base-metadata.json")
    shutil.copy2(model_file, output / "base.pt")
    cfg = json.loads((session_path/"session.json").read_text(encoding="utf-8"))["config"]
    wh = np.array([cfg["screen_width"],cfg["screen_height"]], np.float32)
    position = [cfg[f"camera_offset_{axis}_cm"] for axis in "xyz"]
    if meta["screen"] != {"width":int(wh[0]),"height":int(wh[1])} or meta["camera_position_screen_cm"] != position or meta["screen_diagonal_inches"] != cfg["screen_diagonal_inches"]:
        raise ValueError("forecast capture and gaze-model geometry must match")
    origin = torch.tensor(screen_camera_origin(*wh.astype(int),cfg["screen_diagonal_inches"],position),dtype=torch.float32)
    size = torch.tensor(cfg["screen_diagonal_inches"]*2.54*wh/np.linalg.norm(wh),dtype=torch.float32)
    session = json.loads((session_path/"session.json").read_text(encoding="utf-8"))
    rows = align_frames(read_jsonl(session_path/"frames.jsonl"),read_jsonl(session_path/"stimulus.jsonl"),
                        discarded_segments=session.get("discarded_segments", []))
    rows = [r for r in rows if r["valid"] and r.get("input")]
    splits = np.array([1 if r["block"].startswith("validation-") else 2 if r["block"].startswith("test-") else 0 for r in rows])
    model = torch.jit.load(str(model_file),map_location="cpu").eval()
    smoothing = OneEuroFilter2D(cfg["one_euro_min_cutoff"],cfg["one_euro_beta"],cfg["one_euro_derivative_cutoff"])
    history = ForecastHistory()
    hidden, previous = torch.zeros(1,64),torch.zeros(1,404)
    stable, raw, vectors, activity, segment = [],[],[],[],[]
    eye_features, eye_hidden = [], []
    spatial = []
    group = 0
    with torch.inference_mode():
        for i,row in enumerate(rows):
            if cancelled():
                raise RuntimeError("forecast training cancelled; history retained")
            reset = (i==0 or row["reset"] or row["index"]!=rows[i-1]["index"]+1 or splits[i]!=splits[i-1]
                     or row.get("capture_segment", 0)!=rows[i-1].get("capture_segment", 0)
                     or not 5 <= row["dt_ms"] <= 100)
            if reset:
                smoothing.reset()
                history.reset()
                group += 1
            input_path = (session_path/row["input"]).resolve()
            input_path.relative_to(session_path.resolve())
            with np.load(input_path,allow_pickle=False) as z:
                images = torch.from_numpy(z["images"][:,1].copy())
                geometry = torch.from_numpy(np.concatenate((z["head"],z["points"][:,1],z["crop"]),1).astype(np.float32))
                rotation,center = torch.from_numpy(z["rotation"].copy()).float()[None],torch.from_numpy(z["center"].copy()).float()[None]
            inference = model(images,geometry,torch.tensor([[row["dt_ms"]]]),torch.tensor([[float(reset)]]),hidden,previous)
            directions,weights,hidden,previous,gate = inference[:5]
            if gate.item() >= .8:
                smoothing.reset()
                history.reset()
                group += 1
                reset = True
            xy = project(directions[None],weights[None],rotation,center,origin,size)[0].numpy()
            if instantaneous:
                current = model(images,geometry,torch.tensor([[row["dt_ms"]]]),
                    torch.ones(1,1),torch.zeros(1,64),torch.zeros(1,404))
                current_directions,current_weights = current[:2]
                spatial.append(project(current_directions[None],current_weights[None],rotation,center,origin,size)[0].numpy())
            point = np.asarray(smoothing.update(xy,row["source_ms"]/1000.)) if cfg["one_euro_enabled"] else xy
            vector,strength,_ = history.update(row["source_ms"],xy,point,previous.numpy(),hidden.numpy(),reset)
            raw.append(xy)
            eye_features.append(previous.numpy()[0].copy())
            eye_hidden.append(hidden.numpy()[0].copy())
            stable.append(point)
            vectors.append(np.zeros(483,np.float32) if vector is None else vector)
            activity.append(strength)
            segment.append(group)
            if (i+1)%1000==0:
                progress(f"forecast_replay_{i+1}_of_{len(rows)}")
    values = {"raw":np.array(raw),"stable":np.array(stable),"features":np.array(vectors),"activity":np.array(activity),
              "eye_features":np.array(eye_features), "eye_hidden":np.array(eye_hidden),
              "instantaneous":np.array(spatial),
              "split":splits,"segment":np.array(segment),"frame_index":np.array([r["index"] for r in rows])}
    np.savez_compressed(output/"replay.npz",**values)
    with (output/"alignment.jsonl").open("w",encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps({k:v for k,v in row.items() if k!="diagnostics"},ensure_ascii=False)+"\n")
    return values,rows,meta,cfg


def metrics(delta, data, mask, wh):
    error = np.linalg.norm((delta[mask]-data["delta"][mask])*(wh-1),axis=1)
    moving = np.linalg.norm(data["delta"][mask]*(wh-1),axis=1)>15
    anchor = data["anchor"][mask]
    absolute = np.linalg.norm((data["current"][mask]+delta[mask]-data["target"][mask])*(wh-1),axis=1)
    return {"samples":int(mask.sum()),"mean_future_proxy_px":float(error.mean()),"median_future_proxy_px":float(np.median(error)),
            "p95_future_proxy_px":float(np.percentile(error,95)),
            "moving_mean_future_proxy_px":float(error[moving].mean()) if moving.any() else None,
            "anchor_samples":int(anchor.sum()),"anchor_mean_target_px":float(absolute[anchor].mean()) if anchor.any() else None,
            "anchor_mean_lead_px":float(np.linalg.norm(delta[mask][anchor]*(wh-1),axis=1).mean()) if anchor.any() else None}


def train_forecast(session_path, model_path, output=None, *, epochs=80, publish=False, progress=print, cancelled=lambda:False):
    from .training_runtime import configure_training_threads
    training_threads = configure_training_threads()
    torch.manual_seed(9409)
    session_path,model_path = Path(session_path),Path(model_path)
    output = Path(output) if output else session_path/"forecast-runs"/datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S-%fZ")
    output.mkdir(parents=True,exist_ok=False)
    progress("video_forecast_replay")
    from .paths import RESOURCE_ROOT
    for name in ("video_forecast.py","video_forecast_network.py","video_forecast_training.py","video_dataset.py","one_euro.py"):
        source = Path(__file__).with_name(name)
        if not source.is_file():
            source = RESOURCE_ROOT/"provenance"/"opengazelink_pc"/name
        if source.is_file():
            shutil.copy2(source,output/name)
    values,rows,meta,cfg = replay(session_path,model_path,output,progress,cancelled)
    data = {key:[] for key in ("features","horizon","activity","delta","split","current","target","anchor","indices")}
    for horizon in HORIZONS:
        for i,a,b,fraction in future_pairs(rows,values["segment"],horizon):
            current = values["stable"][i]
            future = values["stable"][a]*(1-fraction)+values["stable"][b]*fraction
            # Exclude impossible/off-screen projection glitches as teacher labels.
            if np.any(np.abs(future-.5)>.75) or np.any(np.abs(current-.5)>.75) or np.linalg.norm(future-current)>.2:
                continue
            for key,value in {"features":values["features"][i],"horizon":[horizon/100.],"activity":[values["activity"][i]],
                              "delta":future-current,"split":values["split"][i],"current":current,"target":rows[b]["target"],
                              "anchor":rows[a]["weight"]>.5 and rows[b]["weight"]>.5,"indices":[i,a,b]}.items():
                data[key].append(value)
    data = {key:np.array(value) for key,value in data.items()}
    np.savez_compressed(output/"examples.npz",**data)
    train,selection,test = (data["split"]==s for s in range(3))
    if train.sum()<100 or selection.sum()<100 or test.sum()<100:
        raise ValueError("forecast requires train, selection and untouched test trajectories")
    mean = data["features"][train].mean(0)
    scale = np.maximum(data["features"][train].std(0),.02)
    model = ForecastNetwork(mean,scale)
    optimizer = torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
    x,h,activity,y = (torch.tensor(data[k],dtype=torch.float32) for k in ("features","horizon","activity","delta"))
    wh = np.array([cfg["screen_width"],cfg["screen_height"]],np.float32)
    zeros = np.zeros_like(data["delta"])
    baseline = metrics(zeros,data,selection,wh)
    best,best_epoch,best_score = copy.deepcopy(model.state_dict()),0,baseline["mean_future_proxy_px"]
    records = []
    train_ids = np.flatnonzero(train)
    rng = np.random.default_rng(9409)
    progress("video_forecast_training")
    for epoch in range(epochs):
        if cancelled():
            raise RuntimeError("forecast training cancelled; history retained")
        model.train()
        rng.shuffle(train_ids)
        for ids in np.array_split(train_ids,max(1,int(np.ceil(len(train_ids)/256)))):
            prediction = model(x[ids],h[ids])*activity[ids]
            loss = F.smooth_l1_loss(prediction,y[ids],beta=.005)+.015*prediction.square().mean()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            optimizer.step()
        model.eval()
        with torch.inference_mode():
            prediction = np.zeros_like(data["delta"])
            prediction[selection]=(model(x[selection],h[selection])*activity[selection]).numpy()
        score = metrics(prediction,data,selection,wh)
        record = dict(epoch=epoch+1,**score)
        records.append(record)
        torch.save({"model":model.state_dict(),"optimizer":optimizer.state_dict(),"epoch":epoch+1,"metrics":score},output/f"epoch-{epoch+1:03d}.pt")
        anchor_safe = score["anchor_mean_target_px"] is None or score["anchor_mean_target_px"]<=baseline["anchor_mean_target_px"]+5.
        if score["mean_future_proxy_px"]<best_score and score["p95_future_proxy_px"]<=baseline["p95_future_proxy_px"]*1.05 and anchor_safe:
            best,best_epoch,best_score = copy.deepcopy(model.state_dict()),epoch+1,score["mean_future_proxy_px"]
        if (epoch+1)%20==0:
            progress(f"forecast_epoch_{epoch+1}_of_{epochs}")
    model.load_state_dict(best)
    model.eval()
    with torch.inference_mode():
        prediction = (model(x,h)*activity).numpy()
    report = {"schema":SCHEMA,"epochs":epochs,"selected_epoch":best_epoch,"history":records,
              "training_samples":int(train.sum()),"horizons_ms":HORIZONS,"seed":9409,
              "selection":{"hold_current":baseline,"forecast":metrics(prediction,data,selection,wh)},
              "test":{"hold_current":metrics(zeros,data,test,wh),"forecast":metrics(prediction,data,test,wh)},
              "protocol":"One frozen current-gaze model. Whole selection/test blocks separated. No history or future-label windows cross split/gap boundaries. Training-only normalization; epochs selected on selection proxy mean with tail and weak-anchor checks; test evaluated once.",
              "label_notice":"Future filtered outputs of the frozen same-session gaze model are pseudo-labels, not eye-tracker ground truth. Weak target anchors diagnose drift. No claim of unseen-head-motion or saccade-destination prediction.",
              "base_sha256":meta["variants"]["conditioned_video"]["module_sha256"],"published":False,
              "training_cpu_threads":training_threads}
    # Test is a one-time acceptance check; never tune epochs/hyperparameters to it.
    chosen = report["selection"]["forecast"]
    held,held_base = report["test"]["forecast"],report["test"]["hold_current"]
    report["accepted"] = bool(best_epoch and chosen["mean_future_proxy_px"]<baseline["mean_future_proxy_px"]*.97
                              and held["mean_future_proxy_px"]<held_base["mean_future_proxy_px"]
                              and held["p95_future_proxy_px"]<=held_base["p95_future_proxy_px"]*1.05
                              and (held["anchor_mean_target_px"] is None or held["anchor_mean_target_px"]<=held_base["anchor_mean_target_px"]+5.))
    module_name = f"conditioned-video-forecast-{output.name}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S-%fZ')}.pt"
    scripted = torch.jit.trace(model,(x[:1],h[:1]),strict=False)
    with torch.inference_mode():
        assert torch.allclose(scripted(x[:32],h[:32]),model(x[:32],h[:32]),atol=1e-7,rtol=1e-5)
    scripted.save(str(output/module_name))
    forecast_meta = {"schema":SCHEMA,"accepted":report["accepted"],"base_sha256":report["base_sha256"],
                     "module_file":module_name,"module_sha256":digest(output/module_name),"max_horizon_ms":100.,
                     "filter_config":{key:cfg[key] for key in ("one_euro_enabled","one_euro_min_cutoff","one_euro_beta","one_euro_derivative_cutoff")},
                     "training_directory":str(output.resolve()),"label_source":"future_frozen_model_observation"}
    write_json(output/"conditioned-video-forecast.json",forecast_meta)
    np.savez_compressed(output/"predictions.npz",prediction=prediction,baseline=zeros)
    if publish and report["accepted"] and not cancelled():
        current_meta = json.loads(model_path.read_text(encoding="utf-8"))
        if current_meta["variants"]["conditioned_video"]["module_sha256"]!=report["base_sha256"]:
            raise ValueError("current VIDEO model changed while training forecast")
        target = model_path.with_name("conditioned-video-forecast.json")
        if target.exists():
            shutil.copy2(target,output/"previous-forecast.json")
        shutil.copy2(output/module_name,model_path.with_name(module_name))
        write_json(target,forecast_meta)
        report["published"] = True
    write_json(output/"report.json",report)
    return report
