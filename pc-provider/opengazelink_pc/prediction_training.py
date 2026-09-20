"""Train and audit an independent dynamic predictor; current gaze weights stay frozen."""
import copy
from datetime import datetime,timezone
import json
from pathlib import Path
import shutil
import numpy as np
import torch
from torch.nn import functional as F
from .prediction import SCHEMA,HORIZONS
from .prediction_dataset import make_examples
from .prediction_network import PredictionNetwork
from .video_forecast_training import replay,digest
from .video_session import write_json


def runtime_delta(delta,scale,state,horizons,raw,stable,wh,max_lead,phase=None):
    confidence=np.clip(.04/np.maximum(np.exp(scale).mean(1,keepdims=True),1e-6),0,1)
    if phase is not None:
        confidence*=np.clip(((1-phase[:,0:1])-.2)/.6,0,1)
    candidate=raw+confidence*delta+(1-confidence)*(stable-raw)
    lead=(candidate-stable)*(wh-1)
    limit=min(.3,max(0,max_lead))*np.linalg.norm(wh-1)
    lead*=np.minimum(1,limit/np.maximum(1e-9,np.linalg.norm(lead,axis=1,keepdims=True)))
    lead[horizons[:,0]==0]=0
    return stable+lead/(wh-1)-raw


def scores(prediction,data,frames,mask,wh):
    ids=np.flatnonzero(mask & (data["horizon"][:,0]>0))
    if not len(ids):
        return {"samples":0}
    pixels=wh-1
    error=np.linalg.norm((prediction[ids]-data["delta"][ids])*pixels,axis=1)
    result={"samples":len(ids),"mean_proxy_px":float(error.mean()),"p95_proxy_px":float(np.percentile(error,95)),"groups":{}}
    for phase,name in enumerate(("fixation","pursuit","saccade_proxy")):
        local=data["phase"][ids]==phase
        if local.any():
            result["groups"][name]={"samples":int(local.sum()),"mean_px":float(error[local].mean()),"p95_px":float(np.percentile(error[local],95))}
    anchor=data["anchor"][ids]
    if anchor.any():
        absolute=frames["raw"][data["frame"][ids]]+prediction[ids]
        result["anchor_target_mean_px"]=float(np.linalg.norm((absolute[anchor]-data["stimulus_target"][ids][anchor])*pixels,axis=1).mean())
    result["balanced_proxy_px"]=float(np.mean([v["mean_px"] for v in result["groups"].values()]))
    return result


def passes(candidate,baseline,*,improvement=False):
    if candidate.get("samples",0)<100 or not candidate.get("groups",{}).get("pursuit",{}).get("samples",0):
        return False
    if candidate["balanced_proxy_px"]>baseline["balanced_proxy_px"]*(.98 if improvement else 1.02):
        return False
    if candidate["p95_proxy_px"]>baseline["p95_proxy_px"]*1.05+1:
        return False
    for name,value in candidate["groups"].items():
        if value["mean_px"]>baseline["groups"][name]["mean_px"]*1.10+2:
            return False
    return candidate.get("anchor_target_mean_px",0)<=baseline.get("anchor_target_mean_px",0)+4


def motion_beats_raw(candidate,raw):
    """Raw is a motion diagnostic, not the live output baseline.

    It is intentionally not required for fixation: bypassing stabilization may
    score well against an instantaneous proxy while making the cursor noisy.
    Pursuit and saccade prediction must still beat that stronger motion control.
    """
    return all(candidate["groups"][name]["mean_px"] < raw["groups"][name]["mean_px"]
               for name in ("pursuit", "saccade_proxy"))


def train_prediction(session_path,model_path,output=None,*,epochs=40,publish_model_path=None,progress=print,cancelled=lambda:False):
    from .training_runtime import configure_training_threads
    training_threads = configure_training_threads()
    torch.manual_seed(9410)
    session_path,model_path=Path(session_path),Path(model_path)
    output=Path(output) if output else session_path/"prediction-runs"/datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S-%fZ")
    output.mkdir(parents=True,exist_ok=False)
    from .paths import RESOURCE_ROOT
    sources={}
    for name in ("prediction.py","prediction_dataset.py","prediction_network.py","prediction_training.py","prediction_evaluation.py","video_forecast_training.py","video_dataset.py"):
        source=Path(__file__).with_name(name)
        if not source.is_file():
            source=RESOURCE_ROOT/"provenance"/"opengazelink_pc"/name
        if source.is_file():
            shutil.copy2(source,output/name)
            sources[name]=digest(source)
    run={"schema":SCHEMA,"session":str(session_path.resolve()),"source_sha256":sources,"seed":9410,"epochs":epochs,"state":"running"}
    write_json(output/"run.json",run)
    try:
        result=_train(session_path,model_path,output,epochs,publish_model_path,progress,cancelled)
        result["training_cpu_threads"] = training_threads
        write_json(output/"run.json",dict(run,state="complete"))
        return result
    except Exception as error:
        write_json(output/"run.json",dict(run,state="cancelled" if cancelled() else "failed",error=str(error)))
        raise


def _train(session_path,model_path,output,epochs,publish_model_path,progress,cancelled):
    if epochs<1:
        raise ValueError("epochs must be positive")
    progress("prediction_replay")
    values,rows,meta,cfg=replay(session_path,model_path,output,lambda s:progress(s.replace("forecast_replay", "prediction_replay")),cancelled,instantaneous=True)
    progress("prediction_labels")
    frames,data,audit=make_examples(values,rows)
    np.savez_compressed(output/"motion-frames.npz",**frames)
    np.savez_compressed(output/"motion-examples.npz",**data)
    write_json(output/"label-audit.json",audit)
    train,selection,test=(data["split"]==s for s in range(3))
    if any(mask.sum()<100 for mask in (train,selection,test)):
        raise ValueError("prediction needs at least 100 examples in each complete trial split")
    sets=[set(data["trial"][mask]) for mask in (train,selection,test)]
    if sets[0]&sets[1] or sets[0]&sets[2] or sets[1]&sets[2]:
        raise ValueError("trial leakage between splits")
    train_frames=np.unique(data["frame"][train])
    observed=frames["sequences"][train_frames].reshape(-1,411)
    observed=observed[observed[:,409]>0]
    mean=observed.mean(0);scale=np.maximum(observed.std(0),.01)
    model=PredictionNetwork(mean,scale)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.005)
    sequences=torch.from_numpy(frames["sequences"])
    states=torch.from_numpy(frames["states"])
    horizon=torch.tensor(data["horizon"],dtype=torch.float32)
    labels=torch.tensor(data["delta"],dtype=torch.float32)
    phase=torch.tensor(data["phase"],dtype=torch.long)
    landing=torch.tensor(data["landing"],dtype=torch.float32)
    remaining=torch.tensor(data["remaining"],dtype=torch.float32)
    quality=torch.tensor(data["quality"],dtype=torch.float32)
    phase_conf=torch.tensor(data["phase_conf"],dtype=torch.float32)
    wh=np.array([cfg["screen_width"],cfg["screen_height"]],dtype=np.float32)
    frame_ids=data["frame"]
    raw,stable=frames["raw"][frame_ids],frames["stable"][frame_ids]
    hold_stable=stable-raw
    safe_delta=torch.tensor(hold_stable,dtype=torch.float32)
    anchor=torch.tensor(data["anchor"],dtype=torch.bool)
    hold_raw=np.zeros_like(raw)
    kinematic=frames["states"][frame_ids,:2]*data["horizon"]
    max_lead=cfg.get("extrapolation_max_lead_fraction",.12)
    kinematic=runtime_delta(kinematic,np.full_like(kinematic,-7),None,data["horizon"],raw,stable,wh,max_lead)

    def evaluate(mask):
        output=np.zeros_like(raw)
        aux={key:[] for key in ("ids","phase","landing","remaining","scale")}
        model.eval()
        with torch.inference_mode():
            for ids in np.array_split(np.flatnonzero(mask),max(1,int(np.ceil(mask.sum()/256)))):
                fi=frame_ids[ids]
                pred,sigma,logits,end,time=model(sequences[fi],states[fi],horizon[ids])
                probabilities=logits.softmax(-1).numpy()
                output[ids]=runtime_delta(pred.numpy(),sigma.numpy(),None,data["horizon"][ids],raw[ids],stable[ids],wh,max_lead,probabilities)
                for key,value in dict(ids=ids,phase=probabilities,landing=end.numpy(),remaining=time.numpy(),scale=sigma.numpy()).items():
                    aux[key].append(value)
        return output,{k:np.concatenate(v) for k,v in aux.items()}

    baselines={name:scores(pred,data,frames,selection,wh) for name,pred in (("hold_stable",hold_stable),("hold_raw",hold_raw),("kinematic",kinematic))}
    best,best_epoch,best_score=copy.deepcopy(model.state_dict()),0,float("inf")
    records=[];rng=np.random.default_rng(9410)
    # Equal draw across observed motion classes so long fixations cannot dominate.
    pools=[np.flatnonzero(train & (data["phase"]==i)) for i in range(3)]
    pools=[p for p in pools if len(p)]
    progress("prediction_training")
    for epoch in range(epochs):
        if cancelled():
            raise RuntimeError("prediction training cancelled; data and checkpoints retained")
        epoch_ids=np.concatenate([rng.choice(p,min(1600,max(300,len(p))),replace=len(p)<300) for p in pools])
        rng.shuffle(epoch_ids)
        model.train();losses=[]
        for ids in np.array_split(epoch_ids,max(1,int(np.ceil(len(epoch_ids)/128)))):
            fi=frame_ids[ids]
            pred,sigma,logits,end,time=model(sequences[fi],states[fi],horizon[ids])
            # Settled target periods supervise zero lead relative to the live
            # stabilized output. Future instantaneous pseudo-labels otherwise
            # reward predicting observation noise and can create static drift.
            target=torch.where(anchor[ids,None],safe_delta[ids],labels[ids])
            distance=F.smooth_l1_loss(pred,target,beta=.01,reduction="none").mean(1)
            # Detached residual trains uncertainty without letting variance hide a bad trajectory.
            calibration=(torch.abs(pred.detach()-labels[ids])*torch.exp(-sigma)+sigma).mean(1)
            classification=F.cross_entropy(logits,phase[ids],reduction="none")*phase_conf[ids]
            loss=(distance*quality[ids]).mean()+.001*calibration.mean()+.012*classification.mean()
            land_mask=torch.from_numpy(data["landing_valid"][ids])
            if land_mask.any():
                loss=loss+.2*F.smooth_l1_loss(end[land_mask],landing[ids][land_mask],beta=.01)+.04*F.smooth_l1_loss(time[land_mask],remaining[ids][land_mask],beta=.02)
            optimizer.zero_grad();loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            optimizer.step();losses.append(float(loss.detach()))
        prediction,_=evaluate(selection)
        score=scores(prediction,data,frames,selection,wh)
        records.append(dict(epoch=epoch+1,loss=float(np.mean(losses)),**score))
        eligible=passes(score,baselines["hold_stable"],improvement=True) and motion_beats_raw(score,baselines["hold_raw"])
        if eligible and score["balanced_proxy_px"]<best_score:
            best,best_epoch,best_score=copy.deepcopy(model.state_dict()),epoch+1,score["balanced_proxy_px"]
        # Retain every epoch and optimizer for reproducibility and later experiments.
        torch.save({"model":model.state_dict(),"optimizer":optimizer.state_dict(),"epoch":epoch+1,"selection":score},output/f"epoch-{epoch+1:03d}.pt")
        if (epoch+1)%5==0:
            progress(f"prediction_training_{epoch+1}_of_{epochs}")
    if not best_epoch:
        # Export the final candidate for offline inspection, never publish it.
        best=copy.deepcopy(model.state_dict())
    model.load_state_dict(best);model.eval()
    prediction,aux=evaluate(selection | test)
    chosen=scores(prediction,data,frames,selection,wh)
    test_baselines={name:scores(pred,data,frames,test,wh) for name,pred in (("hold_stable",hold_stable),("hold_raw",hold_raw),("kinematic",kinematic))}
    held=scores(prediction,data,frames,test,wh)
    accepted=bool(best_epoch and passes(held,test_baselines["hold_stable"],improvement=True)
                  and motion_beats_raw(held,test_baselines["hold_raw"]))
    check_ids=data["frame"][:7]
    # Frozen PyInstaller modules do not expose Python source to inspect.getsource,
    # which torch.jit.script requires. The network has a fixed tensor-only
    # forward graph, so tracing is the portable export path used by the app.
    scripted=torch.jit.trace(
        model,
        (sequences[check_ids[:1]],states[check_ids[:1]],horizon[:1]),
        strict=False,
    )
    with torch.inference_mode():
        a,b=scripted(sequences[check_ids],states[check_ids],horizon[:7]),model(sequences[check_ids],states[check_ids],horizon[:7])
        for actual,expected in zip(a,b):
            torch.testing.assert_close(actual,expected)
    name="conditioned-motion-prediction-"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S-%fZ")+".pt"
    scripted.save(str(output/name))
    forecast_meta={"schema":SCHEMA,"accepted":accepted,"base_sha256":meta["variants"]["conditioned_video"]["module_sha256"],
        "module_file":name,"module_sha256":digest(output/name),"max_horizon_ms":150.,"training_directory":str(output.resolve()),
        "filter_config":{k:cfg[k] for k in ("one_euro_enabled","one_euro_min_cutoff","one_euro_beta","one_euro_derivative_cutoff")},
        "label_source":"offline_reconstructed_gaze_proxy","label_policy":audit["label_policy"],"horizons_ms":HORIZONS}
    forecast_meta["motion_gate"]={"source":"1 - fixation_probability","zero_below":.2,"full_above":.8}
    report={"schema":SCHEMA,"accepted":accepted,"published":False,"selected_epoch":best_epoch,"history":records,
        "training_examples":int(train.sum()),"selection":{"baselines":baselines,"candidate":chosen},
        "test":{"baselines":test_baselines,"candidate":held},"label_audit":audit,
        "notice":"Proxy reconstruction is not measured eye-tracker truth. Test trials are evaluated once after epoch selection. No claim of measured sensor-to-photon compensation.",
        "base_sha256":forecast_meta["base_sha256"]}
    np.savez_compressed(output/"predictions.npz",prediction=prediction,hold_stable=hold_stable,hold_raw=hold_raw,kinematic=kinematic,**aux)
    write_json(output/"conditioned-video-forecast.json",forecast_meta)
    from .prediction_evaluation import diagnostics
    write_json(output/"dynamic-diagnostics.json",{name:diagnostics(prediction,data,frames,rows,aux,index,wh) for name,index in (("selection",1),("test",2))})
    if accepted and publish_model_path is not None and not cancelled():
        destination=Path(publish_model_path)
        active=json.loads(destination.read_text(encoding="utf-8"))
        if active["variants"]["conditioned_video"]["module_sha256"]!=forecast_meta["base_sha256"]:
            raise ValueError("current gaze model changed during prediction training")
        target=destination.with_name("conditioned-video-forecast.json")
        if target.exists():
            shutil.copy2(target,output/"previous-forecast.json")
        shutil.copy2(output/name,destination.with_name(name))
        write_json(target,forecast_meta)
        report["published"]=True
    write_json(output/"report.json",report)
    return report
