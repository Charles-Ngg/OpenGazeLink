"""Versioned motion examples; future observations are used only for offline labels."""
import numpy as np
from .prediction import MotionHistory, HORIZONS
from .video_forecast_training import future_pairs

DATA_SCHEMA="opengazelink-prediction-examples-v2"
LABEL_POLICY={"source":"frozen_current_adapter_without_temporal_memory",
    "reconstruction":"centered robust local linear fit; per-frame and 35ms cumulative-displacement fast boundaries",
    "phase":"stimulus-conditioned-ms-v1; delayed stimulus prior and observed coherent saccades",
    "landing":"first low-speed run following an observed fast transition; ambiguous events masked",
    "clock":"source time; display alignment uses fresh roundtrip probes with minimum-transit fallback",
    "interpolation":"fast-edge interior interpolation is uncertain; training weight reduced, evaluation labels unchanged"}


def interpolation_confidence(left, right, fraction, noise):
    """Do not treat an unobserved fast transition's middle as precise gaze."""
    fast = np.linalg.norm(np.asarray(right)-left)>max(.035,noise*5)
    return .25 if fast and .1 < fraction < .9 else 1.


def event_timing(phase, times, segments):
    """Offline event context for evaluation, never an inference feature."""
    age = np.full(len(phase), -1., dtype=np.float32)
    next_event = np.full(len(phase), np.inf, dtype=np.float32)
    onset = None
    for i in range(len(phase)):
        if i == 0 or segments[i] != segments[i-1] or phase[i] != 2:
            onset = None
        if phase[i] == 2:
            if onset is None:
                onset = times[i]
            age[i] = times[i] - onset
    following = None
    for i in range(len(phase)-1, -1, -1):
        if i == len(phase)-1 or segments[i] != segments[i+1]:
            following = None
        if following is not None:
            next_event[i] = following - times[i]
        if age[i] == 0:
            following = times[i]
    return age, next_event


def reconstruct(points,times,segments,noise,window_ms=100.):
    """Acausal denoising without the one-sided filter's phase lag.

    Fast edges divide reconstruction neighborhoods; no fit borrows a future
    fixation before an observed jump. Isolated reversals are not erased.
    """
    result=np.asarray(points).copy()
    edge=np.zeros(len(points),bool)
    for i in range(1,len(points)):
        edge[i]=(segments[i]!=segments[i-1] or times[i]<=times[i-1]
                 or np.linalg.norm(points[i]-points[i-1])>max(.035,noise*5))
    # At 120 Hz a real fast displacement can consist entirely of increments
    # below the single-frame edge threshold. Preserve that burst instead of
    # spreading it over the +/-100 ms teacher window (including before onset).
    # These are offline label boundaries, never online inference features.
    boundary = max(.035,noise*5)
    for i in range(1,len(points)):
        for j in range(i-1,-1,-1):
            dt = times[i]-times[j]
            if segments[j]!=segments[i] or dt<=0 or dt>35:
                break
            if np.linalg.norm(points[i]-points[j])>boundary:
                edge[j+1:i+1]=True
                break
    groups=np.cumsum(edge)
    starts=np.r_[0,np.flatnonzero(np.diff(groups))+1]
    ends=np.r_[starts[1:],len(points)]
    bounds={int(groups[start]):(int(start),int(end)) for start,end in zip(starts,ends)}
    for i in range(len(points)):
        # Select by source time, not frame count.  Seven samples span about
        # 200 ms at 30 FPS but only 50 ms at 120 FPS; the old frame-count
        # window therefore stopped removing high-rate observation noise.
        start,end=bounds[int(groups[i])]
        # Source clocks may restart between segments. Global searchsorted is
        # undefined on that non-monotonic concatenation.
        lo=start+int(np.searchsorted(times[start:end],times[i]-window_ms,side="left"))
        hi=start+int(np.searchsorted(times[start:end],times[i]+window_ms,side="right"))
        ids=np.arange(lo,hi)
        ids=ids[(segments[ids]==segments[i]) & (groups[ids]==groups[i])]
        if len(ids)<3:
            continue
        age=(times[ids]-times[i])/1000
        design=np.c_[np.ones(len(ids)),age]
        fit=np.linalg.lstsq(design,points[ids],rcond=None)[0]
        residual=np.linalg.norm(points[ids]-design@fit,axis=1)
        weights=np.minimum(1,max(noise,.0015)*2/np.maximum(residual,1e-8))
        fit=np.linalg.lstsq(design*weights[:,None],points[ids]*weights[:,None],rcond=None)[0]
        result[i]=fit[0]
    return result


def make_examples(values,rows,horizons=HORIZONS):
    raw,stable=values["raw"],values["stable"]
    spatial=values["instantaneous"]
    if spatial.shape != raw.shape:
        raise ValueError("prediction replay requires instantaneous current-estimator outputs")
    times=np.array([r["source_ms"] for r in rows])
    splits=np.array([{"train":0,"validation":1,"test":2}[r["split"]] for r in rows])
    segments=np.zeros(len(rows),np.int64)
    for i in range(1,len(rows)):
        segments[i]=segments[i-1]+int(values["segment"][i]!=values["segment"][i-1] or rows[i]["trial_id"]!=rows[i-1]["trial_id"]
            or rows[i].get("capture_segment",0)!=rows[i-1].get("capture_segment",0)
            or not rows[i].get("stimulus_supported",False) or not rows[i-1].get("stimulus_supported",False))
    train_anchor=np.array([r["weight"]>.5 for r in rows]) & (splits==0)
    eligible=train_anchor[1:] & train_anchor[:-1] & (segments[1:]==segments[:-1])
    diffs=np.linalg.norm(np.diff(spatial,axis=0),axis=1)
    noise=float(np.median(diffs[eligible])/1.177) if eligible.sum()>=10 else .008
    noise=float(np.clip(noise,.001,.03))
    teacher=reconstruct(spatial,times,segments,noise)
    from .motion_labels import label_motion
    labels=label_motion(teacher,rows,noise)
    phase,phase_conf=labels["phase"],labels["phase_conf"]
    event_age, next_event = event_timing(phase, times, segments)
    landing,remaining=labels["landing"],labels["remaining"]
    landing_valid,speed=labels["landing_valid"],labels["speed"]
    history=MotionHistory()
    sequences,states,counts=[],[],[]
    for i,row in enumerate(rows):
        seq,state,count=history.update(times[i],raw[i],stable[i],values["eye_features"][i],reset=i==0 or segments[i]!=segments[i-1])
        sequences.append(seq);states.append(state);counts.append(count)
    result={k:[] for k in ("frame","horizon","target","delta","split","phase","phase_conf","landing","remaining","landing_valid","anchor","stimulus_target","future_frame","quality","interpolation_confidence","trial")}
    for horizon in horizons:
        pairs=[(i,i,i,0.) for i in range(len(rows))] if horizon==0 else future_pairs(rows,segments,horizon)
        for i,a,b,f in pairs:
            target=teacher[a]*(1-f)+teacher[b]*f
            if counts[i]<4 or np.any(np.abs(target-.5)>.8) or np.any(np.abs(raw[i]-.5)>.8) or np.linalg.norm(target-raw[i])>.8:
                continue
            # No display telemetry is needed as an inference feature. Missing
            # telemetry is excluded here because trial/split membership is uncertain.
            if not all(rows[k].get("stimulus_supported",False) for k in (i,a,b)):
                continue
            quality=float(np.clip(noise*3/max(noise*3,np.linalg.norm(spatial[b]-teacher[b])),.15,1))
            for key,value in dict(frame=i,horizon=[horizon/1000],target=target,delta=target-raw[i],split=splits[i],
                phase=phase[i],phase_conf=phase_conf[i],landing=landing[i]-raw[i],remaining=[remaining[i]],
                landing_valid=landing_valid[i],anchor=rows[a]["weight"]>.5 and rows[b]["weight"]>.5,
                stimulus_target=rows[b]["target"],future_frame=b,quality=quality,
                interpolation_confidence=interpolation_confidence(teacher[a],teacher[b],f,noise),trial=rows[i]["trial_id"]).items():
                result[key].append(value)
    if not result["frame"]:
        raise ValueError("no supported continuous prediction examples")
    result={key:np.asarray(value) for key,value in result.items()}
    frames={"sequences":np.asarray(sequences,np.float32),"states":np.asarray(states,np.float32),
        "teacher":teacher,"raw":raw,"stable":stable,"segments":segments,"times":times,"phase":phase,
        "speed":speed,"landing_valid":landing_valid,"event_age_ms":event_age,"next_event_ms":next_event}
    audit={"schema":DATA_SCHEMA,"label_policy":LABEL_POLICY,"noise_normalized":noise,"event_labels":labels["audit"],"landing_frames":int(landing_valid.sum()),"frames":len(rows),
        "trials":len(set(r["trial_id"] for r in rows)),"examples":len(result["frame"])}
    return frames,result,audit
