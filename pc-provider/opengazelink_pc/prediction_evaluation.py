"""Dynamic diagnostics kept separate from the epoch-selection objective."""
import numpy as np


def diagnostics(prediction,data,frames,rows,aux,split,wh):
    mask=(data["split"]==split)&(data["horizon"][:,0]>0)
    pixels=wh-1
    result={"reference":"offline reconstructed gaze proxy; not eye-tracker ground truth","by_horizon_ms":{},"by_stimulus_profile":{}}
    error=np.linalg.norm((prediction-data["delta"])*pixels,axis=1)
    for h in np.unique(data["horizon"][mask,0]):
        ids=mask & np.isclose(data["horizon"][:,0],h)
        result["by_horizon_ms"][str(round(h*1000))]={"samples":int(ids.sum()),"mean_px":float(error[ids].mean()),"p95_px":float(np.percentile(error[ids],95))}
    profiles=np.array([rows[i].get("motion_profile","legacy") for i in data["frame"]])
    for name in np.unique(profiles[mask]):
        ids=mask & (profiles==name)
        result["by_stimulus_profile"][str(name)]={"samples":int(ids.sum()),"mean_px":float(error[ids].mean())}
    # Compare phase at 85 ms only, once per source frame, rather than treating
    # correlated horizon replicas as independent eye movements.
    ids=aux["ids"]
    selected=(data["split"][ids]==split)&np.isclose(data["horizon"][ids,0],.085)
    observed=data["phase"][ids[selected]]
    estimated=aux["phase"][selected].argmax(1)
    matrix=np.zeros((3,3),dtype=int)
    for a,b in zip(observed,estimated):
        matrix[a,b]+=1
    result["phase_proxy_confusion"]=matrix.tolist()
    landing_mask=selected & data["landing_valid"][ids]
    if landing_mask.any():
        end_error=np.linalg.norm((aux["landing"][landing_mask]-data["landing"][ids[landing_mask]])*pixels,axis=1)
        result["landing_proxy"]={"source_frames":int(landing_mask.sum()),"mean_px":float(end_error.mean()),
            "p95_px":float(np.percentile(end_error,95)),"remaining_time_mae_ms":float(np.abs(aux["remaining"][landing_mask]-data["remaining"][ids[landing_mask]]).mean()*1000)}
    lag=[]
    selected_ids=ids[selected]
    for trial in np.unique(data["trial"][selected_ids]):
        positions=selected_ids[data["trial"][selected_ids]==trial]
        if len(positions)<12:
            continue
        fi=data["frame"][positions]
        order=np.argsort(frames["times"][fi]);positions=positions[order];fi=fi[order]
        time=frames["times"][fi]
        valid_frame_ids=np.flatnonzero(frames["segments"]==frames["segments"][fi[0]])
        reference_time=frames["times"][valid_frame_ids]
        reference=frames["teacher"][valid_frame_ids]
        output=frames["raw"][fi]+prediction[positions]
        # Keep the same interior frames for every tested temporal shift.
        interior=(time+85-150>=reference_time[0])&(time+85+150<=reference_time[-1])
        if interior.sum()<10 or np.linalg.norm(np.std(reference,axis=0))<.02:
            continue
        def fit_lag(points):
            costs=[]
            for shift in range(-150,151,5):
                target=np.stack([np.interp(time[interior]+85-shift,reference_time,reference[:,axis]) for axis in (0,1)],axis=1)
                residual=points[interior]-target
                residual-=residual.mean(0)  # remove constant spatial bias from phase assessment
                costs.append(float(np.mean(np.square(residual*pixels))))
            best=int(np.argmin(costs))
            return -150+best*5
        lag.append({"trial":str(trial),"candidate_lag_ms":fit_lag(output),"hold_stable_lag_ms":fit_lag(frames["stable"][fi]),"hold_raw_lag_ms":fit_lag(frames["raw"][fi])})
    result["relative_proxy_lag"]=lag
    result["lag_notice"]="Positive means behind the reconstructed trajectory at source time + 85 ms. Boundary optima are censored. This is not end-to-end display latency."
    return result
