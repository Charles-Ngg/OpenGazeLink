"""Causal prediction inputs and runtime. No stimulus fields enter this module."""
from collections import deque
import numpy as np

SCHEMA = "opengazelink-motion-prediction-v2"
WINDOW = 12
INPUT_DIM = 411
HORIZONS = (0., 16., 33., 50., 67., 85., 100., 125., 150.)


class MotionHistory:
    def __init__(self):
        self.samples = deque(maxlen=WINDOW)

    def reset(self):
        self.samples.clear()

    def update(self, timestamp, raw, stable, feature, reset=False):
        raw, stable, feature = map(lambda x:np.asarray(x, dtype=np.float32).reshape(-1), (raw,stable,feature))
        if raw.shape != (2,) or stable.shape != (2,) or feature.shape != (404,) or not np.isfinite(np.r_[timestamp,raw,stable,feature]).all():
            self.reset()
            raise ValueError("invalid prediction inputs")
        if self.samples:
            dt=timestamp-self.samples[-1][0]
            reset=reset or not 5 <= dt <= 150 or np.linalg.norm(raw-self.samples[-1][1])>.65
        if reset:
            self.reset()
        self.samples.append((float(timestamp),raw.copy(),feature.copy()))
        times,points,features=map(np.asarray,zip(*self.samples))
        n=len(times)
        sequence=np.zeros((WINDOW,INPUT_DIM),np.float32)
        velocity=np.zeros((n,2))
        if n>1:
            velocity[1:]=np.diff(points,axis=0)/np.diff(times)[:,None]*1000
            velocity[0]=velocity[1]
        sequence[-n:,:404]=(features-features[-1])
        sequence[-n:,404:406]=points-points[-1]
        sequence[-n:,406:408]=np.clip(velocity,-15,15)
        sequence[-n:,408]=np.r_[0,np.diff(times)]/100
        sequence[-n:,409]=1
        sequence[-n:,410]=(times-times[-1])/1000
        # A robust recent linear fit gives a nonzero constant-velocity baseline;
        # no sign test or absolute motion threshold can shut off slow pursuit.
        count=min(n,5)
        age=(times[-count:]-times[-1])/1000
        if count>=2:
            weights=np.exp(age/.07)
            design=np.c_[np.ones(count),age]
            fit=np.linalg.lstsq(design*weights[:,None],points[-count:]*weights[:,None],rcond=None)[0]
            v=np.clip(fit[1],-10,10)
        else:
            v=np.zeros(2)
        acceleration=np.zeros(2)
        if n>=4:
            acceleration=np.clip((velocity[-1]-velocity[-3])/max(.01,(times[-1]-times[-3])/1000),-30,30)
        state=np.r_[v,acceleration,stable-raw].astype(np.float32)
        return sequence,state,n


class MotionPrediction:
    def __init__(self,module,metadata):
        self.module,self.metadata=module,metadata
        self.history=MotionHistory()

    def reset(self):
        self.history.reset()

    def update(self,raw,stable,timestamp,feature,hidden,screen_size,horizon_ms,max_lead_fraction=.12,reset=False):
        import torch
        wh=np.maximum(1,np.asarray(screen_size)-1)
        sequence,state,n=self.history.update(timestamp,np.asarray(raw)/wh,np.asarray(stable)/wh,feature,reset)
        horizon=float(np.clip(horizon_ms,0,150))
        if n<4 or horizon==0:
            return tuple(stable),{"mode":"prediction_warmup" if n<4 else "disabled","horizon_ms":horizon,"lead_px":[0.,0.],"lead_distance_px":0.}
        with torch.inference_mode():
            delta,log_scale,phase,landing,remaining=self.module(torch.from_numpy(sequence[None]),torch.from_numpy(state[None]),torch.tensor([[horizon/1000]],dtype=torch.float32))
        displacement=delta[0].numpy()
        if not np.isfinite(displacement).all():
            self.reset()
            raise ValueError("prediction output is non-finite")
        # Delta is measured from the unfiltered observation; this also permits
        # learning current-state correction instead of predicting a delayed filter.
        probabilities=phase.softmax(-1)[0].numpy()
        motion_gate=float(np.clip(((1-probabilities[0])-.2)/.6,0,1))
        confidence=float(np.clip(.04/max(1e-6,float(log_scale[0].exp().mean())),0,1))*motion_gate
        lead=confidence*(np.asarray(raw)+displacement*wh-np.asarray(stable))
        limit=max(0,min(.3,max_lead_fraction))*np.linalg.norm(wh)
        lead*=min(1,limit/max(1e-9,np.linalg.norm(lead)))
        mode=("prediction_fixation","prediction_pursuit","prediction_saccade")[int(probabilities.argmax())]
        return tuple(np.asarray(stable)+lead),{"mode":mode,"horizon_ms":horizon,"requested_horizon_ms":float(horizon_ms),
            "lead_px":lead.tolist(),"lead_distance_px":float(np.linalg.norm(lead)),"phase_probabilities":probabilities.tolist(),
            "uncertainty_normalized":log_scale[0].exp().numpy().tolist(),"landing_delta":landing[0].numpy().tolist(),
            "confidence_gain":confidence,
            "motion_gate":motion_gate,
            "remaining_ms":float(remaining[0,0])*1000,"sample_count":n,"target_source_ms":timestamp+horizon,
            "label_source":"offline_reconstructed_gaze_proxy"}
