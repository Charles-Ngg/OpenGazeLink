"""Offline stimulus-conditioned labels. Telemetry must never enter inference."""
import numpy as np

POLICY = {"version": "stimulus-conditioned-ms-v3", "states": ["fixation", "pursuit", "saccade"],
          "lag": "train-only robust trajectory alignment; uncertain fits do not force a delay",
          "events": "observed coherent bursts override delayed stimulus state",
          "units": "milliseconds and normalized-screen distance; no frame-count thresholds",
          "truth": "weak labels; stimulus is a prior, not measured eye position"}


def groups_for(rows):
    groups = np.zeros(len(rows), np.int64)
    for i in range(1, len(rows)):
        a, b = rows[i-1], rows[i]
        broken = (b.get('reset', False) or b.get('index', i) != a.get('index', i-1)+1
                  or b.get('split') != a.get('split') or b.get('trial_id') != a.get('trial_id')
                  or b.get('capture_segment', 0) != a.get('capture_segment', 0)
                  or not 0 < b['source_ms']-a['source_ms'] <= 100
                  or not b.get('stimulus_supported', True) or not a.get('stimulus_supported', True))
        groups[i] = groups[i-1]+int(broken)
    return groups


def interp(times, points, query):
    return np.stack([np.interp(query, times, points[:, k]) for k in range(2)], axis=-1)


def estimate_lag(points, rows, groups):
    """Estimate effective alignment, including response and residual clock lag.

    Only training trajectories participate. This is not a physiological latency
    measurement. Flat/inconsistent objectives return an explicitly uncertain fit.
    """
    records = []
    for group in np.unique(groups):
        ids = np.flatnonzero(groups == group)
        if rows[ids[0]].get('split') != 'train' or len(ids) < 3:
            continue
        t = np.array([rows[i]['source_ms'] for i in ids])
        target = np.array([rows[i].get('target', points[i]) for i in ids])
        pursuit = np.array([rows[i].get('phase') == 'pursuit' for i in ids])
        # Uniform time sampling prevents higher-rate captures dominating the fit.
        q = np.arange(t[0]+350, t[-1]-100, 40.)
        if not len(q):
            continue
        eligible = np.interp(q, t, pursuit.astype(float)) > .99
        q = q[eligible]
        if len(q):
            records.append((t, target, q, interp(t, points[ids], q+40)-interp(t, points[ids], q-40)))
    costs, counts = [], []
    for lag in range(0, 251, 10):
        errors = []
        for t, target, q, observed in records:
            expected = interp(t, target, q-lag+40)-interp(t, target, q-lag-40)
            a, b = np.linalg.norm(expected, axis=1), np.linalg.norm(observed, axis=1)
            good = (a > .004) & (a < .08) & (b > .002) & (b < .12)
            if good.any():
                cosine = (expected[good]*observed[good]).sum(1)/(a[good]*b[good])
                errors.extend((1-np.clip(cosine, -1, 1)).tolist())
        costs.append(float(np.mean(np.minimum(errors, 1.))) if len(errors) else 1.)
        counts.append(len(errors))
    best = int(np.argmin(costs))
    contrast = float(np.median(costs)-costs[best])
    reliable = counts[best] >= 100 and contrast >= .025 and costs[best] < .4 and best not in (0,25)
    return {"lag_ms": float(best*10) if reliable else 0., "candidate_lag_ms":best*10,
            "reliable":bool(reliable), "contrast":contrast, "costs":costs,
            "train_windows":counts[best], "source":"train_only"}


def label_motion(points, rows, noise, *, lag_model=None):
    """Return phase/confidence and event endpoints with split-safe offline context."""
    points = np.asarray(points, np.float64)
    if points.shape != (len(rows),2) or not np.isfinite(points).all():
        raise ValueError('motion labels require finite, aligned 2D observations')
    groups = groups_for(rows)
    lag_model = estimate_lag(points, rows, groups) if lag_model is None else lag_model
    n = len(rows)
    phase = np.zeros(n, np.int64); confidence = np.zeros(n, np.float32)
    landing = points.copy(); remaining = np.zeros(n, np.float32)
    valid = np.zeros(n, bool); speed = np.zeros(n); events = []
    for group in np.unique(groups):
        ids = np.flatnonzero(groups == group)
        if len(ids)<3:
            continue
        t = np.array([rows[i]['source_ms'] for i in ids]); p = points[ids]
        target = np.array([rows[i].get('target', points[i]) for i in ids])
        dt = float(np.median(np.diff(t)))
        left, right = np.maximum(t[0], t-20), np.minimum(t[-1], t+20)
        velocity = (interp(t,p,right)-interp(t,p,left))/np.maximum(.001,(right-left)/1000)[:,None]
        v = np.linalg.norm(velocity,axis=1); speed[ids]=v
        lag = lag_model['lag_ms']
        delayed = np.clip(np.searchsorted(t,t-lag,side='right')-1,0,len(t)-1)
        local_phase = np.array([1 if rows[ids[j]].get('phase')=='pursuit' else 0 for j in delayed])
        conf = np.full(len(t), .8, np.float32)
        supported = np.array([rows[i].get('stimulus_supported',True) and rows[i].get('phase') in ('anchor','jump','pursuit') for i in ids])
        conf[~supported] = 0.
        # Trial entry may contain an unseen departure from the preceding target.
        conf[t-t[0]<350] = np.minimum(conf[t-t[0]<350], .2)
        changes = np.r_[True, np.diff(local_phase)!=0]
        target_steps = np.r_[False,np.linalg.norm(np.diff(target,axis=0),axis=1)>.025]
        # Telemetry jump marks a candidate response interval, never the eye onset.
        jumps = target_steps & np.array([rows[i].get('phase') in ('anchor','jump') for i in ids])
        for onset in t[changes | jumps]:
            uncertain = (t>=onset-40) & (t<=onset+(120 if lag_model['reliable'] else 350))
            conf[uncertain] = np.minimum(conf[uncertain], .2)
        expected_v = (interp(t,target,np.minimum(t[-1],t-lag+40))-interp(t,target,np.maximum(t[0],t-lag-40)))/.08
        expected_speed = np.linalg.norm(expected_v,axis=1)
        # Constant-duration velocity evidence; slow pursuit is never defined as
        # fixation merely because its single-frame displacement is small.
        threshold = max(.55, float(noise)*5/.04)
        candidates = np.flatnonzero(v>threshold)
        claimed = np.zeros(len(t),bool)
        for peak in candidates[np.argsort(v[candidates])[::-1]]:
            if claimed[peak] or not supported[peak]:
                continue
            nearby = (abs(t-t[peak])>=70)&(abs(t-t[peak])<=150)
            background = float(np.median(v[nearby])) if nearby.any() else 0.
            anticipated = max(background, expected_speed[peak] if local_phase[peak]==1 else 0.)
            transition_context = (t[peak]-t[0]<=600 or any(-80<=t[peak]-onset<=600 for onset in t[jumps]))
            # A large transition may occupy the entire background window. Using
            # its own speed as the baseline would erase the main relocation and
            # retain only small subsequent peaks. Telemetry permits a wider
            # observed transition here; confidence still requires task agreement.
            if transition_context:
                anticipated=0.
            if v[peak] < max(threshold,anticipated*1.8):
                continue
            lower = max(threshold*.35, anticipated*1.25, v[peak]*.2)
            a=b=int(peak)
            max_duration=300 if transition_context else 180
            while a>0 and v[a-1]>lower and t[peak]-t[a-1]<=max_duration: a-=1
            while b+1<len(t) and v[b+1]>lower and t[b+1]-t[peak]<=max_duration: b+=1
            a=max(0,a-1); b=min(len(t)-1,b+1)
            if a==0 or b==len(t)-1 or not 10<=t[b]-t[a]<=max_duration or claimed[a:b+1].any():
                continue
            if not supported[a:b+1].all():
                continue
            before=(t>=t[a]-60)&(t<t[a]); after=(t>t[b])&(t<=t[b]+60)
            if not before.any() or not after.any() or t[-1]<t[b]+40:
                continue
            start=np.median(p[before],axis=0); endpoint=np.median(p[after],axis=0)
            amplitude=np.linalg.norm(endpoint-start)
            path=np.linalg.norm(np.diff(p[a:b+1],axis=0),axis=1).sum()
            if amplitude<max(.025,noise*4) or np.linalg.norm(p[b]-p[a])/max(path,1e-8)<.65:
                conf[a:b+1]=np.minimum(conf[a:b+1],.15)
                continue
            # A following pursuit has no stationary landing; fit its position at
            # event end instead of assigning a later moving sample as the endpoint.
            post_t=(t[after]-t[b])/1000
            endpoint=np.linalg.lstsq(np.c_[np.ones(after.sum()),post_t],p[after],rcond=None)[0][0] if after.sum()>=2 else endpoint
            event_ids=ids[a:b+1]
            # A coherent model burst alone does not establish a physiological
            # saccade. Require agreement with the known task before giving it
            # enough confidence for event-balanced sampling.
            event_confidence=.3
            cue='unprompted_burst'
            for change in np.flatnonzero(jumps):
                if -80<=t[peak]-t[change]<=600:
                    direction=target[change]-target[max(0,change-1)]
                    observed=endpoint-start
                    agreement=np.dot(direction,observed)/max(1e-8,np.linalg.norm(direction)*np.linalg.norm(observed))
                    if agreement>.5:
                        event_confidence=.9; cue='target_step_agreement'
            if t[peak]-t[0]<=600 and event_confidence<.65:
                event_confidence=.65; cue='trial_entry'
            if local_phase[peak]==1 and event_confidence<.65:
                direction=expected_v[peak]; observed=endpoint-start
                agreement=np.dot(direction,observed)/max(1e-8,np.linalg.norm(direction)*np.linalg.norm(observed))
                if agreement>.5:
                    event_confidence=.65; cue='pursuit_catchup_agreement'
            local_phase[a:b+1]=2; conf[a:b+1]=min(event_confidence, 30/max(30,dt))
            landing[event_ids]=endpoint; remaining[event_ids]=np.maximum(0,(t[b]-t[a:b+1])/1000)
            valid[event_ids]=conf[a:b+1]>=.5; claimed[a:b+1]=True
            events.append({"start":int(ids[a]),"end":int(ids[b]),"duration_ms":float(t[b]-t[a]),
                           "amplitude":float(amplitude),"split":rows[ids[a]].get('split'),
                           "trial":rows[ids[a]].get('trial_id'),"confidence":float(conf[a]),"cue":cue})
        # Contradictory fast observations are uncertain, not confident fixation.
        conf[(local_phase==0)&(v>threshold)&~claimed]=.1
        phase[ids]=local_phase; confidence[ids]=conf
    return {"phase":phase,"phase_conf":confidence,"landing":landing,"remaining":remaining,
            "landing_valid":valid,"speed":speed,"groups":groups,
            "audit":{"policy":POLICY,"lag_model":lag_model,"events":events,
                     "phase_counts":np.bincount(phase,minlength=3).tolist(),
                     "confident_phase_counts":np.bincount(phase[confidence>=.5],minlength=3).tolist(),
                     "uncertain_frames":int((confidence<.5).sum())}}
