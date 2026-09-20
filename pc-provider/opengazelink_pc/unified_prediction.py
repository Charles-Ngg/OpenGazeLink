"""Shared-state, joint denoising and prediction. All inputs are causal.

The VIDEO encoder's hidden state is reused instead of training a second GRU.
Local trajectory fits supply interpretable candidate positions; a small head
learns their mixture and correction at both zero and future horizons.
"""
from collections import deque
import numpy as np
import torch
from torch import nn

SCHEMA = "opengazelink-unified-prediction-v3"
HORIZONS = (0., 16., 33., 50., 67., 85., 100., 125., 150., 175., 200., 225., 250.)
CONTEXT_DIM = 64 + 12 * 7 + 3 * 8 + 3 * 4


class UnifiedHistory:
    def __init__(self, sample_interval_ms=0., intrinsic_stability=False, recent_interval_ms=0., recent_velocity_expert=False):
        self.sample_interval_ms = float(sample_interval_ms)
        self.intrinsic_stability = bool(intrinsic_stability)
        self.recent_interval_ms = float(recent_interval_ms)
        self.recent_velocity_expert = bool(recent_velocity_expert)
        self.samples = deque(maxlen=128 if self.sample_interval_ms > 0 else 12)

    def reset(self):
        self.samples.clear()

    def update(self, timestamp, raw, stable, feature, hidden, reset=False):
        raw, stable, feature, hidden = [np.asarray(a, np.float32).reshape(-1) for a in (raw, stable, feature, hidden)]
        if (raw.shape != (2,) or stable.shape != (2,) or feature.shape != (404,) or hidden.shape != (64,)
                or not np.isfinite(np.r_[timestamp, raw, stable, feature, hidden]).all()):
            self.reset()
            raise ValueError("invalid unified prediction input")
        if self.samples:
            dt = timestamp - self.samples[-1][0]
            reset = reset or not 5 <= dt <= 150 or np.linalg.norm(raw-self.samples[-1][1]) > .65
        if reset:
            self.reset()
        self.samples.append((float(timestamp), raw.copy(), feature[:8].copy()))
        times, points, eye = map(np.asarray, zip(*self.samples))
        source_times, source_points = times, points
        if self.sample_interval_ms > 0:
            # Fixed source-time coverage: 12 frames otherwise collapse to 92 ms
            # at 120 Hz. Interpolation uses only observations already received.
            grid = times[-1] - np.arange(11, -1, -1)*self.sample_interval_ms
            grid = grid[grid >= times[0]]
            points = np.stack([np.interp(grid, times, points[:, j]) for j in range(2)], 1)
            eye = np.stack([np.interp(grid, times, eye[:, j]) for j in range(8)], 1)
            times = grid
        n = len(times)
        age = (times-times[-1])/1000
        velocity = np.zeros_like(points)
        if n > 1:
            velocity[1:] = np.diff(points, axis=0)/np.diff(times)[:, None]*1000
            velocity[0] = velocity[1]
        trajectory = np.zeros((12, 7), np.float32)
        trajectory[-n:] = np.c_[points-raw, np.clip(velocity, -15, 15), age, np.r_[0, np.diff(times)]/100, np.ones(n)]
        fits = []
        for window in (3, 5, 8):
            count = min(window, n)
            if count > 1:
                a = age[-count:]
                weights = np.exp(a/.12)
                design = np.c_[np.ones(count), a]
                fit = np.linalg.lstsq(design*weights[:, None], (points[-count:]-raw)*weights[:, None], rcond=None)[0]
                fits.append(np.r_[np.clip(fit[0], -.3, .3), np.clip(fit[1], -10, 10)])
            else:
                fits.append(np.zeros(4))
        differences = np.stack([eye[-1]-eye[max(0, n-1-k)] for k in (1, 3, 6)])
        if self.recent_interval_ms > 0:
            # Preserve high-rate onset/deceleration evidence alongside the
            # long-window fits. The ABI size stays fixed, its metadata changes.
            recent_times = source_times[-1]-np.arange(11, -1, -1)*self.recent_interval_ms
            recent_times = recent_times[recent_times >= source_times[0]]
            recent = np.stack([np.interp(recent_times, source_times, source_points[:, j]) for j in range(2)], 1)
            recent_velocity = np.zeros_like(recent)
            if len(recent) > 1:
                recent_velocity[1:] = np.diff(recent, axis=0)/np.diff(recent_times)[:, None]*1000
                recent_velocity[0] = recent_velocity[1]
            trajectory[:] = 0
            trajectory[-len(recent):] = np.c_[recent-raw, np.clip(recent_velocity, -15, 15),
                (recent_times-recent_times[-1])/1000, np.r_[0, np.diff(recent_times)]/100, np.ones(len(recent))]
            if self.recent_velocity_expert:
                velocity = recent_velocity
        context = np.r_[hidden, trajectory.ravel(), differences.ravel(), np.asarray(fits).ravel()].astype(np.float32)
        # [current offset x/y, velocity x/y] for each expert.
        experts = np.zeros((6, 4), np.float32)
        experts[1, :2] = stable-raw
        if self.intrinsic_stability:
            # A stationary expert must still exist with external filtering off.
            # Otherwise this candidate duplicates raw and the remaining experts
            # are all noisy extrapolating fits, even on a fixation.
            smoothing = np.exp(age/.065)
            experts[1, :2] = ((points-raw)*smoothing[:, None]).sum(0)/smoothing.sum()
        experts[2:5] = fits
        experts[5, 2:] = np.clip(velocity[-1], -10, 10)
        return context, experts, n


class UnifiedNetwork(nn.Module):
    def __init__(self, mean, scale, use_filter=True, use_shared=True, residual_scale=.025):
        super().__init__()
        self.use_filter = use_filter
        self.use_shared = use_shared
        self.residual_scale = residual_scale
        self.register_buffer("mean", torch.as_tensor(mean, dtype=torch.float32))
        self.register_buffer("scale", torch.as_tensor(scale, dtype=torch.float32))
        self.encoder = nn.Sequential(nn.Linear(CONTEXT_DIM, 64), nn.SiLU(), nn.Linear(64, 48), nn.SiLU())
        self.head = nn.Sequential(nn.Linear(49, 48), nn.SiLU(), nn.Linear(48, 8))
        self.phase = nn.Linear(48, 3)
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, context, experts, horizon):
        x = ((context-self.mean)/self.scale).clamp(-8, 8)
        if not self.use_shared:
            x = torch.cat((torch.zeros_like(x[:, :64]), x[:, 64:]), 1)
        z = self.encoder(x)
        head = self.head(torch.cat((z, horizon/.25), 1))
        logits = head[:, :6]
        if not self.use_filter:
            logits = logits + torch.tensor([0., -10000., 0., 0., 0., 0.], device=logits.device)
        weights = logits.softmax(1)
        # Limit single-frame extrapolation duration; fitted velocity remains
        # available at the full horizon for sustained pursuit.
        duration = horizon.expand(-1, 6).clone()
        duration[:, 5] = .07*(1-torch.exp(-horizon[:, 0]/.07))
        positions = experts[:, :, :2] + experts[:, :, 2:]*duration[:, :, None]
        delta = (positions*weights[:, :, None]).sum(1)+self.residual_scale*head[:, 6:8].tanh()
        return delta, self.phase(z), weights


class EventTrajectoryNetwork(nn.Module):
    """Explicit fixation/pursuit/landing experiment with the V3 input ABI.

    Each call revises the landing estimate from the latest causal history.
    Endpoint/remaining-time supervision is offline-only; no target is an input.
    """
    def __init__(self, mean, scale):
        super().__init__()
        self.register_buffer('mean', torch.as_tensor(mean, dtype=torch.float32))
        self.register_buffer('scale', torch.as_tensor(scale, dtype=torch.float32))
        self.encoder = nn.Sequential(nn.Linear(CONTEXT_DIM-64, 64), nn.SiLU(), nn.Linear(64, 48), nn.SiLU())
        self.phase = nn.Linear(48, 3)
        self.parameters_head = nn.Linear(48, 5)
        self.horizon_gate = nn.Sequential(nn.Linear(49, 24), nn.SiLU(), nn.Linear(24, 3))
        nn.init.zeros_(self.parameters_head.weight)
        nn.init.zeros_(self.parameters_head.bias)
        nn.init.zeros_(self.horizon_gate[-1].weight)
        nn.init.zeros_(self.horizon_gate[-1].bias)

    def state(self, context, experts):
        x = ((context[:,64:]-self.mean[64:])/self.scale[64:]).clamp(-8,8)
        z = self.encoder(x)
        p = self.parameters_head(z)
        trajectory = context[:,64:148].reshape(-1,12,7)
        points, age, valid = trajectory[:,-4:,:2], trajectory[:,-4:,4], trajectory[:,-4:,6]
        mass = valid.sum(1,keepdim=True).clamp_min(1)
        average_age = (age*valid).sum(1,keepdim=True)/mass
        centered = (age-average_age)*valid
        velocity = (points*centered[:,:,None]).sum(1)/centered.square().sum(1,keepdim=True).clamp_min(1e-6)
        velocity = velocity.clamp(-10,10)
        remaining = .008+.112*p[:,:1].sigmoid()
        gain = .2+1.6*p[:,1:2].sigmoid()
        perpendicular = torch.stack((-velocity[:,1],velocity[:,0]),1)
        endpoint = (velocity*gain+perpendicular*.5*p[:,2:3].tanh())*remaining
        return z, p, velocity, endpoint, remaining

    def candidates(self, context, experts, horizon):
        z,p,velocity,endpoint,remaining = self.state(context,experts)
        # Time-limited acceleration corrects bends already visible in history.
        acceleration = ((velocity-experts[:,3,2:])/.04).clamp(-20,20)
        acceleration_time = .07*horizon-.0049*(1-torch.exp(-horizon/.07))
        pursuit = experts[:,2,:2]+experts[:,2,2:]*horizon*(.5+p[:,3:4].sigmoid())
        pursuit = pursuit+acceleration*acceleration_time*p[:,4:5].tanh()
        landing = endpoint*(1-torch.exp(-3*horizon/remaining))
        candidates = torch.stack((experts[:,1,:2], pursuit, landing),1)
        return candidates, z

    def forward(self, context, experts, horizon):
        candidates, z = self.candidates(context, experts, horizon)
        phase = self.phase(z)
        weights = (phase+self.horizon_gate(torch.cat((z,horizon/.25),1))).softmax(1)
        return (candidates*weights[:,:,None]).sum(1), phase, weights


class MotionStateNetwork(nn.Module):
    """V3 input ABI with internal causal stability and bounded motion experts.

    Uses raw (unstandardized) physical trajectory values to construct experts.
    No stimulus, future observation, or offline phase enters inference.
    """
    def __init__(self, mean, scale, dynamics=True, landing=True):
        super().__init__()
        self.dynamics, self.landing = dynamics, landing
        self.register_buffer('mean', torch.as_tensor(mean, dtype=torch.float32))
        self.register_buffer('scale', torch.as_tensor(scale, dtype=torch.float32))
        self.encoder = nn.Sequential(nn.Linear(CONTEXT_DIM-64, 64), nn.SiLU(), nn.Linear(64, 48), nn.SiLU())
        self.head = nn.Sequential(nn.Linear(49, 48), nn.SiLU(), nn.Linear(48, 10))
        self.phase = nn.Linear(48, 3)
        self.endpoint = nn.Linear(48, 2)
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)
        nn.init.zeros_(self.endpoint.weight)
        nn.init.zeros_(self.endpoint.bias)

    def state(self, context):
        x = ((context[:, 64:]-self.mean[64:])/self.scale[64:]).clamp(-8, 8)
        z = self.encoder(x)
        trajectory = context[:, 64:148].reshape(-1, 12, 7)
        age, valid = trajectory[:, :, 4], trajectory[:, :, 6]
        smoothing = torch.exp(age/.065)*valid
        stable = (trajectory[:, :, :2]*smoothing[:, :, None]).sum(1)/smoothing.sum(1, keepdim=True).clamp_min(1.)
        fits = context[:, -12:].reshape(-1, 3, 4)
        velocity = fits[:, 0, 2:]
        # Recent and longer fits have different effective observation times.
        separation = ((age[:, -3:].mean(1)-age[:, -8:].mean(1))).clamp_min(.02)
        acceleration = ((velocity-fits[:, 2, 2:])/separation[:, None]).clamp(-30, 30)
        # Landing distance/time are coupled to observed velocity, not free bias.
        remaining = .015+.165*torch.sigmoid(self.endpoint(z)[:, :1])
        gain = .25+1.5*torch.sigmoid(self.endpoint(z)[:, 1:])
        endpoint = velocity*remaining*gain
        return z, stable, velocity, acceleration, endpoint, remaining

    def forward(self, context, experts, horizon):
        z, stable, velocity, acceleration, endpoint, remaining = self.state(context)
        head = self.head(torch.cat((z, horizon/.25), 1))
        duration = horizon.expand(-1, 6).clone()
        duration[:, 5] = .07*(1-torch.exp(-horizon[:, 0]/.07))
        positions = experts[:, :, :2]+experts[:, :, 2:]*duration[:, :, None]
        # Replace optional externally filtered duplicate with intrinsic causal mean.
        positions = torch.cat((positions[:, :1], stable[:, None], positions[:, 2:]), 1)
        damped_time = .1*(1-torch.exp(-horizon/.1))
        acceleration_delta = acceleration*(.1*horizon-.01*(1-torch.exp(-horizon/.1)))
        accelerated = experts[:, 2, :2]+velocity*horizon+acceleration_delta
        braking = experts[:, 2, :2]+velocity*damped_time
        landing = endpoint*(1-torch.exp(-3*horizon/remaining))
        candidates = torch.cat((positions, accelerated[:, None], braking[:, None], landing[:, None],
                                (stable+velocity*damped_time)[:, None]), 1)
        mask = context.new_tensor([0., 0., 0., 0., 0., 0.,
            0. if self.dynamics else -10000., 0. if self.dynamics else -10000.,
            0. if self.landing else -10000., 0.])
        weights = (head+mask).softmax(1)
        return (candidates*weights[:, :, None]).sum(1), self.phase(z), weights


class RefinedMotionNetwork(nn.Module):
    """Retain a frozen validated predictor; learn evidence-based corrections."""
    def __init__(self, incumbent, mean, scale, dynamics=True, landing=True):
        super().__init__()
        self.incumbent = incumbent
        for parameter in incumbent.parameters():
            parameter.requires_grad_(False)
        self.motion = MotionStateNetwork(mean,scale,dynamics,landing)
        self.dynamics, self.landing = dynamics, landing
        self.gate = nn.Sequential(nn.Linear(49,32),nn.SiLU(),nn.Linear(32,4))
        nn.init.zeros_(self.gate[-1].weight)
        with torch.no_grad():
            self.gate[-1].bias.copy_(torch.tensor([3.,0.,0.,0.]))

    def state(self, context):
        return self.motion.state(context)

    def forward(self, context, experts, horizon):
        old,_,_ = self.incumbent(context,experts,horizon)
        z,stable,velocity,acceleration,endpoint,remaining = self.motion.state(context)
        predicted,phase,_ = self.motion(context,experts,horizon)
        landing = endpoint*(1-torch.exp(-3*horizon/remaining))
        candidates = torch.stack((old,stable,predicted,landing),1)
        mask = context.new_tensor([0.,0.,0. if self.dynamics else -10000.,0. if self.landing else -10000.])
        weights = (self.gate(torch.cat((z,horizon/.25),1))+mask).softmax(1)
        return (candidates*weights[:,:,None]).sum(1),phase,weights


class ResidualTrajectoryNetwork(nn.Module):
    """Zero-initialized, motion-bounded correction to a frozen predictor."""
    def __init__(self, incumbent, mean, scale, stability=False):
        super().__init__()
        self.incumbent=incumbent
        for parameter in incumbent.parameters():parameter.requires_grad_(False)
        self.motion=MotionStateNetwork(mean,scale,False,False)
        self.stability=stability
        self.correction=nn.Sequential(nn.Linear(49,48),nn.SiLU(),nn.Linear(48,5))
        nn.init.zeros_(self.correction[-1].weight)
        nn.init.zeros_(self.correction[-1].bias)
        with torch.no_grad():self.correction[-1].bias[-1]=-5.

    def forward(self, context, experts, horizon):
        old,phase,_=self.incumbent(context,experts,horizon)
        z,stable,velocity,acceleration,endpoint,remaining=self.motion.state(context)
        parameters=self.correction(torch.cat((z,horizon/.25),1))
        trajectory=context[:,64:148].reshape(-1,12,7)
        movement=trajectory[:,:,:2].square().sum(2).mean(1,keepdim=True).sqrt()
        evidence=(movement/.025).clamp(0,1)
        # Joint current offset and finite-horizon displacement; no displacement
        # when every observed candidate is stationary.
        correction=(.015*parameters[:,:2].tanh()+.12*horizon*parameters[:,2:4].tanh())*evidence
        gate=parameters[:,4:5].sigmoid() if self.stability else torch.zeros_like(horizon)
        result=(old+correction)*(1-gate)+stable*gate
        return result,phase,torch.cat((1-gate,gate),1)


def cap_delta(delta, pixels, max_lead):
    lead = delta*pixels
    limit = min(.3, max(0., max_lead))*np.linalg.norm(pixels)
    return delta*np.minimum(1., limit/np.maximum(1e-9, np.linalg.norm(lead, axis=-1, keepdims=True)))


class UnifiedPrediction:
    def __init__(self, module, metadata):
        self.module, self.metadata = module, metadata
        self.history = UnifiedHistory(metadata.get("history_sample_interval_ms", 0.),
                                      metadata.get("intrinsic_stability", False),
                                      metadata.get("history_recent_interval_ms", 0.),
                                      metadata.get("recent_velocity_expert", False))

    def reset(self):
        self.history.reset()

    def update(self, raw, stable, timestamp, feature, hidden, screen_size, horizon_ms, max_lead_fraction=.12, reset=False):
        pixels = np.maximum(1., np.asarray(screen_size)-1)
        context, experts, n = self.history.update(timestamp, np.asarray(raw)/pixels, np.asarray(stable)/pixels, feature, hidden, reset)
        horizon = float(np.clip(horizon_ms, 0, 250))
        if n < 4 or horizon_ms <= 0 or max_lead_fraction <= 0:
            return tuple(stable), {"mode":"prediction_warmup" if n < 4 else "disabled", "horizon_ms":horizon,
                                  "lead_px":[0., 0.], "lead_distance_px":0.}
        with torch.inference_mode():
            delta, phase, weights = self.module(torch.from_numpy(context[None]), torch.from_numpy(experts[None]), torch.tensor([[horizon/1000]], dtype=torch.float32))
        displacement = cap_delta(delta[0].numpy(), pixels, max_lead_fraction)*pixels
        if not np.isfinite(displacement).all():
            self.reset()
            raise ValueError("non-finite unified prediction output")
        point = np.asarray(raw)+displacement
        probabilities = phase.softmax(-1)[0].numpy()
        lead = point-np.asarray(stable)
        return tuple(point), {"mode":("prediction_fixation", "prediction_pursuit", "prediction_saccade")[int(probabilities.argmax())],
            "horizon_ms":horizon, "requested_horizon_ms":float(horizon_ms), "sample_count":n,
            "lead_px":lead.tolist(), "lead_distance_px":float(np.linalg.norm(lead)),
            "prediction_from_raw_px":displacement.tolist(), "phase_probabilities":probabilities.tolist(),
            "expert_weights":weights[0].numpy().tolist(), "shared_temporal_state":bool(self.metadata.get("shared_temporal_state", False)),
            "target_source_ms":timestamp+horizon, "label_source":"offline_reconstructed_gaze_proxy"}
