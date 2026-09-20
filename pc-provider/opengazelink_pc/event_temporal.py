"""Causal event processing in source time, independent of spatial-model weights.

Fixation uses 1 Euro filtering (Casiez et al., 2012). Observed saccades use a
bounded continuation and a stopping-distance model when deceleration is
supported. No stimulus, trial identity, future samples or trained trajectory
teacher enter this module.
"""
from collections import deque
import numpy as np
from .one_euro import OneEuroFilter2D
from .saccade_prediction import short_continuation, braking_prediction, MAX_CONTINUATION_S


def fit_landing(times_ms, points, noise):
    """Legacy compressed-exponential estimator, retained for comparisons.

    p(t) = start + amplitude * (1 - exp(-(t/tau)**shape)).  Amplitude is
    solved analytically for each time-scale/shape; disagreement is retained.
    Coordinates and noise are in degrees (screen-plane local approximation).
    """
    times, points = np.asarray(times_ms, float), np.asarray(points, float)
    if len(times) < 4 or times[-1] - times[0] < 16:
        return None
    elapsed = times - times[0]
    displacement = points - points[0]
    traveled = np.linalg.norm(displacement[-1])
    if traveled < max(.6, 4 * noise):
        return None
    tau, shape = np.meshgrid(np.arange(12., 72., 2.), [2., 3., 4.])
    f = 1 - np.exp(-np.power(elapsed[None, :] / tau.reshape(-1, 1), shape.reshape(-1, 1)))
    amplitude = f @ displacement / np.maximum(1e-9, (f * f).sum(1))[:, None]
    residual = displacement[None] - f[..., None] * amplitude[:, None]
    cost = np.mean(np.minimum(np.sum(residual * residual, axis=2), (4 * noise + .3)**2), axis=1)
    lengths = np.linalg.norm(amplitude, axis=1)
    feasible = ((lengths >= traveled * .9) & (lengths <= traveled * 2.5)
                & (lengths <= 35.) & (f[:, -1] >= .25))
    if not feasible.any():
        return None
    cost[~feasible] = np.inf
    best = int(np.argmin(cost))
    if cost[best] > max(.12, 2 * noise)**2:
        return None
    plausible = feasible & (cost <= cost[best] + max(.035, noise * .6)**2)
    candidates = points[0] + amplitude[plausible]
    landing = np.median(candidates, axis=0)
    spread = float(np.max(np.linalg.norm(candidates - landing, axis=1)))
    if spread > max(.5, traveled * .22):
        return None
    return landing, dict(endpoint_spread_deg=spread, fit_rmse_deg=float(np.sqrt(cost[best])),
                         hypotheses=int(plausible.sum()), elapsed_ms=float(elapsed[-1]))


class EventTemporalFilter:
    def __init__(self, min_cutoff=1., beta=2., derivative_cutoff=1., *, prediction_mode="braking"):
        self.filter = OneEuroFilter2D(min_cutoff, beta, derivative_cutoff)
        # The continuation-only arm supports a controlled offline ablation.
        if prediction_mode not in ("velocity", "braking"):
            raise ValueError("unsupported saccade prediction mode")
        self.prediction_mode = prediction_mode
        self.set_stability_profile()
        self.reset()

    def set_stability_profile(self, profile=None):
        """Stage-two evidence refines gating; it is not a user-facing switch."""
        profile = profile or {}
        values = [float(profile.get("noise_prior_deg", .12)), float(profile.get("noise_floor_deg", .04)),
                  float(profile.get("settle_ms", 16.))]
        if not np.isfinite(values).all():
            raise ValueError("invalid stability profile")
        self.noise_prior_deg = float(np.clip(values[0], .04, 1.5))
        self.noise_floor_deg = float(np.clip(values[1], .04, .75))
        self.settle_ms = float(np.clip(values[2], 16., 50.))
        self.stability_calibrated = bool(profile.get("calibrated", False))
        self.reset()

    def configure(self, min_cutoff, beta, derivative_cutoff):
        self.filter.configure(min_cutoff, beta, derivative_cutoff)
        self.reset()

    def reset(self):
        self.samples = deque(maxlen=256)
        self.noise_samples = deque(maxlen=120)
        self.event = None
        self.quiet_since = None
        self.filter.reset()
        self.last_head = None
        self.last_head_ms = None
        self.head_guard_until_ms = -np.inf
        self.last_output = None
        self.last_scale = None
        self.onset_ms = None
        self.candidate_since = None
        self.rearm_ms = -np.inf
        self.pursuit_active = False
        self.pursuit_since = None
        self.pursuit_quiet_since = None

    def update(self, raw, timestamp_ms, screen_size, *, pixels_per_degree,
               head_rotation=None, horizon_ms=0., max_lead_fraction=.12, smooth=True):
        raw, scale = np.asarray(raw, float), float(pixels_per_degree)
        now = float(timestamp_ms)
        size = np.asarray(screen_size, float)
        if (raw.shape != (2,) or size.shape != (2,) or np.any(size <= 1)
                or not np.isfinite(np.r_[raw, now, scale, size, horizon_ms, max_lead_fraction]).all() or scale <= 0):
            self.reset()
            raise ValueError("invalid event temporal observation")
        if self.samples and (now <= self.samples[-1][0] or now - self.samples[-1][0] > 100):
            self.reset()
        # Avoid apparent eye velocity caused by a changing conversion scale.
        if self.last_scale is not None and abs(scale / self.last_scale - 1) > .10:
            self.reset()
        self.last_scale = scale
        wh = size - 1.
        head_speed = 0.
        if head_rotation is not None:
            rotation = np.asarray(head_rotation, float)
            if rotation.shape == (3, 3) and np.isfinite(rotation).all():
                if self.last_head is not None and self.last_head_ms is not None:
                    angle = np.degrees(np.arccos(np.clip((np.trace(rotation @ self.last_head.T) - 1) / 2, -1., 1.)))
                    head_speed = angle * 1000 / max(1., now - self.last_head_ms)
                self.last_head = rotation.copy()
                self.last_head_ms = now
        if head_speed >= 35:
            self.head_guard_until_ms = now + 60
        self.samples.append((now, raw.copy()))
        while len(self.samples) > 2 and now - self.samples[0][0] > 200:
            self.samples.popleft()
        times = np.array([s[0] for s in self.samples])
        points = np.array([s[1] for s in self.samples]) / scale
        if self.event is None and len(times) >= 6:
            # Estimate noise without requiring low *frame-to-frame* velocity:
            # noisy fixation itself can have very large numerical derivatives.
            # Exclude the latest two points so an onset is not absorbed as noise.
            t = (times[:-2] - times[-3]) / 1000
            q = points[:-2]
            design = np.c_[np.ones(len(t)), t]
            fit = np.linalg.lstsq(design, q, rcond=None)[0]
            residual = np.linalg.norm(q - design @ fit, axis=1)
            self.noise_samples.append(float(np.median(residual) / 1.177))
        noise = max(self.noise_floor_deg, float(np.median(self.noise_samples))
                    if self.noise_samples else self.noise_prior_deg)
        # Do not let an underdetermined startup fit erase the calibrated prior.
        if times[-1] - times[0] < 100:
            noise = max(noise, self.noise_prior_deg)
        mode, fit_info, landing = "fixation", {}, None
        prediction_delta = None
        speed = 0.
        onset = False
        coherence = 0.
        if len(times) >= 3:
            # At high frame rates two adjacent derivatives amplify pixel noise.
            # Use at least 16 ms, with an actual observed middle sample; keep
            # native three-sample evidence at 30/60 Hz instead of inventing it.
            first = min(len(times)-3, int(np.searchsorted(times, now-16, side="right"))-1)
            first = max(0, first)
            middle = max(first+1, min(len(times)-2, (first+len(times)-1)//2))
            evidence_ids = [first, middle, len(times)-1]
            velocity = np.diff(points[evidence_ids], axis=0) / (np.diff(times[evidence_ids])[:, None] / 1000)
            speed = float(np.linalg.norm(velocity[-1]))
            coherence = float(np.dot(velocity[0], velocity[1]) / max(1e-9, np.prod(np.linalg.norm(velocity, axis=1))))
            distance = np.linalg.norm(points[-1] - points[first])
            onset = (len(times) >= 6 and times[-1] - times[0] >= 40 and coherence > .8
                     and min(np.linalg.norm(velocity, axis=1)) > 60 and distance > max(.7, noise * 7)
                     and now >= self.head_guard_until_ms)
            strong = onset and distance > max(2., noise * 10)
            if self.event is None:
                self.candidate_since = (now if self.candidate_since is None else self.candidate_since) if onset else None
            confirmed = onset and (strong or (self.candidate_since is not None and now-self.candidate_since >= 8))
            if self.event is None and confirmed and (now >= self.rearm_ms or strong):
                self.onset_ms = float(times[first])
                self.event = [s for s in self.samples if s[0] >= self.onset_ms - 50]
                self.quiet_since = None
                self.pursuit_active = False
                self.pursuit_since = self.pursuit_quiet_since = None
                self.candidate_since = None
                self.filter.reset()
            elif self.event is not None:
                self.event.append((now, raw.copy()))
            if self.event is not None:
                mode = "saccade_observed"
                quiet = speed < max(25., noise * 2000 / max(1., now - times[middle]))
                self.quiet_since = (now if self.quiet_since is None else self.quiet_since) if quiet else None
                displacement = (self.event[-2][1] - self.event[0][1]) / scale
                step = points[-1] - points[-2]
                reverse = np.dot(step, displacement) < -max(.2, noise * 3) * max(.1, np.linalg.norm(displacement))
                finished = reverse or (self.quiet_since is not None and now - self.quiet_since >= self.settle_ms) or now - self.onset_ms >= 160
                if finished:
                    self.event = None
                    self.rearm_ms = now + 40
                    self.filter.reset()  # a fixation starts at observed landing, never predicted endpoint
                    mode = "landing"
                elif now >= self.head_guard_until_ms and 0 < horizon_ms <= 150 and max_lead_fraction > 0:
                    event_times = np.array([s[0] for s in self.event])
                    event_points = np.array([s[1] for s in self.event]) / scale
                    fit = braking_prediction(event_times, event_points, horizon_ms, noise) if self.prediction_mode == "braking" else None
                    if fit is not None:
                        delta, endpoint, fit_info = fit
                        prediction_delta, landing = delta * scale, endpoint * scale
                        mode = "saccade_landing_prediction"
                    else:
                        continuation = short_continuation(event_times, event_points, horizon_ms, noise)
                        if continuation is not None:
                            prediction_delta = continuation * scale
                            mode = "saccade_velocity_prediction"
                            fit_info["trusted_horizon_ms"] = min(float(horizon_ms), MAX_CONTINUATION_S * 1000,
                                                                float(np.diff(event_times[-3:]).mean()))
            elif len(times) >= 4 and mode != "landing":
                pursuit_first = min(len(times)-3, int(np.searchsorted(times, now-60)))
                selected = np.arange(len(times)) >= pursuit_first
                t = (times[selected] - now) / 1000
                q = points[selected]
                if len(t) >= 3 and t[-1] - t[0] >= .020:
                    design = np.c_[np.ones(len(t)), t]
                    fit = np.linalg.lstsq(design, q, rcond=None)[0]
                    residual = np.median(np.linalg.norm(q - design @ fit, axis=1))
                    movement = np.linalg.norm(fit[1]) * (t[-1] - t[0])
                    moving = movement > max(.35, residual * 5, noise * 4)
                    quiet = movement < max(.18, residual * 3, noise * 2)
                    self.pursuit_since = (now if self.pursuit_since is None else self.pursuit_since) if moving else None
                    self.pursuit_quiet_since = (now if self.pursuit_quiet_since is None else self.pursuit_quiet_since) if quiet else None
                    if not self.pursuit_active and self.pursuit_since is not None and now-self.pursuit_since >= 24:
                        self.pursuit_active = True
                    if self.pursuit_active and self.pursuit_quiet_since is not None and now-self.pursuit_quiet_since >= 40:
                        self.pursuit_active = False
                    if self.pursuit_active:
                        mode = "pursuit"  # confirmed motion; no unconditional extrapolation
        # Smoothing belongs to the fixation state, not to a parallel baseline.
        # Motion, head movement and landing always use the current observation.
        stability_active = bool(smooth and mode == "fixation" and self.event is None
                                and now >= self.head_guard_until_ms)
        if not stability_active:
            self.filter.reset()
        stable = (np.array(self.filter.update(tuple(raw / wh), now / 1000)) * wh
                  if stability_active else raw.copy())
        output = stable.copy()
        if prediction_delta is not None:
            lead = prediction_delta
            limit = max(0., min(.3, max_lead_fraction)) * np.linalg.norm(wh)
            output = raw + lead * min(1., limit / max(1e-9, np.linalg.norm(lead)))
        lead = output - raw
        self.last_output = output.copy()
        return tuple(output), dict(mode="event_" + mode, horizon_ms=float(horizon_ms),
            lead_px=lead.tolist(), lead_distance_px=float(np.linalg.norm(lead)),
            speed_deg_s=speed, noise_deg=noise, head_speed_deg_s=float(head_speed),
            onset_candidate=bool(onset), onset_coherence=coherence,
            pursuit_confirmed=self.pursuit_active,
            sample_count=len(self.samples), landing_px=landing.tolist() if landing is not None else None,
            prediction_active=prediction_delta is not None,
            stability_active=stability_active, stability_calibrated=self.stability_calibrated,
            stable_px=stable.tolist(),
            head_guard_active=now < self.head_guard_until_ms,
            prediction_model="event_kinematic_v3",
            prediction_reference="bounded source-time motion estimate; no independent eye-tracker truth", **fit_info)
