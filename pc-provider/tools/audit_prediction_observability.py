"""Time-domain forecasting controls on a retained run. Never publishes a model."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from opengazelink_pc.unified_prediction import cap_delta


def histories(points, times, segments, frame_ids, ages):
    result = np.empty((len(frame_ids), len(ages), points.shape[1]), dtype=np.float32)
    starts = np.r_[0, np.flatnonzero(np.diff(segments)) + 1]
    ends = np.r_[starts[1:], len(points)]
    for start, end in zip(starts, ends):
        local = np.flatnonzero((frame_ids >= start) & (frame_ids < end))
        query = times[frame_ids[local], None] - ages[None]
        for axis in range(points.shape[1]):
            result[local, :, axis] = np.interp(query, times[start:end], points[start:end, axis])
    return result


def run(source, output):
    output.mkdir(parents=True, exist_ok=False)
    data = dict(np.load(source / 'motion-examples.npz'))
    frames = dict(np.load(source / 'motion-frames.npz'))
    record = json.loads((source / 'run.json').read_text(encoding='utf8'))
    cfg = json.loads((Path(record['session']) / 'session.json').read_text(encoding='utf8'))['config']
    pixels = np.array([cfg['screen_width'] - 1, cfg['screen_height'] - 1])
    max_lead = cfg.get('extrapolation_max_lead_fraction', .12)
    data = {k: v[np.isclose(data['horizon'][:, 0], .085)] for k, v in data.items()}
    fi = data['frame']
    train, selection, test = [data['split'] == i for i in range(3)]
    raw, target = frames['raw'][fi], data['target']
    ages = np.unique(np.r_[np.arange(0, 100, 1000 / 120), np.arange(100, 301, 25)])
    confidence = data['phase_conf']
    weights = data['quality'] * data.get('interpolation_confidence', 1.)

    def summarize(point):
        error = np.linalg.norm((point - target) * pixels, axis=1)
        result = {}
        for split, mask in zip(('train', 'selection', 'test'), (train, selection, test)):
            groups = {'all': mask, 'uncertain': mask & (confidence < .5)}
            groups.update({name: mask & (data['phase'] == i) & (confidence >= .5)
                           for i, name in enumerate(('fixation', 'pursuit', 'saccade'))})
            result[split] = {name: {'n': int(m.sum()), 'mean_px': float(error[m].mean()),
                                   'p95_px': float(np.percentile(error[m], 95))}
                             for name, m in groups.items() if m.any()}
        return result

    def constrained(point):
        return raw + cap_delta(point - raw, pixels, max_lead)

    results = {'notice': 'Same-session development holdouts, not independent eye truth. '
               'Teacher-input arms are acausal diagnostics, never deployable. '
               'Ridge strength/window selected using validation only; no phase is an input.',
               'source': str(source.resolve()), 'ages_ms': ages.tolist(), 'horizon_ms': 85,
               'raw_hold': summarize(raw), 'teacher_current_oracle': summarize(frames['teacher'][fi]),
               'models': {}}
    saved = {}
    for source_name in ('raw', 'teacher'):
        history = histories(frames[source_name], frames['times'], frames['segments'], fi, ages)
        center = history[:, 0]
        # One coefficient per lag shared by screen axes: translation/rotation
        # equivariant and no learning of where a particular trial occurs.
        x = ((history[:, 1:] - center[:, None]) * pixels).transpose(0, 2, 1)
        y = (target - center) * pixels
        xx = x[train].reshape(-1, x.shape[-1])
        yy = y[train].reshape(-1)
        mass = np.repeat(weights[train], 2)
        gram = xx.T @ (mass[:, None] * xx)
        rhs = xx.T @ (mass * yy)
        best = None
        candidates = []
        for penalty in (.001, .01, .1, 1., 10., 100., 1000.):
            coef = np.linalg.solve(gram + penalty * np.trace(gram) / len(gram) * np.eye(len(gram)), rhs)
            point = constrained(center + (x @ coef) / pixels)
            score = summarize(point)
            cost = score['selection']['all']['mean_px']
            candidates.append({'penalty': penalty, 'selection_px': cost})
            if best is None or cost < best[0]:
                best = (cost, point, coef, penalty, score)
        name = source_name + '_shared_fir'
        saved[name] = best[1]
        results['models'][name] = {'scores': best[4], 'penalty': best[3],
                                  'coefficients': best[2].tolist(), 'validation_search': candidates}
        windows = []
        best = None
        for window in (25, 50, 75, 100, 150, 200, 300):
            local = ages <= window
            age = -ages[local] / 1000
            design = np.c_[np.ones(len(age)), age]
            coef = np.linalg.pinv(design)
            fit = np.einsum('ij,njk->nik', coef, history[:, local])
            point = constrained(fit[:, 0] + .085 * fit[:, 1])
            score = summarize(point)
            cost = score['selection']['all']['mean_px']
            windows.append({'window_ms': window, 'selection_px': cost})
            if best is None or cost < best[0]:
                best = (cost, point, window, score)
        name = source_name + '_linear_velocity'
        results['models'][name] = {'scores': best[3], 'window_ms': best[2], 'validation_search': windows}
        saved[name] = best[1]
    np.savez_compressed(output / 'predictions.npz', frame=fi, **saved)
    (output / 'audit.json').write_text(json.dumps(results, indent=2), encoding='utf8')
    print(json.dumps({name: value['scores']['test'] for name, value in results['models'].items()}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    run(args.source, args.output)
