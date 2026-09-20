"""Evaluate completed event trials from an interrupted capture in a new workspace.

The source session and active models are never changed. Original whole-trial
splits are preserved; incomplete attempts are excluded before model inference.
"""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from opengazelink_pc.video_dataset import align_frames, read_jsonl
from opengazelink_pc.video_session import write_json
from opengazelink_pc.unified_capture import coverage_report


def completed_attempts(events, discarded=()):
    return {(e['trial_id'], e.get('capture_segment', 0)) for e in events
            if e.get('trial_complete') and e.get('capture_segment', 0) not in discarded}


def restore_capture_clocks(frames, captured):
    """Restore recorded probes only for exactly matching sensor/send timestamps.

    Raw archive reconstruction retains packet timing but not the live receiver's
    roundtrip probes. Never infer those probes for unobserved archive frames.
    """
    def identity(row):
        timing = row.get('timing', {})
        return timing.get('phone_sensor_time_ns'), timing.get('phone_send_time_ns')
    clocks = {identity(row): row.get('timing', {}) for row in captured if all(identity(row))}
    restored = 0
    for row in frames:
        saved = clocks.get(identity(row), {})
        fields = {key: saved[key] for key in ('phone_to_pc_offset_ns', 'clock_probe_rtt_ms',
                  'clock_probe_uncertainty_ms', 'clock_probe_age_ms', 'clock_probe_samples') if key in saved}
        if fields:
            row['timing'].update(fields)
            restored += 1
    return restored


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('session', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--reuse-replay', action='store_true', help='Reuse an existing independent workspace, never the source session')
    args = parser.parse_args()
    source, output = args.session.resolve(), args.output.resolve()
    if source == output:
        raise ValueError('Output must differ from source')
    metadata = json.loads((source / 'session.json').read_text(encoding='utf-8'))
    if not metadata.get('plan') or any(s.get('calibration_stage') != 'events_v1' for s in metadata['plan']):
        raise ValueError('Expected an event-stage recording')
    if args.reuse_replay:
        audit = json.loads((output / 'partial-capture-audit.json').read_text(encoding='utf-8'))
        if Path(audit['source']).resolve() != source:
            raise ValueError('Independent workspace belongs to a different source')
        replay = json.loads((output / 'replay-report.json').read_text(encoding='utf-8'))
    else:
        output.mkdir(parents=True, exist_ok=False)
        for name in ('session.json', 'stimulus.jsonl', 'frames.jsonl'):
            shutil.copy2(source / name, output / name)
        shutil.copytree(source / 'base-model', output / 'base-model')
        # Large immutable archives can be shared; all derived files live in output.
        (output / 'raw-camera').mkdir()
        for path in (source / 'raw-camera').iterdir():
            if path.is_file():
                target = output / 'raw-camera' / path.name
                try:
                    os.link(path, target)
                except OSError:
                    shutil.copy2(path, target)
        from opengazelink_pc.video_replay import prepare_replay
        replay = prepare_replay(output, workers=args.workers, progress=lambda p: print(p, flush=True))
    frames = read_jsonl(Path(replay['manifest']))
    restored = restore_capture_clocks(frames, read_jsonl(source / 'frames.jsonl'))
    events = read_jsonl(output / 'stimulus.jsonl')
    discarded = metadata.get('discarded_segments', [])
    complete = completed_attempts(events, discarded)
    rows = align_frames(frames, events, discarded_segments=discarded)
    eligible = [r for r in rows if (r.get('trial_id'), r.get('capture_segment', 0)) in complete]
    coverage = coverage_report(eligible, metadata['plan'])
    accepted = {(t['trial_id'], t['capture_segment']) for t in coverage['trials'] if t['accepted']}
    indices = {r['index'] for r in eligible if (r['trial_id'], r.get('capture_segment', 0)) in accepted}
    with (output / 'frames.jsonl').open('w', encoding='utf-8') as stream:
        for frame in frames:
            if frame['index'] in indices:
                stream.write(json.dumps(frame, ensure_ascii=False) + '\n')
    audit = dict(source=str(source), original_state=metadata.get('state'),
                 clock_probes_restored=restored,
                 completed_display_attempts=len(complete), coverage=coverage,
                 retained_frames=len(indices),
                 bands_by_split={split: dict(Counter(s['amplitude_band'] for s in metadata['plan']
                     if s['split'] == split and any(s['trial_id'] == trial for trial, _ in accepted)))
                     for split in ('train', 'validation', 'test')})
    write_json(output / 'partial-capture-audit.json', audit)
    print(json.dumps({k: v for k, v in audit.items() if k != 'coverage'}, ensure_ascii=False), flush=True)
    if not indices:
        raise ValueError('No complete usable trials; source retained')
    import torch
    torch.set_num_threads(2)
    from opengazelink_pc.event_evaluation import evaluate_session
    report = evaluate_session(output, progress=lambda p: print(p, flush=True))
    print(json.dumps(dict(report=report['training_directory'], stability=report['stability_calibration']),
                     ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
