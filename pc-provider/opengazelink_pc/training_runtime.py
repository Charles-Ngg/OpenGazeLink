"""Shared CPU training runtime configuration."""
from __future__ import annotations

import os
import math


def configure_training_threads(requested: int | None = None) -> int:
    """Use all available CPU cores for offline training by default."""
    detected = int(os.cpu_count() or 1)
    if requested is None:
        requested = int(os.environ.get("OPENGAZELINK_TRAIN_THREADS", "0") or 0)
    threads = max(1, min(detected, int(requested) if int(requested) > 0 else detected))
    import torch
    torch.set_num_threads(threads)
    try:
        # Offline calibration consists of many independent tensor kernels.
        # Keep inter-op parallelism high enough to occupy the available logical
        # CPUs; callers can still cap the total with OPENGAZELINK_TRAIN_THREADS.
        torch.set_num_interop_threads(threads)
    except RuntimeError:
        # Torch only permits changing inter-op threads before parallel work starts.
        pass
    return threads


def training_batch_size(default: int = 256) -> int:
    """Keep short personal captures from becoming one optimizer step per epoch.

    At 30 FPS a complete spatial capture can have only ~1,100 supervised
    frames. A 2,048-frame batch gave it just 30 updates in a 30-epoch fit.
    Leave the explicit environment override available for offline experiments.
    """
    import os
    try:
        value = int(os.environ.get("OPENGAZELINK_TRAIN_BATCH", str(default)))
    except ValueError:
        value = default
    return max(256, min(8192, value))


def spatial_update_budget(replay_report: dict) -> int:
    """Allow a 120 FPS spatial capture to converge; retain the 30 FPS budget.

    Use measured raw source rate, not requested camera settings or the replay
    sampling ceiling. A missing/invalid report retains the conservative budget.
    """
    try:
        source_hz = float(replay_report.get("source_fps_estimate", 0))
    except (TypeError, ValueError):
        source_hz = 0.
    return 2160 if math.isfinite(source_hz) and source_hz >= 100 else 300


def spatial_epoch_limit(requested_epochs: int, supervised_frames: int, batch_size: int,
                        max_updates: int | None = 300) -> int:
    """Budget updates (default ~300), rounded to whole epochs, across rates.

    A faster camera supplies more correlated frames, not a reason to train
    twice as long. Experiments may increase/disable this cap explicitly;
    short requested runs and validation early stopping still win.
    """
    if max_updates is None:
        return requested_epochs
    if max_updates < 1:
        raise ValueError("max_updates must be positive or None")
    updates_per_epoch = max(1, math.ceil(supervised_frames / batch_size))
    return min(requested_epochs, max(1, math.ceil(max_updates / updates_per_epoch)))
