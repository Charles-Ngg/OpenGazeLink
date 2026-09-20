"""One high-resolution, monotonic clock for every PC pipeline timestamp.

Python 3.8 on Windows implements monotonic with 15.625 ms GetTickCount64.
Use QPC for intervals, anchored once near the legacy monotonic epoch so local
consumers remain compatible. The anchor's fixed uncertainty cancels in intervals
and phone minimum-transit alignment. Never mix the two clocks within an interval.
"""
import time


_coarse = time.get_clock_info("monotonic").resolution > 0.000001
_before_ns = time.perf_counter_ns()
_anchor_ns = time.monotonic_ns()
_after_ns = time.perf_counter_ns()
_offset_ns = _anchor_ns - (_before_ns + _after_ns) // 2


def monotonic_ns() -> int:
    return time.perf_counter_ns() + _offset_ns if _coarse else time.monotonic_ns()


def monotonic() -> float:
    return monotonic_ns() / 1_000_000_000.0


def info() -> dict:
    source = time.get_clock_info("perf_counter" if _coarse else "monotonic")
    return {"implementation": source.implementation, "resolution_ms": source.resolution * 1000,
            "legacy_resolution_ms": time.get_clock_info("monotonic").resolution * 1000,
            "epoch": "process-fixed legacy monotonic anchor" if _coarse else "monotonic"}
