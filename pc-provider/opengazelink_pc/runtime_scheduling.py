"""Small Windows scheduling helpers for the real-time tracking path."""
from __future__ import annotations

import ctypes
from ctypes import wintypes
import os
import sys
import threading


class TrackingThreadSwitch:
    """Bound Python GIL handoff delay without adding threads or spinning.

    Native decode/inference releases the GIL but must reacquire it to publish.
    The default 5ms interval lets small Python stages stall native completions.
    A scoped 1ms interval leaves library thread counts and OS affinity unchanged.
    """
    interval_s = .001
    _lock = threading.Lock()
    _users = 0
    _previous = None
    _applied = None

    def __init__(self):
        self.active = False
        self.status = {}

    def __enter__(self):
        cls = type(self)
        with cls._lock:
            if self.active:
                raise RuntimeError('Thread switch scope already entered')
            if cls._users == 0:
                cls._previous = sys.getswitchinterval()
                cls._applied = min(cls._previous, self.interval_s)
                sys.setswitchinterval(cls._applied)
                cls._applied = sys.getswitchinterval()
            cls._users += 1
            self.active = True
            self.status = dict(python_switch_interval_ms=sys.getswitchinterval()*1000.)
        return self

    def __exit__(self, *args):
        cls = type(self)
        with cls._lock:
            if not self.active:
                return
            cls._users -= 1
            self.active = False
            if cls._users == 0:
                # Respect another owner's explicit change during this scope.
                if sys.getswitchinterval() == cls._applied:
                    sys.setswitchinterval(cls._previous)
                cls._previous = cls._applied = None

def _kernel():
    api = ctypes.WinDLL("kernel32", use_last_error=True)
    api.GetCurrentProcess.restype = wintypes.HANDLE
    api.GetCurrentThread.restype = wintypes.HANDLE
    api.GetPriorityClass.argtypes = [wintypes.HANDLE]
    api.GetPriorityClass.restype = wintypes.DWORD
    api.SetPriorityClass.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    api.SetThreadPriority.argtypes = [wintypes.HANDLE, ctypes.c_int]
    return api


def configure_tracking_process() -> dict:
    """Raise priority without constraining tracking/training to arbitrary cores."""
    result = {"platform": os.name, "priority": False, "affinity": False, "cores": [],
              "policy": "above_normal; OS-managed affinity; pipeline threads above_normal"}
    if os.name != "nt":
        return result
    try:
        kernel32 = _kernel()
        process = kernel32.GetCurrentProcess()
        result["previous_priority"] = int(kernel32.GetPriorityClass(process))
        # ABOVE_NORMAL preserves game responsiveness while winning short scheduling races.
        result["priority"] = bool(kernel32.SetPriorityClass(process, 0x00008000))
        # Last logical CPUs can be E-cores/SMT siblings. Affinity does not reserve
        # them and would also restrict later offline training in this process.
        if not result["priority"]:
            result["error"] = ctypes.get_last_error()
    except Exception as error:
        result["error"] = str(error)
    return result


def restore_tracking_process(state: dict) -> None:
    if os.name == "nt" and state.get("priority") and state.get("previous_priority"):
        api = _kernel()
        api.SetPriorityClass(api.GetCurrentProcess(), state["previous_priority"])


def set_realtime_thread_priority() -> bool:
    if os.name != "nt":
        return False
    try:
        # THREAD_PRIORITY_ABOVE_NORMAL; avoid TIME_CRITICAL/HIGH which can starve the game.
        api = _kernel()
        return bool(api.SetThreadPriority(api.GetCurrentThread(), 1))
    except Exception:
        return False
