"""Frame deadlines without Python 3.8/Windows' coarse Event.wait rounding."""
import ctypes
import os
import time


class FramePacer:
    def __enter__(self):
        self.handle = None
        if os.name == 'nt':
            from ctypes import wintypes
            self.kernel = ctypes.WinDLL('kernel32', use_last_error=True)
            self.kernel.CreateWaitableTimerExW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD]
            self.kernel.CreateWaitableTimerExW.restype = wintypes.HANDLE
            self.kernel.SetWaitableTimer.argtypes = [wintypes.HANDLE, ctypes.POINTER(ctypes.c_longlong), wintypes.LONG, ctypes.c_void_p, ctypes.c_void_p, wintypes.BOOL]
            self.kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
            self.kernel.CloseHandle.argtypes = [wintypes.HANDLE]
            self.handle = self.kernel.CreateWaitableTimerExW(None, None, 2, 0x1F0003)
            if not self.handle:
                raise ctypes.WinError(ctypes.get_last_error())
        return self

    def wait_until(self, deadline, stop):
        if stop.is_set():
            return True
        remaining = deadline - time.perf_counter()
        if remaining > 0:
            if self.handle:
                due = ctypes.c_longlong(-max(1, int(remaining * 10_000_000)))
                if not self.kernel.SetWaitableTimer(self.handle, ctypes.byref(due), 0, None, None, False):
                    raise ctypes.WinError(ctypes.get_last_error())
                result = self.kernel.WaitForSingleObject(self.handle, 1000)
                if result != 0:
                    raise RuntimeError('Frame pacing timer failed: %s' % result)
            else:
                stop.wait(remaining)
        return stop.is_set()

    def __exit__(self, *args):
        if self.handle:
            self.kernel.CloseHandle(self.handle)
            self.handle = None
