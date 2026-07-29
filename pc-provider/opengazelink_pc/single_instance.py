from __future__ import annotations

import ctypes
from multiprocessing.connection import Client, Listener
import os
import threading
from typing import Callable


MUTEX_NAME = "Local\\OpenGazeLinkProviderV1"
PIPE_NAME = r"\\.\pipe\OpenGazeLinkProviderV1"
PIPE_AUTHKEY = b"OpenGazeLinkProviderV1"
ERROR_ALREADY_EXISTS = 183


class SingleInstance:
    def __init__(self, name: str = MUTEX_NAME) -> None:
        self._handle = None
        self.is_primary = True
        if os.name != "nt":
            return
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.CreateMutexW.argtypes = (ctypes.c_void_p, ctypes.c_bool, ctypes.c_wchar_p)
        kernel32.CreateMutexW.restype = ctypes.c_void_p
        handle = kernel32.CreateMutexW(None, False, name)
        if not handle:
            raise ctypes.WinError(ctypes.get_last_error())
        self._handle = handle
        self.is_primary = ctypes.get_last_error() != ERROR_ALREADY_EXISTS

    def close(self) -> None:
        if self._handle is not None:
            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel32.CloseHandle.argtypes = (ctypes.c_void_p,)
            kernel32.CloseHandle.restype = ctypes.c_bool
            kernel32.CloseHandle(self._handle)
            self._handle = None


class CommandServer:
    def __init__(self, handler: Callable[[dict], dict]) -> None:
        self._handler = handler
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="instance-ipc", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def _run(self) -> None:
        listener = Listener(PIPE_NAME, family="AF_PIPE", authkey=PIPE_AUTHKEY)
        while not self._stop.is_set():
            try:
                connection = listener.accept()
            except (OSError, EOFError):
                break
            try:
                request = connection.recv()
                response = self._handler(request if isinstance(request, dict) else {})
                connection.send(response)
            except Exception as error:
                try:
                    connection.send({"ok": False, "error": str(error)})
                except Exception:
                    pass
            finally:
                connection.close()

    def close(self) -> None:
        self._stop.set()
        try:
            send_command({"command": "wake"}, timeout_s=0.5)
        except Exception:
            pass
        self._thread.join(timeout=1.0)


def send_command(command: dict, timeout_s: float = 3.0) -> dict:
    # AF_PIPE has no public connect timeout. The named pipe connection itself is
    # local and should complete immediately once the primary process is ready.
    connection = Client(PIPE_NAME, family="AF_PIPE", authkey=PIPE_AUTHKEY)
    try:
        connection.send(command)
        if not connection.poll(timeout_s):
            raise TimeoutError("OpenGazeLink instance did not answer")
        response = connection.recv()
        return response if isinstance(response, dict) else {"ok": False}
    finally:
        connection.close()
