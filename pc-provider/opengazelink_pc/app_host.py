from __future__ import annotations

from pathlib import Path
import threading
import time

from .config import DEFAULT_CONFIG_PATH, load_config
from .control_server import ControlApplication, ControlServer
from .engine import EyeTrackingEngine
from .model_registry import ModelRegistry
from .pairing import PairingService


class ApplicationHost:
    def __init__(self, config_path: Path = DEFAULT_CONFIG_PATH) -> None:
        self.config_path = config_path
        self.config = load_config(config_path)
        self.registry = ModelRegistry()
        self.engine = EyeTrackingEngine(self.config, self.registry)
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._control: ControlServer | None = None
        self._mode = "idle"
        self.pairing: PairingService | None = None
        self.application = ControlApplication(
            self.config,
            config_path,
            registry=self.registry,
            engine=self.engine,
            host_status=self.status,
            enter_background=self.enter_background,
            accept_pairing=self.accept_pairing,
            forget_pairing=self.forget_pairing,
            config_updated=self._config_updated,
            shutdown_application=self.request_shutdown,
        )
        if self.config.paired_phone_id:
            self._set_allowed_source_ip("0.0.0.0")
        self.pairing = PairingService(
            self.config.udp_bind,
            self.config.discovery_port,
            lambda: self.application.config.udp_port,
            self._paired_phone,
            self._set_allowed_source_ip,
        )

    def _set_allowed_source_ip(self, address: str | None) -> None:
        setter = getattr(self.engine.camera, "set_allowed_source_ip", None)
        if setter is not None:
            setter(address)

    def _camera_source_status(self) -> dict:
        getter = getattr(self.engine.camera, "source_status", None)
        return getter() if getter is not None else {
            "last_source_ip": "", "allowed_source_ip": "",
        }

    def status(self) -> dict:
        with self._lock:
            result = {
                "mode": self._mode,
                "control_active": self._control is not None,
                "tracking": self.engine.is_tracking(),
            }
            if self.pairing is not None:
                result["pairing"] = self.pairing.status()
            result["phone_source"] = self._camera_source_status()
            return result

    def _paired_phone(self) -> tuple[str, str]:
        config = self.application.config
        return config.paired_phone_id, config.paired_phone_name

    def _config_updated(self, config) -> None:
        self.config = config
        if not config.paired_phone_id:
            self._set_allowed_source_ip(None)
            return
        pending = self.pairing.pending_phone(config.paired_phone_id) if self.pairing else None
        self._set_allowed_source_ip(pending.address if pending else "0.0.0.0")

    def accept_pairing(self, phone_id: str) -> dict:
        pairing = self.pairing
        if pairing is None:
            raise RuntimeError("phone discovery is unavailable")
        pending = pairing.pending_phone(str(phone_id or "").strip())
        if pending is None:
            raise RuntimeError("the phone discovery request has expired")
        self.application.update_config({
            "paired_phone_id": pending.phone_id,
            "paired_phone_name": pending.name,
        })
        self.config = self.application.config
        self._set_allowed_source_ip(pending.address)
        return pairing.status()

    def forget_pairing(self) -> dict:
        self.application.update_config({
            "paired_phone_id": "",
            "paired_phone_name": "",
        })
        self.config = self.application.config
        self._set_allowed_source_ip(None)
        return self.pairing.status() if self.pairing is not None else {}

    def open_control(self, open_browser: bool = True) -> str:
        with self._lock:
            if self._control is None:
                self._control = ControlServer(self.application)
                self._control.start(open_browser=False)
            self._mode = "control"
            url = self._control.url
        if open_browser:
            self._control.start(open_browser=True)
        return url

    def start_runtime(self) -> None:
        with self._lock:
            # Runtime is allowed to start before the phone connects. The engine
            # waits for a fresh frame and never queues stale frames.
            self.engine.start_tracking()
            self._mode = "runtime"

    def enter_background(self) -> None:
        def transition() -> None:
            # Let the HTTP response reach the browser before closing its server.
            time.sleep(0.2)
            with self._lock:
                server = self._control
                self._control = None
                self._mode = "runtime"
            if server is not None:
                server.close()

        threading.Thread(target=transition, name="enter-background", daemon=True).start()

    def request_shutdown(self) -> None:
        threading.Timer(0.2, self.stop).start()

    def handle_command(self, request: dict) -> dict:
        command = str(request.get("command") or "")
        if command == "control":
            return {"ok": True, "url": self.open_control(bool(request.get("open_browser", True)))}
        if command == "runtime":
            self.start_runtime()
            self.enter_background()
            return {"ok": True, **self.status()}
        if command == "stop":
            self._stop.set()
            return {"ok": True}
        if command == "status":
            return {"ok": True, **self.status()}
        if command == "wake":
            return {"ok": True}
        return {"ok": False, "error": f"unknown command: {command}"}

    def wait(self) -> None:
        self._stop.wait()

    def stop(self) -> None:
        self._stop.set()

    def close(self) -> None:
        with self._lock:
            server = self._control
            self._control = None
        if server is not None:
            server.close()
        if self.pairing is not None:
            self.pairing.close()
        self.application.close()
