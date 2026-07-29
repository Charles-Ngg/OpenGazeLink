from __future__ import annotations

from dataclasses import dataclass
import json
import platform
from pathlib import Path
import socket
import threading
import time
import uuid
from typing import Callable

from .paths import USER_ROOT


DISCOVERY_MAGIC = "EYETRACING_DISCOVERY_V1"
DISCOVERY_VERSION = 1
INSTANCE_PATH = USER_ROOT / "instance.json"


def _load_instance_id(path: Path = INSTANCE_PATH) -> str:
    if path.exists():
        try:
            value = str(json.loads(path.read_text(encoding="utf-8")).get("instance_id") or "")
            if value:
                return value
        except Exception:
            pass
    value = str(uuid.uuid4())
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps({"instance_id": value}, indent=2), encoding="utf-8")
    temporary.replace(path)
    return value


@dataclass
class PendingPhone:
    phone_id: str
    name: str
    address: str
    last_seen: float


class PairingService:
    def __init__(
        self,
        bind: str,
        port: int,
        data_port: Callable[[], int],
        paired_phone: Callable[[], tuple[str, str]],
        on_paired_source: Callable[[str], None],
        instance_path: Path = INSTANCE_PATH,
    ) -> None:
        self.instance_id = _load_instance_id(instance_path)
        self.pc_name = platform.node() or "OpenGazeLink PC"
        self.bind = bind
        self.port = int(port)
        self._data_port = data_port
        self._paired_phone = paired_phone
        self._on_paired_source = on_paired_source
        self._pending: dict[str, PendingPhone] = {}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._socket.bind((bind, port))
        self.port = int(self._socket.getsockname()[1])
        self._socket.settimeout(0.25)
        self._thread = threading.Thread(target=self._run, name="phone-discovery", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                packet, address = self._socket.recvfrom(8192)
            except socket.timeout:
                self._expire()
                continue
            except OSError:
                break
            try:
                self._handle(packet, address)
            except Exception:
                continue

    def _handle(self, packet: bytes, address: tuple[str, int]) -> None:
        payload = json.loads(packet.decode("utf-8"))
        if payload.get("magic") != DISCOVERY_MAGIC or payload.get("type") != "discover":
            return
        if int(payload.get("version", 0)) != DISCOVERY_VERSION:
            return
        phone_id = str(payload.get("phone_id") or "").strip()
        nonce = str(payload.get("nonce") or "").strip()
        if not phone_id or not nonce:
            return
        phone_name = str(payload.get("phone_name") or "Android phone")[:80]
        requested_instance = str(payload.get("instance_id") or "")
        paired_id, _paired_name = self._paired_phone()
        accepted = bool(
            paired_id and paired_id == phone_id
            and (not requested_instance or requested_instance == self.instance_id)
        )
        with self._lock:
            self._pending[phone_id] = PendingPhone(
                phone_id, phone_name, address[0], time.monotonic(),
            )
        if accepted:
            self._on_paired_source(address[0])
        response = {
            "magic": DISCOVERY_MAGIC,
            "type": "offer",
            "version": DISCOVERY_VERSION,
            "nonce": nonce,
            "instance_id": self.instance_id,
            "pc_name": self.pc_name,
            "data_port": int(self._data_port()),
            "accepted": accepted,
        }
        encoded = json.dumps(response, separators=(",", ":")).encode("utf-8")
        self._socket.sendto(encoded, address)

    def _expire(self) -> None:
        threshold = time.monotonic() - 15.0
        with self._lock:
            self._pending = {
                key: phone for key, phone in self._pending.items()
                if phone.last_seen >= threshold
            }

    def pending_phone(self, phone_id: str) -> PendingPhone | None:
        with self._lock:
            return self._pending.get(phone_id)

    def status(self) -> dict:
        self._expire()
        paired_id, paired_name = self._paired_phone()
        with self._lock:
            pending = [
                {
                    "phone_id": phone.phone_id,
                    "name": phone.name,
                    "address": phone.address,
                    "age_ms": max(0.0, (time.monotonic() - phone.last_seen) * 1000.0),
                }
                for phone in self._pending.values()
                if phone.phone_id != paired_id
            ]
        return {
            "instance_id": self.instance_id,
            "pc_name": self.pc_name,
            "discovery_port": self.port,
            "paired_phone_id": paired_id,
            "paired_phone_name": paired_name,
            "pending": pending,
        }

    def close(self) -> None:
        self._stop.set()
        self._socket.close()
        self._thread.join(timeout=1.0)
