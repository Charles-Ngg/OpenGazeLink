from __future__ import annotations

import argparse
import json
from pathlib import Path
import signal
import sys
import time
from . import runtime_clock

from .config import DEFAULT_CONFIG_PATH
from .paths import ensure_user_layout
from .logging_utils import configure_logging
from .single_instance import CommandServer, SingleInstance, send_command


def _send_existing(command: dict) -> dict:
    deadline = runtime_clock.monotonic() + 3.0
    last_error: Exception | None = None
    while runtime_clock.monotonic() < deadline:
        try:
            return send_command(command)
        except (FileNotFoundError, ConnectionRefusedError, OSError) as error:
            last_error = error
            time.sleep(0.1)
    raise RuntimeError(
        f"OpenGazeLink is running but its command channel is unavailable: {last_error}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="OpenGazeLink production PC provider")
    parser.add_argument(
        "mode", choices=("control", "runtime", "stop", "status", "train-video"),
        nargs="?", default="control",
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--no-open", action="store_true")
    parser.add_argument("--migrate-from", type=Path)
    parser.add_argument("--session", type=Path, help="Saved continuous VIDEO session for offline training")
    parser.add_argument("--base", type=Path, help="Binocular model metadata for VIDEO training")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--no-publish", action="store_true")
    args = parser.parse_args()

    ensure_user_layout(args.migrate_from)
    logger = configure_logging()
    logger.info("starting mode=%s config=%s", args.mode, args.config)
    if args.mode == "train-video":
        if args.session is None:
            parser.error("train-video requires --session")
        from .video_training import train_session
        report = train_session(args.session, base_path=args.base, epochs=args.epochs,
                               publish=not args.no_publish, progress=lambda phase: logger.info("%s", phase))
        logger.info("VIDEO training completed: %s", report["candidate"])
        return
    instance = SingleInstance()
    if not instance.is_primary:
        response = _send_existing({
            "command": args.mode,
            "open_browser": not args.no_open,
        })
        if not response.get("ok"):
            raise RuntimeError(str(response.get("error") or "OpenGazeLink command failed"))
        if args.mode == "status":
            print(json.dumps(response, ensure_ascii=False, indent=2))
        instance.close()
        return

    if args.mode in ("stop", "status"):
        if args.mode == "status":
            print(json.dumps({"ok": True, "running": False}, indent=2))
        instance.close()
        return

    from .app_host import ApplicationHost

    host: ApplicationHost | None = None
    commands: CommandServer | None = None
    try:
        host = ApplicationHost(args.config)
        commands = CommandServer(host.handle_command)
        commands.start()
        signal.signal(signal.SIGINT, lambda *_: host.stop())
        signal.signal(signal.SIGTERM, lambda *_: host.stop())
        if args.mode == "runtime":
            try:
                host.start_runtime()
                print("OpenGazeLink runtime started")
            except Exception as error:
                logger.warning("runtime prerequisites incomplete", exc_info=True)
                print(f"Runtime prerequisites are incomplete: {error}", file=sys.stderr)
                host.open_control(open_browser=not args.no_open)
        else:
            host.open_control(open_browser=not args.no_open)
        host.wait()
    finally:
        if commands is not None:
            commands.close()
        if host is not None:
            host.close()
        instance.close()


if __name__ == "__main__":
    main()
