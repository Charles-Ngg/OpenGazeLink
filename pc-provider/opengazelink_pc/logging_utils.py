from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
import sys
import threading

from .paths import LOG_DIR


def configure_logging() -> logging.Logger:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("eyetracing")
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = RotatingFileHandler(
            LOG_DIR / "eyetracing.log",
            maxBytes=2 * 1024 * 1024,
            backupCount=4,
            encoding="utf-8",
        )
        handler.setFormatter(logging.Formatter(
            "%(asctime)s %(levelname)s %(threadName)s %(message)s",
        ))
        logger.addHandler(handler)

    def process_exception(exception_type, exception, traceback) -> None:
        logger.critical("uncaught exception", exc_info=(exception_type, exception, traceback))

    def thread_exception(args) -> None:
        logger.error(
            "uncaught thread exception",
            exc_info=(args.exc_type, args.exc_value, args.exc_traceback),
        )

    sys.excepthook = process_exception
    threading.excepthook = thread_exception
    return logger
