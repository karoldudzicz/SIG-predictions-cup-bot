"""Logging setup: human-readable console output + rotating JSON-lines file.

Structured fields are passed via `extra`, e.g.
    logger.info("Order placed", extra={"exchange_id": ex_id, "price": 0.42})
They show up as key=value on the console and as top-level JSON keys in the file.
All timestamps are UTC.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

LOG_FILE_NAME = "bot.log"
MAX_BYTES = 10 * 1024 * 1024  # 10 MB per file
BACKUP_COUNT = 10  # bot.log.1 ... bot.log.10

# Third-party loggers that are too chatty at INFO/DEBUG.
_NOISY_LOGGERS = ("aiohttp", "websockets", "asyncio", "realtime", "urllib3")

# Attributes every LogRecord has; anything else on the record came from `extra`.
_STANDARD_ATTRS = set(vars(logging.LogRecord("", 0, "", 0, "", None, None))) | {
    "message",
    "asctime",
    "taskName",
}


def _extra_fields(record: logging.LogRecord) -> dict[str, Any]:
    return {
        key: value
        for key, value in record.__dict__.items()
        if key not in _STANDARD_ATTRS and not key.startswith("_")
    }


def _utc_timestamp(record: logging.LogRecord) -> str:
    return datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(timespec="milliseconds")


class JsonFormatter(logging.Formatter):
    """One JSON object per line - easy to grep, jq, or load into pandas later."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": _utc_timestamp(record),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        payload.update(_extra_fields(record))
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        if record.stack_info:
            payload["stack"] = self.formatStack(record.stack_info)
        return json.dumps(payload, default=str, ensure_ascii=False)


class ConsoleFormatter(logging.Formatter):
    """Single readable line, with structured fields appended as key=value."""

    def format(self, record: logging.LogRecord) -> str:
        line = f"{_utc_timestamp(record)} {record.levelname:<8} {record.name}: {record.getMessage()}"
        extras = _extra_fields(record)
        if extras:
            line += " | " + " ".join(f"{key}={value!r}" for key, value in extras.items())
        if record.exc_info:
            line += "\n" + self.formatException(record.exc_info)
        if record.stack_info:
            line += "\n" + self.formatStack(record.stack_info)
        return line


def setup_logging(level: str = "INFO", log_dir: Path | str = "logs") -> Path:
    """Configure the root logger. Safe to call more than once. Returns the log file path."""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / LOG_FILE_NAME

    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)
        handler.close()
    root.setLevel(level)

    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setFormatter(ConsoleFormatter())
    root.addHandler(console_handler)

    file_handler = RotatingFileHandler(
        log_path, maxBytes=MAX_BYTES, backupCount=BACKUP_COUNT, encoding="utf-8"
    )
    file_handler.setFormatter(JsonFormatter())
    root.addHandler(file_handler)

    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)

    # Route warnings.warn(...) (e.g. deprecation notices) through logging too.
    logging.captureWarnings(True)

    return log_path
