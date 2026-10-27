"""Structured (JSON-lines) logging.

Use :func:`log_event` instead of formatting free-text messages so every log line carries
the same machine-readable fields: incident_id, operation, stage, outcome, duration_ms and
error_category. Never pass raw incident text, LLM output, or credentials as fields.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any

_RESERVED_FIELD = "cade_fields"


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "event": record.getMessage(),
        }
        payload.update(getattr(record, _RESERVED_FIELD, {}))
        if record.exc_info and record.levelno >= logging.ERROR:
            # Exception type only; full tracebacks can contain incident data.
            payload["exception_type"] = record.exc_info[0].__name__ if record.exc_info[0] else None
        return json.dumps(payload, default=str)


def configure_logging(level: str = "INFO") -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger("cade")
    root.handlers[:] = [handler]
    root.setLevel(level.upper())
    root.propagate = False


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"cade.{name}")


def log_event(
    logger: logging.Logger,
    event: str,
    *,
    level: int = logging.INFO,
    incident_id: str | None = None,
    operation: str | None = None,
    stage: str | None = None,
    outcome: str | None = None,
    duration_ms: float | None = None,
    error_category: str | None = None,
    **extra: Any,
) -> None:
    fields = {
        "incident_id": incident_id,
        "operation": operation,
        "stage": stage,
        "outcome": outcome,
        "duration_ms": round(duration_ms, 1) if duration_ms is not None else None,
        "error_category": error_category,
        **extra,
    }
    logger.log(level, event, extra={_RESERVED_FIELD: {k: v for k, v in fields.items() if v is not None}})
