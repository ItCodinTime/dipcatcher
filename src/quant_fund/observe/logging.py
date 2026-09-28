"""JSON logs with a correlation id. Silent unless observability is enabled."""

from __future__ import annotations

import json
import sys
import uuid
from contextvars import ContextVar
from datetime import UTC, datetime
from typing import Any

from quant_fund.observe.flags import observe_enabled

_correlation: ContextVar[str | None] = ContextVar("dipcatcher_correlation_id", default=None)
_sink: list[dict[str, Any]] | None = None


def new_correlation_id() -> str:
    value = uuid.uuid4().hex
    _correlation.set(value)
    return value


def get_correlation_id() -> str | None:
    return _correlation.get()


def set_log_sink(sink: list[dict[str, Any]] | None) -> None:
    """Tests capture records here. ``None`` writes JSON lines to stderr."""
    global _sink
    _sink = sink


def log_event(event: str, **fields: Any) -> None:
    """Emit one JSON object. Returns immediately when observability is off.

    Field names ending in ``_key``, ``_token``, or ``_secret``, and names
    containing ``password``, are replaced with ``[redacted]``.
    """
    if not observe_enabled():
        return
    record: dict[str, Any] = {
        "ts": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
        "level": str(fields.pop("level", "info")),
        "event": event,
        "correlation_id": get_correlation_id(),
        "service": "dipcatcher",
    }
    for key, value in fields.items():
        record[str(key)] = "[redacted]" if _secret_field(str(key)) else value
    if _sink is not None:
        _sink.append(record)
        return
    sys.stderr.write(json.dumps(record, sort_keys=True, default=str) + "\n")


def _secret_field(key: str) -> bool:
    lowered = key.lower().replace("-", "_")
    if "password" in lowered or lowered in {
        "key",
        "token",
        "secret",
        "authorization",
        "apikey",
        "api_key",
        "private_key",
        "privatekey",
        "session_key",
        "access_token",
    }:
        return True
    return lowered.endswith(("_key", "_token", "_secret"))
