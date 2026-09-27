"""OpenTelemetry-compatible spans exported as OTLP/HTTP JSON.

When ``DIPCATCHER_OBSERVE`` is unset, ``span`` yields ``None`` after one flag
check and does not allocate ids, touch the metric registry, or open a socket.
The official OpenTelemetry SDK is intentionally not imported: the collector
speaks OTLP, and skipping the SDK keeps the disabled path free of that
dependency.
"""

from __future__ import annotations

import http.client
import json
import os
import time
import urllib.parse
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any

from quant_fund.observe.flags import STAGES, observe_enabled, otel_endpoint
from quant_fund.observe.metrics import record_error, record_latency

_stack: ContextVar[tuple[Span, ...]] = ContextVar("dipcatcher_span_stack", default=())
_finished: list[Span] = []
_MAX_FINISHED = 256


@dataclass
class Span:
    name: str
    trace_id: str
    span_id: str
    parent_span_id: str | None
    start_time_unix_nano: int
    end_time_unix_nano: int | None = None
    attributes: dict[str, Any] = field(default_factory=dict)
    status: str = "ok"
    correlation_id: str | None = None


def finished_spans() -> list[Span]:
    return list(_finished)


def clear_spans() -> None:
    _finished.clear()


@contextmanager
def span(name: str, **attributes: Any) -> Iterator[Span | None]:
    """Record ``name`` when observability is enabled. Otherwise do nothing."""
    if not observe_enabled():
        yield None
        return
    if name not in STAGES:
        raise ValueError(f"unknown stage {name}")
    parent_stack = _stack.get()
    parent = parent_stack[-1] if parent_stack else None
    from quant_fund.observe.logging import get_correlation_id

    current = Span(
        name=name,
        trace_id=parent.trace_id if parent is not None else os.urandom(16).hex(),
        span_id=os.urandom(8).hex(),
        parent_span_id=None if parent is None else parent.span_id,
        start_time_unix_nano=time.time_ns(),
        attributes={key: _attr_value(value) for key, value in attributes.items()},
        correlation_id=get_correlation_id(),
    )
    token = _stack.set((*parent_stack, current))
    started = time.perf_counter()
    try:
        yield current
    except Exception:
        current.status = "error"
        record_error(name)
        raise
    finally:
        current.end_time_unix_nano = time.time_ns()
        record_latency(name, time.perf_counter() - started)
        _stack.reset(token)
        _finished.append(current)
        if len(_finished) > _MAX_FINISHED:
            del _finished[: len(_finished) - _MAX_FINISHED]
        _export(current)


def _attr_value(value: Any) -> str | int | float | bool:
    if isinstance(value, bool | int | float | str):
        return value
    return str(value)


def _export(current: Span) -> None:
    endpoint = otel_endpoint()
    if endpoint is None:
        return
    parsed = urllib.parse.urlsplit(endpoint)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        record_error("otel_export")
        return
    body = json.dumps(_otlp(current)).encode("utf-8")
    port = parsed.port
    if port is None:
        port = 443 if parsed.scheme == "https" else 80
    path = parsed.path or "/"
    if parsed.query:
        path = f"{path}?{parsed.query}"
    connection: http.client.HTTPConnection | None = None
    try:
        if parsed.scheme == "https":
            connection = http.client.HTTPSConnection(parsed.hostname, port, timeout=1.0)
        else:
            connection = http.client.HTTPConnection(parsed.hostname, port, timeout=1.0)
        connection.request(
            "POST",
            path,
            body=body,
            headers={"Content-Type": "application/json"},
        )
        response = connection.getresponse()
        response.read()
    except OSError:
        record_error("otel_export")
    finally:
        if connection is not None:
            connection.close()


def _otlp(current: Span) -> dict[str, Any]:
    attributes = [
        _otlp_attribute("correlation_id", current.correlation_id or ""),
        *(_otlp_attribute(key, value) for key, value in current.attributes.items()),
    ]
    span_body: dict[str, Any] = {
        "traceId": current.trace_id,
        "spanId": current.span_id,
        "name": current.name,
        "startTimeUnixNano": str(current.start_time_unix_nano),
        "endTimeUnixNano": str(current.end_time_unix_nano or current.start_time_unix_nano),
        "attributes": attributes,
        "status": {"code": 2 if current.status == "error" else 1},
    }
    if current.parent_span_id is not None:
        span_body["parentSpanId"] = current.parent_span_id
    return {
        "resourceSpans": [
            {
                "resource": {
                    "attributes": [_otlp_attribute("service.name", "dipcatcher")],
                },
                "scopeSpans": [
                    {
                        "scope": {"name": "quant_fund.observe"},
                        "spans": [span_body],
                    }
                ],
            }
        ]
    }


def _otlp_attribute(key: str, value: str | int | float | bool) -> dict[str, Any]:
    if isinstance(value, bool):
        encoded: dict[str, Any] = {"boolValue": value}
    elif isinstance(value, int):
        encoded = {"intValue": str(value)}
    elif isinstance(value, float):
        encoded = {"doubleValue": value}
    else:
        encoded = {"stringValue": value}
    return {"key": key, "value": encoded}
