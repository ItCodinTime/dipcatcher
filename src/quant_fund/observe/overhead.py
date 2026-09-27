"""Measure the cost of a disabled span versus an enabled in-process span.

Numbers from this function are one process on one machine. They are not a
published benchmark and they do not include network export.
"""

from __future__ import annotations

import os
import time
from typing import Any

from quant_fund.observe.metrics import reset_metrics
from quant_fund.observe.tracing import clear_spans, span


def measure(*, iterations: int = 4000) -> dict[str, Any]:
    """Time ``iterations`` empty spans with observability off, then on.

    The OTLP endpoint is cleared for the measurement so enabled mode does not
    open sockets. The previous environment is restored before returning.
    """
    if iterations < 1:
        raise ValueError("iterations must be positive")
    saved_observe = os.environ.get("DIPCATCHER_OBSERVE")
    saved_endpoint = os.environ.get("DIPCATCHER_OTEL_ENDPOINT")
    os.environ.pop("DIPCATCHER_OTEL_ENDPOINT", None)
    try:
        os.environ.pop("DIPCATCHER_OBSERVE", None)
        direct_ns = _time(_direct, iterations)
        disabled_ns = _time(_disabled_span, iterations)
        os.environ["DIPCATCHER_OBSERVE"] = "1"
        clear_spans()
        reset_metrics()
        enabled_ns = _time(_enabled_span, iterations)
        clear_spans()
        reset_metrics()
    finally:
        _restore("DIPCATCHER_OBSERVE", saved_observe)
        _restore("DIPCATCHER_OTEL_ENDPOINT", saved_endpoint)
    return {
        "iterations": iterations,
        "direct_ns_per_call": direct_ns,
        "disabled_ns_per_call": disabled_ns,
        "enabled_ns_per_call": enabled_ns,
        "disabled_minus_direct_ns": disabled_ns - direct_ns,
        "enabled_minus_disabled_ns": enabled_ns - disabled_ns,
        "otel_endpoint_cleared": True,
        "note": (
            "Single-process measurement with no network export. Not a market or research result."
        ),
    }


def _direct() -> None:
    return None


def _disabled_span() -> None:
    with span("ingest"):
        return None


def _enabled_span() -> None:
    with span("ingest"):
        return None


def _time(fn: Any, iterations: int) -> float:
    for _ in range(min(50, iterations)):
        fn()
    started = time.perf_counter_ns()
    for _ in range(iterations):
        fn()
    return (time.perf_counter_ns() - started) / iterations


def _restore(name: str, value: str | None) -> None:
    if value is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = value
