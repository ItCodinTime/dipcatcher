"""Feature flags for optional observability.

Disabled is the default. Reading the flag does not import OpenTelemetry,
Prometheus, or the network stack.
"""

from __future__ import annotations

import os

_TRUE = frozenset({"1", "true", "yes", "on"})

STAGES = (
    "ingest",
    "features",
    "model",
    "decision",
    "simulated_execution",
)


def observe_enabled() -> bool:
    """Return whether instrumentation should record spans, metrics, and logs."""
    return os.environ.get("DIPCATCHER_OBSERVE", "").strip().lower() in _TRUE


def otel_endpoint() -> str | None:
    """OTLP/HTTP JSON traces URL, for example ``http://127.0.0.1:4318/v1/traces``."""
    value = os.environ.get("DIPCATCHER_OTEL_ENDPOINT", "").strip()
    return value or None


def metrics_host() -> str:
    """Bind address for ``/metrics``. Loopback unless the operator sets a host."""
    value = os.environ.get("DIPCATCHER_METRICS_HOST", "").strip()
    return value or "127.0.0.1"


def metrics_port() -> int | None:
    raw = os.environ.get("DIPCATCHER_METRICS_PORT", "").strip()
    if not raw:
        return None
    port = int(raw)
    if port < 0 or port > 65535:
        raise ValueError("DIPCATCHER_METRICS_PORT is outside 0..65535")
    return port
