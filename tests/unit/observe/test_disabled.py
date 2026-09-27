"""Observability stays out of the process until the flag is set."""

from __future__ import annotations

from pathlib import Path

import pytest

from quant_fund.observe.flags import observe_enabled
from quant_fund.observe.logging import log_event, set_log_sink
from quant_fund.observe.metrics import (
    record_error,
    record_latency,
    render_prometheus,
    reset_metrics,
)
from quant_fund.observe.tracing import span


def test_default_is_off(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DIPCATCHER_OBSERVE", raising=False)
    assert observe_enabled() is False
    reset_metrics()
    with span("not-a-stage") as current:
        assert current is None
    record_latency("ingest", 1.0)
    record_error("also-unknown")
    assert render_prometheus() == ""
    sink: list[dict[str, object]] = []
    set_log_sink(sink)
    try:
        log_event("quiet", password="nope")
        assert sink == []
    finally:
        set_log_sink(None)


def test_sources_do_not_import_otel_or_prometheus() -> None:
    root = Path("src/quant_fund/observe")
    for path in root.glob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert "opentelemetry" not in text
        assert "prometheus_client" not in text
        assert "urlopen" not in text
