"""Histograms, JSON logs, nested spans, and a local OTLP post."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from quant_fund.observe.logging import (
    get_correlation_id,
    log_event,
    new_correlation_id,
    set_log_sink,
)
from quant_fund.observe.metrics import (
    METRIC_HELP,
    histogram_quantile_upper,
    record_latency,
    render_prometheus,
    reset_metrics,
    set_data_freshness,
    set_simulation_state,
)
from quant_fund.observe.serve import start_metrics_server, stop_metrics_server
from quant_fund.observe.tracing import clear_spans, finished_spans, span


def _enable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DIPCATCHER_OBSERVE", "1")
    monkeypatch.delenv("DIPCATCHER_OTEL_ENDPOINT", raising=False)
    reset_metrics()
    clear_spans()


def test_histogram_buckets_are_cumulative(monkeypatch: pytest.MonkeyPatch) -> None:
    _enable(monkeypatch)
    record_latency("ingest", 0.01)
    text = render_prometheus()
    assert 'dipcatcher_stage_latency_seconds_bucket{stage="ingest",le="0.001"} 0' in text
    assert 'dipcatcher_stage_latency_seconds_bucket{stage="ingest",le="0.005"} 0' in text
    assert 'dipcatcher_stage_latency_seconds_bucket{stage="ingest",le="0.01"} 1' in text
    assert 'dipcatcher_stage_latency_seconds_bucket{stage="ingest",le="+Inf"} 1' in text
    assert 'dipcatcher_stage_latency_seconds_count{stage="ingest"} 1' in text
    assert histogram_quantile_upper("ingest", 0.99) == 0.01
    record_latency("ingest", 0.001)
    record_latency("ingest", 10.0)
    text = render_prometheus()
    assert 'dipcatcher_stage_latency_seconds_bucket{stage="ingest",le="0.001"} 1' in text
    assert 'dipcatcher_stage_latency_seconds_bucket{stage="ingest",le="0.01"} 2' in text
    assert 'dipcatcher_stage_latency_seconds_bucket{stage="ingest",le="+Inf"} 3' in text


def test_simulation_gauges_say_simulation_and_not_live(monkeypatch: pytest.MonkeyPatch) -> None:
    _enable(monkeypatch)
    help_text = METRIC_HELP["dipcatcher_simulation_pnl"]
    assert "simulation" in help_text.lower() or "Simulated" in help_text
    assert "live" in help_text.lower()
    set_simulation_state(pnl=-1.5, gross_exposure=2.0, net_exposure=-0.25)
    set_data_freshness(12.0)
    text = render_prometheus()
    assert "Not a live P&L claim" in text
    assert "dipcatcher_simulation_pnl -1.5" in text
    assert "dipcatcher_simulation_gross_exposure 2" in text
    assert "dipcatcher_simulation_net_exposure -0.25" in text
    assert "dipcatcher_data_freshness_seconds 12" in text


def test_unknown_stage_raises_only_when_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    _enable(monkeypatch)
    with pytest.raises(ValueError), span("broker"):
        pass
    with pytest.raises(ValueError):
        record_latency("broker", 0.1)
    monkeypatch.delenv("DIPCATCHER_OBSERVE", raising=False)
    with span("broker"):
        pass
    record_latency("broker", 0.1)


def test_logs_carry_correlation_ids_and_redact_secrets(monkeypatch: pytest.MonkeyPatch) -> None:
    _enable(monkeypatch)
    sink: list[dict[str, object]] = []
    set_log_sink(sink)
    try:
        correlation = new_correlation_id()
        log_event("stage", api_key="secret", password="x", key_id="visible", level="info")
        assert sink[0]["correlation_id"] == correlation
        assert sink[0]["correlation_id"] == get_correlation_id()
        assert sink[0]["api_key"] == "[redacted]"
        assert sink[0]["password"] == "[redacted]"
        assert sink[0]["key_id"] == "visible"
    finally:
        set_log_sink(None)


def test_nested_spans_share_a_trace(monkeypatch: pytest.MonkeyPatch) -> None:
    _enable(monkeypatch)
    new_correlation_id()
    with span("ingest") as outer:
        assert outer is not None
        with span("features") as inner:
            assert inner is not None
            assert inner.trace_id == outer.trace_id
            assert inner.parent_span_id == outer.span_id
            assert len(inner.trace_id) == 32
            assert len(inner.span_id) == 16
            assert inner.correlation_id == get_correlation_id()
    names = [item.name for item in finished_spans()]
    assert names == ["features", "ingest"]


class _Handler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:  # noqa: N802 - stdlib handler name
        length = int(self.headers.get("Content-Length", "0"))
        payload = self.rfile.read(length)
        payloads = getattr(self.server, "payloads", None)
        if isinstance(payloads, list):
            payloads.append(payload)
        self.send_response(200)
        self.end_headers()

    def log_message(self, format: str, *args: object) -> None:
        return


def test_otlp_json_posts_one_span(monkeypatch: pytest.MonkeyPatch) -> None:
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    server.payloads = []  # type: ignore[attr-defined]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = int(server.server_address[1])
    try:
        _enable(monkeypatch)
        monkeypatch.setenv("DIPCATCHER_OTEL_ENDPOINT", f"http://127.0.0.1:{port}/v1/traces")
        with span("model", instrument="synthetic") as current:
            assert current is not None
            with span("decision"):
                pass
        with pytest.raises(RuntimeError, match="keep"), span("simulated_execution"):
            raise RuntimeError("keep")
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
    payloads = server.payloads  # type: ignore[attr-defined]
    assert len(payloads) == 3
    first = json.loads(payloads[0])
    span_body = first["resourceSpans"][0]["scopeSpans"][0]["spans"][0]
    assert span_body["name"] == "decision"
    assert len(span_body["traceId"]) == 32
    assert len(span_body["spanId"]) == 16
    assert span_body["status"]["code"] == 1
    assert "parentSpanId" in span_body
    resource = first["resourceSpans"][0]["resource"]["attributes"]
    assert resource[0]["key"] == "service.name"
    parent = json.loads(payloads[1])["resourceSpans"][0]["scopeSpans"][0]["spans"][0]
    assert parent["name"] == "model"
    assert "parentSpanId" not in parent
    failed = json.loads(payloads[2])["resourceSpans"][0]["scopeSpans"][0]["spans"][0]
    assert failed["status"]["code"] == 2


def test_export_failure_does_not_mask_the_result(monkeypatch: pytest.MonkeyPatch) -> None:
    _enable(monkeypatch)
    monkeypatch.setenv("DIPCATCHER_OTEL_ENDPOINT", "http://127.0.0.1:1/v1/traces")

    def work() -> int:
        with span("decision"):
            return 7

    assert work() == 7

    def explode() -> None:
        with span("decision"):
            raise RuntimeError("x")

    with pytest.raises(RuntimeError, match="x"):
        explode()
    text = render_prometheus()
    assert 'dipcatcher_errors_total{stage="otel_export"}' in text


def test_metrics_http_serves_text(monkeypatch: pytest.MonkeyPatch) -> None:
    _enable(monkeypatch)
    set_data_freshness(3.5)
    port = start_metrics_server(0)
    try:
        import http.client

        connection = http.client.HTTPConnection("127.0.0.1", port, timeout=2)
        connection.request("GET", "/metrics")
        response = connection.getresponse()
        body = response.read().decode()
        assert response.status == 200
        assert "dipcatcher_data_freshness_seconds" in body
        connection.request("GET", "/other")
        missing = connection.getresponse()
        missing.read()
        assert missing.status == 404
        connection.close()
    finally:
        stop_metrics_server()


def test_export_protocol_failure_never_reaches_the_caller(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A collector answering with a malformed response is a transport failure."""
    import http.client

    _enable(monkeypatch)
    monkeypatch.setenv("DIPCATCHER_OTEL_ENDPOINT", "http://127.0.0.1:4318/v1/traces")

    class GarbageConnection:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def request(self, *args: object, **kwargs: object) -> None:
            pass

        def getresponse(self) -> object:
            raise http.client.BadStatusLine("not http")

        def close(self) -> None:
            pass

    monkeypatch.setattr(http.client, "HTTPConnection", GarbageConnection)

    def work() -> int:
        with span("decision"):
            return 7

    assert work() == 7

    def explode() -> None:
        with span("decision"):
            raise RuntimeError("x")

    with pytest.raises(RuntimeError, match="x"):
        explode()
    assert 'dipcatcher_errors_total{stage="otel_export"}' in render_prometheus()


def test_redaction_covers_hyphenated_and_unseparated_names(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _enable(monkeypatch)
    sink: list[dict[str, object]] = []
    set_log_sink(sink)
    try:
        log_event(
            "stage",
            **{"x-api-key": "s1"},
            apikey="s2",
            privatekey="s3",
            session_secret="s4",
            auth_token="s5",
            key_id="visible",
        )
        record = sink[0]
        for field in ("x-api-key", "apikey", "privatekey", "session_secret", "auth_token"):
            assert record[field] == "[redacted]", field
        assert record["key_id"] == "visible"
    finally:
        set_log_sink(None)
