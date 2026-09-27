"""SLO files fail closed when a sample is missing."""

from __future__ import annotations

import math

import pytest

from quant_fund.observe.metrics import record_latency, reset_metrics
from quant_fund.observe.slo import current_slo_snapshot, evaluate, load_slos, objectives_met


def test_committed_slos_and_no_data(monkeypatch: pytest.MonkeyPatch) -> None:
    slos = load_slos()
    assert {item.metric for item in slos} == {
        "latency_p99_seconds",
        "error_ratio",
        "freshness_seconds",
    }
    missing = evaluate({}, slos)
    assert [item.status for item in missing] == ["no_data", "no_data", "no_data"]
    assert objectives_met(missing) is False
    passing = evaluate(
        {"latency_p99_seconds": 1.0, "error_ratio": 0.0, "freshness_seconds": 10.0},
        slos,
    )
    assert objectives_met(passing) is True
    failing = evaluate(
        {"latency_p99_seconds": 9.0, "error_ratio": 0.5, "freshness_seconds": 10.0},
        slos,
    )
    assert [item.status for item in failing] == ["fail", "fail", "pass"]
    monkeypatch.setenv("DIPCATCHER_OBSERVE", "1")
    reset_metrics()
    record_latency("model", 0.01)
    snapshot = current_slo_snapshot()
    assert snapshot["latency_p99_seconds"] == 0.01
    assert snapshot["error_ratio"] == 0.0
    assert "freshness_seconds" not in snapshot
    assert not math.isnan(snapshot["latency_p99_seconds"])
