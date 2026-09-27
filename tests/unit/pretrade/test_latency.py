"""Hot-path latency gate. The distribution is measured, not assumed.

Coverage tracing changes the timed path, so the numeric gate is asserted
only when the process is uninstrumented. The CI ``pretrade-risk`` job runs
``python -m quant_fund.pretrade.bench --gate`` without coverage.
"""

from __future__ import annotations

import sys

from quant_fund.pretrade.bench import P50_LIMIT_NS, P99_LIMIT_NS, run_benchmark


def _coverage_running() -> bool:
    mod = sys.modules.get("coverage")
    if mod is None:
        return False
    current = getattr(getattr(mod, "Coverage", None), "current", None)
    if not callable(current):
        return False
    try:
        return current() is not None
    except Exception:
        return False


def test_allow_path_meets_latency_gate() -> None:
    report = run_benchmark(trials=3, samples=3000, warmup=800, gate=True)
    assert report["implementation"] == "cpython-slots"
    assert report["numba"] is False
    assert report["rust"] is False
    assert report["min_ns"] <= report["p50_ns"] <= report["p99_ns"] <= report["max_ns"]
    assert len(report["trials"]) == 3
    assert "kill_switch" in report["checks"]
    assert "good_faith_violation" in report["checks"]
    within = report["p50_ns"] < P50_LIMIT_NS and report["p99_ns"] < P99_LIMIT_NS
    assert report["gate_pass"] is within
    if not _coverage_running():
        assert report["gate_pass"] is True
