"""Static import graph plus runtime traces for the two origins."""

from __future__ import annotations

from pathlib import Path

from quant_fund.config.models import AppConfig
from quant_fund.parity.reference import run_backtest_session
from quant_fund.parity.session import MarketSession
from quant_fund.parity.shadow import run_shadow_session
from quant_fund.parity.strategy import FixedWeightStrategy, TargetDecision
from quant_fund.parity.trace import (
    RUNNER_MODULES,
    CallTracer,
    code_path_guard,
    compare_import_graphs,
    imports_from_source,
    static_callees,
)


def test_runner_modules_share_an_import_graph() -> None:
    graph = compare_import_graphs(RUNNER_MODULES[0], RUNNER_MODULES[1])
    assert graph["equal"], graph
    assert graph["only_left"] == []
    assert graph["only_right"] == []


def test_import_parser_sees_a_direct_import() -> None:
    left = imports_from_source("import json\nfrom quant_fund.parity.replay import replay_session\n")
    right = imports_from_source("import json\n")
    assert "json" in left and "quant_fund.parity.replay" in left
    assert left - right == {"quant_fund.parity.replay"}


def test_static_callees_catch_a_shadow_only_helper() -> None:
    def _extra(weight: float) -> float:
        return weight

    class _Clean:
        def decide(self, ctx):
            del ctx
            return TargetDecision(weights={"A": 0.05, "B": -0.02})

    class _Dirty:
        def decide(self, ctx):
            del ctx
            _extra(0.05)
            return TargetDecision(weights={"A": 0.05, "B": -0.02})

    only_shadow = static_callees(_Dirty.decide) - static_callees(_Clean.decide)
    assert "_extra" in only_shadow


def test_runtime_trace_of_divergent_strategies(session: MarketSession, config: AppConfig) -> None:
    def _extra(weight: float) -> float:
        return weight

    class _Clean:
        def decide(self, ctx):
            del ctx
            return TargetDecision(weights={"A": 0.05, "B": -0.02})

    class _Dirty:
        def decide(self, ctx):
            del ctx
            _extra(0.05)
            return TargetDecision(weights={"A": 0.05, "B": -0.02})

    clean = _Clean()
    dirty = _Dirty()
    backtest = run_backtest_session(session, clean, config)
    shadow = run_shadow_session(session, dirty, config)
    guard = code_path_guard(
        backtest_fn=clean.decide,
        shadow_fn=dirty.decide,
        backtest_traces=backtest.call_traces,
        shadow_traces=shadow.call_traces,
    )
    assert guard["static_callees_equal"] is False
    assert "_extra" in guard["callees_only_shadow"]
    assert guard["runtime_trace_equal"] is False
    assert any(name.endswith("._extra") for name in guard["runtime_only_shadow"])
    assert guard["same_code_path"] is False
    assert guard["import_graph_equal"] is True
    assert guard["live_pnl_claim"] is False


def test_tracer_chains_and_records_a_nested_call() -> None:
    def _inner() -> int:
        return 1

    def _outer() -> int:
        return _inner()

    seen: list[str] = []

    def _previous(frame, event, arg) -> None:
        del frame, arg
        if event == "call":
            seen.append("prev")

    import sys

    previous = sys.getprofile()
    sys.setprofile(_previous)
    try:
        with CallTracer() as tracer:
            _outer()
    finally:
        sys.setprofile(previous)
    assert any(name.endswith("._outer") for name in tracer.calls)
    assert any(name.endswith("._inner") for name in tracer.calls)
    assert seen
    assert sys.getprofile() is previous


def test_package_does_not_reference_live_submission() -> None:
    root = Path("src/quant_fund/parity")
    banned = (
        "requests.",
        "httpx.",
        "aiohttp",
        "place_order",
        "send_order",
        "ib_insync",
        "alpaca_trade",
    )
    for path in root.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for token in banned:
            assert token not in text, f"{path} contains {token}"
    broker_hits = [
        path.name
        for path in root.rglob("*.py")
        if "SimulatedBroker" in path.read_text(encoding="utf-8")
    ]
    assert "replay.py" in broker_hits


def test_fixed_strategy_trace_is_stable(session: MarketSession, config: AppConfig) -> None:
    strategy = FixedWeightStrategy({"A": 0.05, "B": -0.02})
    first = run_backtest_session(session, strategy, config)
    second = run_shadow_session(session, strategy, config)
    assert first.code_path_digest == second.code_path_digest
    assert first.call_traces == second.call_traces
