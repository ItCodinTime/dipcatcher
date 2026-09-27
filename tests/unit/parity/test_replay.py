"""Shadow and backtest reference share one strategy and the simulated broker."""

from __future__ import annotations

from dataclasses import replace
from datetime import timedelta
from pathlib import Path

import pytest

from quant_fund.config.models import AppConfig, FillConvention, RuntimeMode
from quant_fund.execution.simulated_broker import SimulatedBroker
from quant_fund.parity.checker import check_parity
from quant_fund.parity.reference import run_backtest_session
from quant_fund.parity.replay import ReplayOptions
from quant_fund.parity.report import build_report
from quant_fund.parity.session import Bar, MarketSession
from quant_fund.parity.shadow import run_shadow_session
from quant_fund.parity.shortfall import attribute_shortfall
from quant_fund.parity.strategy import FixedWeightStrategy, TargetDecision
from tests.unit.parity.conftest import make_session


def _pair(session: MarketSession, strategy: object, config: AppConfig, **shadow_kw: object):
    backtest = run_backtest_session(session, strategy, config)
    shadow_session = shadow_kw.pop("session", session)
    shadow_strategy = shadow_kw.pop("strategy", strategy)
    options = shadow_kw.pop("options", None)
    shadow = run_shadow_session(shadow_session, shadow_strategy, config, options=options)
    return backtest, shadow


def _causes(backtest, shadow) -> set[str]:
    report = check_parity(backtest, shadow)
    return {row["cause"] for row in report["ledger"]}


def test_zero_divergence_golden(
    session: MarketSession, strategy: FixedWeightStrategy, config: AppConfig
) -> None:
    backtest, shadow = _pair(session, strategy, config)
    report = check_parity(backtest, shadow)
    assert report["match"] is True
    assert report["n_divergent"] == 0
    assert report["n_bars"] == len(session.bars)
    assert report["live_pnl_claim"] is False
    assert backtest.call_traces and backtest.call_traces[0]
    built = build_report(backtest, shadow, backtest_strategy=strategy, shadow_strategy=strategy)
    assert built["summary"]["same_code_path"] is True
    assert built["summary"]["decisions_match"] is True
    shortfall = built["shortfall"]
    assert shortfall["terminal_gap"] == pytest.approx(0.0, abs=1e-6)
    assert shortfall["residual"] == pytest.approx(0.0, abs=1e-6)
    assert shortfall["algebraic_residual"] == pytest.approx(0.0, abs=1e-6)
    assert all(abs(shortfall["components"][name]) < 1e-6 for name in shortfall["components"])


def test_streamed_tape_matches_the_recording(
    session: MarketSession, strategy: FixedWeightStrategy, config: AppConfig
) -> None:
    backtest = run_backtest_session(session, strategy, config)

    def _stream():
        yield from session.bars

    shadow = run_shadow_session(_stream(), strategy, config)
    assert check_parity(backtest, shadow)["match"] is True
    assert shadow.synthetic is True


def test_wall_clock_paces_and_still_matches(
    session: MarketSession, strategy: FixedWeightStrategy, config: AppConfig
) -> None:
    slept: list[float] = []
    paced = ReplayOptions(pacing="wall_clock", speed=86_400.0, sleeper=slept.append)
    backtest = run_backtest_session(session, strategy, config)
    shadow = run_shadow_session(session, strategy, config, options=paced)
    assert check_parity(backtest, shadow)["match"] is True
    assert slept == pytest.approx([1.0, 1.0, 1.0])
    accelerated = ReplayOptions(pacing="accelerated", sleeper=slept.append)
    slept.clear()
    run_shadow_session(session, strategy, config, options=accelerated)
    assert slept == []


def test_close_auction_golden(
    session: MarketSession, strategy: FixedWeightStrategy, config: AppConfig
) -> None:
    config.execution.fill = FillConvention.CLOSE_AUCTION
    config.execution.allow_close_auction = True
    backtest, shadow = _pair(session, strategy, config)
    assert check_parity(backtest, shadow)["match"] is True
    assert backtest.fills.height == len(session.bars)


def test_revision_and_close_are_data_not_fills(
    session: MarketSession, strategy: FixedWeightStrategy, config: AppConfig
) -> None:
    def _bump(bar: Bar) -> Bar:
        if bar.security_id == "A" and bar.event_time == session.bars[0].event_time:
            return replace(bar, revision_id="r1", close=bar.close + 1.0, high=bar.close + 1.0)
        return bar

    bumped = MarketSession(_bump(bar) for bar in session.bars)
    backtest = run_backtest_session(session, strategy, config)
    shadow = run_shadow_session(bumped, strategy, config)
    report = check_parity(backtest, shadow)
    assert report["n_divergent"] == 1
    assert report["ledger"][0]["cause"] == "data"
    assert "revision_id" in report["ledger"][0]["detail"]
    assert "close" in report["ledger"][0]["detail"]
    shortfall = attribute_shortfall(
        backtest.fills,
        shadow.fills,
        {sid: (backtest.end_marks[sid], shadow.end_marks[sid]) for sid in backtest.end_marks},
        backtest_terminal_delta=backtest.terminal_delta,
        paper_terminal_delta=shadow.terminal_delta,
    )
    assert shortfall["residual"] == pytest.approx(0.0, abs=1e-6)
    assert shortfall["terminal_gap"] == pytest.approx(0.0, abs=1e-4)


def test_decision_lag_is_timing(
    session: MarketSession, strategy: FixedWeightStrategy, config: AppConfig
) -> None:
    shadow = run_shadow_session(
        session,
        strategy,
        config,
        options=ReplayOptions(decision_lag=timedelta(seconds=-1)),
    )
    backtest = run_backtest_session(session, strategy, config)
    report = check_parity(backtest, shadow)
    assert report["n_divergent"] == report["n_bars"]
    assert _causes(backtest, shadow) == {"timing"}


def test_lot_size_is_rounding(
    session: MarketSession, strategy: FixedWeightStrategy, config: AppConfig
) -> None:
    shadow = run_shadow_session(session, strategy, config, options=ReplayOptions(lot_size=0.03))
    backtest = run_backtest_session(session, strategy, config)
    assert _causes(backtest, shadow) == {"rounding"}


def test_commission_override_is_costs(
    session: MarketSession, strategy: FixedWeightStrategy, config: AppConfig
) -> None:
    before = config.costs.commission_bps
    shadow = run_shadow_session(
        session,
        strategy,
        config,
        options=ReplayOptions(cost_overrides={"commission_bps": 10.0}),
    )
    assert config.costs.commission_bps == before
    backtest = run_backtest_session(session, strategy, config)
    assert _causes(backtest, shadow) == {"costs"}
    shortfall = attribute_shortfall(
        backtest.fills,
        shadow.fills,
        backtest.end_marks,
        backtest_terminal_delta=backtest.terminal_delta,
        paper_terminal_delta=shadow.terminal_delta,
    )
    assert shortfall["residual"] == pytest.approx(0.0, abs=1e-5)
    assert shortfall["components"]["fees"] != pytest.approx(0.0, abs=1e-6)


def test_fill_price_hook_is_fills(
    session: MarketSession, strategy: FixedWeightStrategy, config: AppConfig
) -> None:
    def _worse(
        *, side: str, open_px: float, close_px: float, decision_px: float, use_close: bool
    ) -> float:
        del decision_px
        base = close_px if use_close else open_px
        return base * (1.01 if side == "buy" else 0.99)

    shadow = run_shadow_session(session, strategy, config, options=ReplayOptions(fill_price=_worse))
    backtest = run_backtest_session(session, strategy, config)
    report = check_parity(backtest, shadow)
    assert report["first_divergence"]["cause"] == "fills"
    assert _causes(backtest, shadow) <= {"fills", "state_drift"}
    assert all(row["detail"] != "after_restart" for row in report["ledger"])
    shortfall = attribute_shortfall(
        backtest.fills,
        shadow.fills,
        backtest.end_marks,
        backtest_terminal_delta=backtest.terminal_delta,
        paper_terminal_delta=shadow.terminal_delta,
    )
    assert shortfall["residual"] == pytest.approx(0.0, abs=1e-4)
    assert shortfall["algebraic_residual"] == pytest.approx(0.0, abs=1e-6)
    assert shortfall["components"]["delay"] != pytest.approx(0.0, abs=1e-6)


def test_clean_restart_stays_at_zero_divergence(
    session: MarketSession, strategy: FixedWeightStrategy, config: AppConfig
) -> None:
    restart_at = session.groups[0][0].event_time
    options = ReplayOptions(restart_after=restart_at)
    backtest = run_backtest_session(session, strategy, config, options=options)
    shadow = run_shadow_session(session, strategy, config, options=options)
    assert check_parity(backtest, shadow)["match"] is True
    untouched = run_backtest_session(session, strategy, config)
    assert check_parity(backtest, untouched)["match"] is True


def test_tampered_restart_is_state_drift(
    session: MarketSession, strategy: FixedWeightStrategy, config: AppConfig
) -> None:
    def _tamper(state: dict) -> dict:
        shares = dict(state.get("shares") or {})
        shares["A"] = float(shares.get("A", 0.0)) + 1.0
        state["shares"] = shares
        return state

    restart_at = session.groups[0][0].event_time
    shadow = run_shadow_session(
        session,
        strategy,
        config,
        options=ReplayOptions(restart_after=restart_at, state_mutator=_tamper),
    )
    backtest = run_backtest_session(session, strategy, config)
    report = check_parity(backtest, shadow)
    assert report["n_divergent"] > 0
    assert _causes(backtest, shadow) == {"state_drift"}
    assert any(row["detail"] == "after_restart" for row in report["ledger"])
    assert all(row["event_time"] != restart_at for row in report["ledger"])


def test_origin_branch_is_code_path(session: MarketSession, config: AppConfig) -> None:
    def _secret(weight: float) -> float:
        return weight

    class _Branched:
        def decide(self, ctx):
            weight = 0.05
            if ctx.origin == "backtest":
                weight = _secret(weight)
            return TargetDecision(weights={"A": weight, "B": -0.02})

    strategy = _Branched()
    backtest = run_backtest_session(session, strategy, config)
    shadow = run_shadow_session(session, strategy, config)
    assert _causes(backtest, shadow) == {"code_path"}
    built = build_report(backtest, shadow, backtest_strategy=strategy, shadow_strategy=strategy)
    assert built["code_path"]["runtime_trace_equal"] is False
    assert built["code_path"]["same_code_path"] is False
    assert any(name.endswith("._secret") for name in built["code_path"]["runtime_only_backtest"])
    shortfall = built["shortfall"]
    assert shortfall["terminal_gap"] == pytest.approx(0.0, abs=1e-6)


def test_refuses_live_mode(
    session: MarketSession, strategy: FixedWeightStrategy, config: AppConfig
) -> None:
    config.runtime.mode = RuntimeMode.LIVE
    with pytest.raises(RuntimeError, match="refuses live"):
        run_shadow_session(session, strategy, config)


def test_only_the_simulated_broker_is_constructed(
    session: MarketSession, strategy: FixedWeightStrategy, config: AppConfig
) -> None:
    run_shadow_session(session, strategy, config)
    assert SimulatedBroker.__name__ == "SimulatedBroker"


def test_session_rejects_a_bad_tape(tmp_path: Path) -> None:
    del tmp_path
    stamp = make_session(1).bars[0].event_time
    with pytest.raises(ValueError, match="timezone-aware"):
        Bar(
            security_id="A",
            event_time=stamp.replace(tzinfo=None),
            open=1,
            high=1,
            low=1,
            close=1,
            volume=1,
            source="synthetic",
            revision_id="r0",
            available_time=stamp,
        )
    with pytest.raises(ValueError, match="duplicate"):
        MarketSession([make_session(1).bars[0], make_session(1).bars[0]])
    with pytest.raises(ValueError, match="empty"):
        MarketSession([])


def test_from_frame_round_trip(session: MarketSession) -> None:
    restored = MarketSession.from_frame(session.to_frame())
    assert restored.bars == session.bars
    assert restored.synthetic is True
