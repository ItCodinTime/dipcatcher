"""Pre-trade checks, latch, and audit trail."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from quant_fund.pretrade.codes import (
    BUYING_POWER,
    COLLAR,
    CONCENTRATION,
    DAILY_LOSS,
    DUPLICATE,
    GFV,
    GROSS,
    HALTED,
    INTERNAL,
    KILL,
    KIND_CANCEL,
    MARKET_CLOSED,
    MAX_NOTIONAL,
    MAX_QTY,
    MESSAGE_RATE,
    NET,
    NON_FINITE,
    NON_MONOTONIC,
    ORDER_RATE,
    PDT,
    POSITION_NOTIONAL,
    POSITION_QTY,
    REG_SHO,
    STALE,
    STATE_FULL,
    TRAILING_DD,
    UNKNOWN_SYMBOL,
    decision_allowed,
    reason_names,
)
from quant_fund.pretrade.config import PretradeConfig, SessionConfig
from quant_fund.pretrade.engine import PretradeEngine
from quant_fund.pretrade.kernel import hot_check
from tests.unit.pretrade.support import HMAC_KEY, TS_NS, arm, limit_config, make_engine, order


def test_in_limit_order_is_allowed() -> None:
    engine = make_engine()
    sid = arm(engine)
    assert engine.check(order(sid), apply=False) == 0
    assert engine.last_allowed is True
    decision = engine.decide(order(sid, qty=11, ts_ns=TS_NS + 1))
    assert decision.allowed is True
    assert decision.reasons == ()
    assert decision.config_sha256 == engine.config_sha256


def test_hard_ceilings_are_strict_and_inclusive_at_the_boundary() -> None:
    engine = make_engine(max_order_quantity=5, max_order_notional=500)
    sid = arm(engine, ref=100)
    at_qty = engine.check(order(sid, qty=5, px=100), apply=False)
    assert at_qty & MAX_QTY == 0
    over_qty = engine.check(order(sid, qty=5.1, px=100, ts_ns=TS_NS + 1), apply=False)
    assert over_qty & MAX_QTY
    assert decision_allowed(over_qty, 1) is False
    # 6 * 100 = 600 > 500, and quantity 6 > 5. Both bits are reported together.
    both = engine.check(order(sid, qty=6, px=100, ts_ns=TS_NS + 2), apply=False)
    assert both & MAX_QTY
    assert both & MAX_NOTIONAL


def test_price_collar_on_limits_only() -> None:
    engine = make_engine(price_collar_bps=100)
    sid = arm(engine, ref=100)
    assert engine.check(order(sid, px=101, is_limit=1), apply=False) & COLLAR == 0
    breached = engine.check(order(sid, px=102, ts_ns=TS_NS + 1), apply=False)
    assert breached & COLLAR
    market = engine.check(order(sid, px=50, ts_ns=TS_NS + 2, is_limit=0, qty=1), apply=False)
    assert market & COLLAR == 0


def test_position_gross_net_and_concentration() -> None:
    engine = make_engine(
        max_position_quantity=10,
        max_position_notional=1_500,
        max_gross_notional=1_500,
        max_net_notional=1_000,
        max_name_concentration=0.25,
        nav=1_000,
    )
    sid = arm(engine, ref=100, pos=8)
    assert engine.book.name_limit == pytest.approx(250)
    qty = engine.check(order(sid, qty=3, px=100), apply=False)
    assert qty & POSITION_QTY
    notion = engine.check(order(sid, qty=8, px=100, ts_ns=TS_NS + 1), apply=False)
    assert notion & POSITION_NOTIONAL
    # 8*100 = 800 now; buy 8 -> 1600 gross and net, above both ceilings.
    exposure = engine.check(order(sid, qty=8, px=100, ts_ns=TS_NS + 2), apply=False)
    assert exposure & GROSS
    assert exposure & NET
    # Flat book, buy 3 shares = 300 > 25% of 1000.
    fresh = make_engine(max_name_concentration=0.25, nav=1_000, cash=1_000)
    sid2 = arm(fresh, ref=100, pos=0)
    conc = fresh.check(order(sid2, qty=3, px=100), apply=False)
    assert conc & CONCENTRATION
    assert decision_allowed(conc, 1) is False


def test_stale_and_non_finite_fail_closed_without_latching() -> None:
    engine = make_engine(max_reference_age_ns=1_000, max_mark_age_ns=1_000)
    sid = arm(engine, ts_ns=TS_NS)
    stale = engine.check(order(sid, ts_ns=TS_NS + 5_000), apply=False)
    assert stale & STALE
    assert engine.book.killed == 0
    bad = engine.check(order(sid, qty=float("nan"), ts_ns=TS_NS + 1), apply=False)
    assert bad == NON_FINITE
    assert engine.book.killed == 0
    assert "non_finite" in reason_names(bad)


def test_market_hours_and_halt() -> None:
    saturday = int(
        datetime(2024, 1, 6, 15, 0, tzinfo=ZoneInfo("America/New_York")).timestamp() * 1_000_000_000
    )
    closed = make_engine(ts_ns=saturday)
    sid = arm(closed, ts_ns=saturday)
    assert closed.check(order(sid, ts_ns=saturday), apply=False) & MARKET_CLOSED

    early = int(
        datetime(2024, 1, 3, 8, 0, tzinfo=ZoneInfo("America/New_York")).timestamp() * 1_000_000_000
    )
    pre = make_engine(ts_ns=early)
    sid_pre = arm(pre, ts_ns=early)
    assert pre.check(order(sid_pre, ts_ns=early), apply=False) & MARKET_CLOSED
    pre.set_symbol(sid_pre, pos=0.0, ref_px=100.0, ref_ts_ns=pre.book.open_ns)
    pre.update_account(nav=1_000_000.0, mark_ts_ns=pre.book.open_ns)
    opened = pre.check(order(sid_pre, ts_ns=pre.book.open_ns, qty=2), apply=False)
    assert opened & MARKET_CLOSED == 0
    assert opened == 0
    assert pre.check(order(sid_pre, ts_ns=pre.book.close_ns, qty=3), apply=False) & MARKET_CLOSED

    halted = make_engine()
    sid_h = arm(halted, halted=True)
    assert halted.check(order(sid_h), apply=False) & HALTED
    cancel = halted.check(order(sid_h, kind=KIND_CANCEL, ts_ns=TS_NS + 1, is_limit=0), apply=False)
    assert decision_allowed(cancel, KIND_CANCEL) is True


def test_duplicate_rate_and_message_throttles() -> None:
    engine = make_engine(max_orders_per_window=3, max_messages_per_window=6)
    sid = arm(engine)
    assert engine.check(order(sid, qty=1), apply=False) == 0
    assert engine.check(order(sid, qty=1, ts_ns=TS_NS + 1), apply=False) & DUPLICATE
    assert (
        engine.check(order(sid, qty=1, ts_ns=TS_NS + 2_000_000_000), apply=False) & DUPLICATE == 0
    )

    rated = make_engine(max_orders_per_window=3, duplicate_window_ns=1)
    sid_r = arm(rated)
    for i in range(3):
        assert rated.check(order(sid_r, qty=i + 1, ts_ns=TS_NS + i), apply=False) == 0
    assert rated.check(order(sid_r, qty=4, ts_ns=TS_NS + 3), apply=False) & ORDER_RATE

    messages = make_engine(max_messages_per_window=2, max_orders_per_window=10)
    sid_m = arm(messages)
    assert (
        messages.check(order(sid_m, kind=KIND_CANCEL, is_limit=0, ts_ns=TS_NS), apply=False)
        & MESSAGE_RATE
        == 0
    )
    assert (
        messages.check(order(sid_m, kind=KIND_CANCEL, is_limit=0, ts_ns=TS_NS + 1), apply=False)
        & MESSAGE_RATE
        == 0
    )
    third = messages.check(order(sid_m, kind=KIND_CANCEL, is_limit=0, ts_ns=TS_NS + 2), apply=False)
    assert third & MESSAGE_RATE
    assert decision_allowed(third, KIND_CANCEL) is False


def test_non_monotonic_time_is_denied() -> None:
    engine = make_engine()
    sid = arm(engine)
    assert engine.check(order(sid, ts_ns=TS_NS + 1_000), apply=False) == 0
    back = engine.check(order(sid, qty=2, ts_ns=TS_NS + 10), apply=False)
    assert back & NON_MONOTONIC
    assert engine.book.killed == 0


def test_unknown_symbol_is_denied() -> None:
    engine = make_engine()
    arm(engine)
    bits = engine.check(order(5), apply=False)
    assert bits & UNKNOWN_SYMBOL


def test_cash_account_reg_sho_and_rule_201() -> None:
    cash = make_engine()
    sid = arm(cash, pos=0)
    assert cash.check(order(sid, side=-1, qty=1), apply=False) & REG_SHO
    long_book = make_engine()
    sid_long = arm(long_book, pos=10)
    assert long_book.check(order(sid_long, side=-1, qty=1, ts_ns=TS_NS), apply=False) & REG_SHO == 0

    margin = make_engine(cash_account=False, restrict_to_settled_cash=False)
    blocked = arm(margin, pos=0, locate_ok=False)
    assert margin.check(order(blocked, side=-1, qty=1), apply=False) & REG_SHO

    rule = make_engine(cash_account=False, restrict_to_settled_cash=False)
    sid_r = arm(rule, pos=0, sho_restricted=True, locate_ok=True, bid=100)
    assert rule.check(order(sid_r, side=-1, qty=1, px=99), apply=False) & REG_SHO
    assert (
        rule.check(order(sid_r, side=-1, qty=2, px=100, ts_ns=TS_NS + 1, is_limit=0), apply=False)
        & REG_SHO
    )
    allowed = rule.check(order(sid_r, side=-1, qty=1, px=100, ts_ns=TS_NS + 2), apply=False)
    assert allowed & REG_SHO == 0
    assert allowed == 0


def test_buying_power_and_good_faith_violation() -> None:
    poor = make_engine(cash=50)
    sid = arm(poor)
    assert poor.check(order(sid, qty=1, px=100), apply=False) & BUYING_POWER

    engine = make_engine(
        cash=0,
        restrict_to_settled_cash=False,
        settlement_ns=1_000_000_000,
        cash_account=True,
    )
    sid_g = arm(engine, ref=10)
    assert engine.check(order(sid_g, qty=1, px=10), apply=True) == 0
    assert engine.book.syms[sid_g].lu[0] == 1
    sell = engine.check(order(sid_g, side=-1, qty=1, px=10, ts_ns=TS_NS + 10), apply=False)
    assert sell & GFV
    assert engine.book.killed == 0


def test_pattern_day_trader_blocks_the_fourth_round_trip_only() -> None:
    engine = make_engine(
        nav=10_000,
        cash=10_000,
        pdt_equity_threshold=25_000,
        pdt_max_day_trades=3,
        settlement_ns=0,
        max_name_concentration=1,
        duplicate_window_ns=1,
    )
    sid = arm(engine, ref=10, ts_ns=TS_NS)
    stamp = TS_NS
    for _ in range(3):
        buy = engine.check(order(sid, qty=1, px=10, ts_ns=stamp, session_id=10), apply=True)
        assert buy == 0, reason_names(buy)
        stamp += 1
        sell = engine.check(
            order(sid, side=-1, qty=1, px=10, ts_ns=stamp, session_id=10), apply=True
        )
        assert sell == 0, reason_names(sell)
        stamp += 1
    assert engine.book.dt_n == 3
    assert engine.check(order(sid, qty=1, px=10, ts_ns=stamp, session_id=10), apply=True) == 0
    stamp += 1
    blocked = engine.check(
        order(sid, side=-1, qty=1, px=10, ts_ns=stamp, session_id=10), apply=False
    )
    assert blocked & PDT
    assert engine.book.killed == 0
    # A later weekday session falls outside the rolling window.
    later = engine.check(
        order(sid, side=-1, qty=1, px=10, ts_ns=stamp + 2, session_id=15), apply=False
    )
    assert later & PDT == 0


def test_daily_loss_and_trailing_drawdown_latch_until_manual_reset() -> None:
    engine = make_engine(
        nav=1_000,
        cash=1_000,
        max_daily_loss_fraction=0.1,
        max_trailing_drawdown_fraction=0.5,
    )
    sid = arm(engine)
    engine.update_account(nav=800, mark_ts_ns=TS_NS, session_start_nav=1_000, peak_nav=1_000)
    tripped = engine.check(order(sid), apply=False)
    assert tripped & DAILY_LOSS
    assert tripped & KILL
    assert engine.book.killed == 1
    assert engine.audit
    engine.update_account(nav=1_000, mark_ts_ns=TS_NS + 5, session_start_nav=1_000, peak_nav=1_000)
    still = engine.check(order(sid, qty=2, ts_ns=TS_NS + 5), apply=False)
    assert still & KILL
    assert decision_allowed(still, 1) is False
    with pytest.raises(ValueError, match="actor"):
        engine.reset(" ", "reviewed", TS_NS + 6)
    assert engine.book.killed == 1
    engine.reset("alice", "desk reviewed the breach", TS_NS + 6)
    assert engine.book.killed == 0
    assert engine.check(order(sid, qty=3, ts_ns=TS_NS + 6), apply=False) == 0

    draw = make_engine(
        nav=1_000,
        cash=1_000,
        max_daily_loss_fraction=0.5,
        max_trailing_drawdown_fraction=0.1,
    )
    sid_d = arm(draw)
    draw.update_account(nav=850, mark_ts_ns=TS_NS, session_start_nav=1_000, peak_nav=1_000)
    bits = draw.check(order(sid_d, qty=1), apply=False)
    assert bits & TRAILING_DD
    assert bits & DAILY_LOSS == 0
    assert draw.book.killed == 1
    draw.update_account(nav=1_000, mark_ts_ns=TS_NS + 1, session_start_nav=1_000, peak_nav=1_000)
    assert draw.check(order(sid_d, qty=2, ts_ns=TS_NS + 1), apply=False) & KILL


def test_manual_trip_allows_cancels_until_an_internal_fault() -> None:
    engine = make_engine()
    sid = arm(engine)
    engine.trip("ops", "maintenance", TS_NS)
    assert engine.check(order(sid, ts_ns=TS_NS + 1), apply=False) & KILL
    cancel = engine.check(order(sid, kind=KIND_CANCEL, is_limit=0, ts_ns=TS_NS + 2), apply=False)
    assert decision_allowed(cancel, KIND_CANCEL) is True
    with pytest.raises(ValueError):
        engine.trip("", "no actor", TS_NS + 3)


def test_internal_error_latches_and_blocks_cancels(monkeypatch: pytest.MonkeyPatch) -> None:
    engine = make_engine()
    sid = arm(engine)

    def boom(*_args: object, **_kwargs: object) -> int:
        raise RuntimeError("kernel failed")

    monkeypatch.setattr("quant_fund.pretrade.engine.hot_check", boom)
    bits = engine.check(order(sid), apply=False)
    assert bits & KILL
    assert engine.book.kill_reason & (1 << 21)
    monkeypatch.undo()
    cancel = engine.check(order(sid, kind=KIND_CANCEL, is_limit=0, ts_ns=TS_NS + 1), apply=False)
    assert decision_allowed(cancel, KIND_CANCEL) is False
    engine.reset("ops", "new engine state reviewed", TS_NS + 2)
    assert engine.check(order(sid, qty=2, ts_ns=TS_NS + 2), apply=False) == 0


def test_audit_trail_is_hash_chained() -> None:
    engine = make_engine(nav=1_000, cash=1_000, max_daily_loss_fraction=0.1)
    sid = arm(engine)
    engine.update_account(nav=100, mark_ts_ns=TS_NS, session_start_nav=1_000, peak_nav=1_000)
    engine.check(order(sid), apply=False)
    engine.reset("alice", "cleared", TS_NS + 1)
    assert len(engine.audit) >= 2
    previous = "0" * 64
    for event in engine.audit:
        body = {key: value for key, value in event.items() if key != "event_hash"}
        blob = json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
        assert hashlib.sha256(blob).hexdigest() == event["event_hash"]
        assert event["prev_hash"] == previous
        assert event["config_sha256"] == engine.config_sha256
        previous = str(event["event_hash"])


def test_restore_lot_and_bad_fill_fail_closed() -> None:
    engine = make_engine(max_lots_per_symbol=2, settlement_ns=0)
    sid = arm(engine)
    engine.restore_lot(sid, qty=1.0, settles_ns=TS_NS, unsettled_funded=True)
    engine.restore_lot(sid, qty=1.0, settles_ns=TS_NS, unsettled_funded=False)
    with pytest.raises(RuntimeError, match="lot table"):
        engine.restore_lot(sid, qty=1.0, settles_ns=TS_NS, unsettled_funded=False)
    with pytest.raises(ValueError, match="finite"):
        engine.restore_lot(sid, qty=float("nan"), settles_ns=TS_NS, unsettled_funded=False)
    fresh = make_engine()
    fresh_sid = arm(fresh)
    fresh.note_fill(
        symbol_id=fresh_sid,
        side=1,
        qty=float("nan"),
        px=10.0,
        fee=0.0,
        ts_ns=TS_NS,
        session_id=10,
        pos_before=0.0,
    )
    assert fresh.book.killed == 1
    assert fresh.book.kill_reason & INTERNAL


def test_lot_table_fail_closed() -> None:
    engine = make_engine(max_lots_per_symbol=2, duplicate_window_ns=1)
    sid = arm(engine, ref=10)
    assert engine.check(order(sid, qty=1, px=10, ts_ns=TS_NS), apply=True) == 0
    assert engine.check(order(sid, qty=2, px=10, ts_ns=TS_NS + 1), apply=True) == 0
    third = engine.check(order(sid, qty=3, px=10, ts_ns=TS_NS + 2), apply=True)
    assert third & STATE_FULL
    assert engine.book.syms[sid].nlot == 2


def test_same_inputs_are_deterministic() -> None:
    def run() -> list[int]:
        engine = make_engine()
        sid = arm(engine)
        return [
            engine.check(order(sid, qty=index + 1, ts_ns=TS_NS + index), apply=True)
            for index in range(4)
        ]

    assert run() == run()


def test_hot_check_symbol_is_the_engine_symbol() -> None:
    engine = make_engine()
    sid = arm(engine)
    view = order(sid)
    bits = hot_check(
        engine.book,
        engine.book.syms[sid],
        side=view.side,
        qty=view.qty,
        px=view.px,
        ts=view.ts_ns,
        session_id=view.session_id,
        is_limit=view.is_limit,
        kind=view.kind,
        symbol_id=sid,
    )
    assert bits == 0


def test_initial_nav_and_cash_reject_infinity() -> None:
    with pytest.raises(ValueError, match="finite"):
        make_engine(cash=float("inf"))
    with pytest.raises(ValueError, match="finite"):
        make_engine(nav=float("inf"))
    with pytest.raises(ValueError, match="finite"):
        make_engine(cash=float("nan"))


def test_set_symbol_rejects_non_finite_position() -> None:
    engine = make_engine()
    sid = engine.ensure_symbol("A")
    with pytest.raises(ValueError, match="finite"):
        engine.set_symbol(sid, pos=float("nan"), ref_px=100.0, ref_ts_ns=TS_NS)
    with pytest.raises(ValueError, match="finite"):
        engine.set_symbol(sid, pos=float("inf"), ref_px=100.0, ref_ts_ns=TS_NS)


def test_future_dated_reference_and_mark_are_stale() -> None:
    engine = make_engine()
    sid = arm(engine, ts_ns=TS_NS + 60_000_000_000)
    assert engine.check(order(sid), apply=False) & STALE

    marked = make_engine()
    sid_m = arm(marked)
    marked.update_account(nav=1_000_000.0, mark_ts_ns=TS_NS + 60_000_000_000)
    assert marked.check(order(sid_m), apply=False) & STALE


def test_backward_session_roll_does_not_rebase_loss_floors() -> None:
    engine = make_engine(nav=1_000, cash=1_000, max_daily_loss_fraction=0.1)
    sid = arm(engine)
    jan4 = int(
        datetime(2024, 1, 4, 15, 0, tzinfo=ZoneInfo("America/New_York")).timestamp() * 1_000_000_000
    )
    engine.update_account(nav=1_100, mark_ts_ns=jan4)
    engine.set_symbol(sid, pos=0.0, ref_px=100.0, ref_ts_ns=jan4)
    assert engine.check(order(sid, ts_ns=jan4), apply=False) == 0
    assert engine.book.session_start == 1_100
    engine.update_account(nav=1_000, mark_ts_ns=jan4 + 1)
    engine.set_symbol(sid, pos=0.0, ref_px=100.0, ref_ts_ns=TS_NS)
    back = engine.check(order(sid, ts_ns=TS_NS), apply=False)
    assert back & NON_MONOTONIC
    assert engine.book.session_start == 1_100
    assert engine.book.daily_floor == pytest.approx(990.0)


def test_midnight_close_wraps_to_the_next_day() -> None:
    config = PretradeConfig(
        schema_version=1,
        limits=limit_config(),
        session=SessionConfig(timezone="America/New_York", open_minute=570, close_minute=1440),
    )
    engine = PretradeEngine(
        config, hmac_key=HMAC_KEY, initial_nav=1_000, initial_cash=1_000, ts_ns=TS_NS
    )
    sid = arm(engine)
    span = engine.book.close_ns - engine.book.open_ns
    assert span == (1440 - 570) * 60 * 1_000_000_000
    assert engine.check(order(sid), apply=False) == 0


def test_rule_201_denies_a_short_when_the_bid_is_missing() -> None:
    margin = make_engine(cash_account=False, restrict_to_settled_cash=False)
    sid = arm(margin, pos=0, sho_restricted=True, locate_ok=True, bid=0.0)
    assert margin.check(order(sid, side=-1, qty=1, px=100), apply=False) & REG_SHO
    negative = make_engine(cash_account=False, restrict_to_settled_cash=False)
    sid_n = arm(negative, pos=0, sho_restricted=True, locate_ok=True, bid=-1.0)
    assert negative.check(order(sid_n, side=-1, qty=1, px=100), apply=False) & REG_SHO


def test_note_fill_rejects_degenerate_fields() -> None:
    bad = (
        {"qty": float("inf")},
        {"px": float("inf")},
        {"fee": float("nan")},
        {"ts_ns": -1},
        {"pos_before": float("nan")},
        {"side": 0},
    )
    for override in bad:
        engine = make_engine()
        sid = arm(engine)
        fields: dict[str, object] = {
            "symbol_id": sid,
            "side": 1,
            "qty": 1.0,
            "px": 10.0,
            "fee": 0.0,
            "ts_ns": TS_NS,
            "session_id": 10,
            "pos_before": 0.0,
        }
        fields.update(override)
        engine.note_fill(**fields)  # type: ignore[arg-type]
        assert engine.book.killed == 1, override
        assert engine.book.kill_reason & INTERNAL


def test_unbounded_sell_proceeds_fail_closed() -> None:
    engine = make_engine()
    sid = arm(engine)
    engine.note_fill(
        symbol_id=sid,
        side=-1,
        qty=1e308,
        px=1e308,
        fee=0.0,
        ts_ns=TS_NS,
        session_id=10,
        pos_before=0.0,
    )
    assert engine.book.killed == 1
    assert engine.book.settled < float("inf")


def test_pending_trip_is_audited_when_the_kernel_is_driven_directly() -> None:
    engine = make_engine(nav=1_000, cash=1_000, max_daily_loss_fraction=0.1)
    sid = arm(engine)
    engine.update_account(nav=500.0, mark_ts_ns=TS_NS)
    sym = engine.book.syms[sid]
    bits = hot_check(
        engine.book,
        sym,
        side=1,
        qty=1.0,
        px=100.0,
        ts=TS_NS,
        session_id=10,
        is_limit=1,
        kind=1,
        symbol_id=sid,
    )
    assert bits & DAILY_LOSS
    assert engine.book.killed == 1
    engine.reset("ops", "reviewed the breach", TS_NS + 1)
    assert any(
        event["action"] == "trip" and event["reason_bits"] & DAILY_LOSS for event in engine.audit
    )
