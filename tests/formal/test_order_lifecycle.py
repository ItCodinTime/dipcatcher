"""Executable spec: safety enumeration, hand traces, broker conformance."""

from datetime import UTC, datetime, timedelta

import pytest

from quant_fund.config.loader import load_config
from quant_fund.execution.simulated_broker import SimulatedBroker
from quant_fund.formal.order_lifecycle import check_trace, enumerate_safety
from quant_fund.formal.traces import TraceSession, events_from_broker
from quant_fund.schemas.orders import Order, OrderSide, OrderStatus


def test_python_twin_enumerates_without_safety_violations() -> None:
    states, violations = enumerate_safety(max_qty=2, max_fill_id=2)
    assert not violations
    assert states > 1


def test_hand_traces_for_races_restarts_and_duplicate_fills() -> None:
    filled = check_trace(
        [
            {"op": "submit", "order_id": "A", "qty": 3},
            {"op": "fill", "order_id": "A", "fill_id": "f1", "qty": 3, "rest": False},
        ]
    )
    assert filled.ok
    assert filled.book.orders["A"].status == "filled"

    # Cancel wins the race: no fill is applied.
    cancel_wins = check_trace(
        [
            {"op": "submit", "order_id": "A", "qty": 3},
            {"op": "cancel", "order_id": "A"},
            {"op": "late_fill", "order_id": "A", "fill_id": "late"},
        ]
    )
    assert cancel_wins.ok
    assert cancel_wins.book.orders["A"].status == "canceled"
    assert cancel_wins.book.orders["A"].filled == 0

    # Fill wins the race. A following cancel is disabled.
    fill_then_cancel = check_trace(
        [
            {"op": "submit", "order_id": "A", "qty": 3},
            {"op": "fill", "order_id": "A", "fill_id": "f1", "qty": 3, "rest": False},
            {"op": "cancel", "order_id": "A"},
        ]
    )
    assert not fill_then_cancel.ok
    assert any("cancel from filled" in item for item in fill_then_cancel.violations)

    # Duplicate fill id does not increase filled.
    duplicate = check_trace(
        [
            {"op": "submit", "order_id": "A", "qty": 4},
            {"op": "fill", "order_id": "A", "fill_id": "f1", "qty": 1, "rest": True},
            {"op": "fill", "order_id": "A", "fill_id": "f1", "qty": 99, "rest": False},
        ]
    )
    assert duplicate.ok
    assert duplicate.book.orders["A"].status == "partial"
    assert duplicate.book.orders["A"].filled == pytest.approx(1)

    # Uncommitted fill is lost on crash. Checkpointed fill survives.
    lost = check_trace(
        [
            {"op": "submit", "order_id": "A", "qty": 2},
            {"op": "checkpoint"},
            {"op": "fill", "order_id": "A", "fill_id": "f1", "qty": 2, "rest": False},
            {"op": "crash"},
        ]
    )
    assert lost.ok
    assert lost.book.orders["A"].status == "new"
    assert lost.book.orders["A"].filled == 0

    kept = check_trace(
        [
            {"op": "submit", "order_id": "A", "qty": 2},
            {"op": "fill", "order_id": "A", "fill_id": "f1", "qty": 2, "rest": False},
            {"op": "checkpoint"},
            {"op": "crash"},
        ]
    )
    assert kept.ok
    assert kept.book.orders["A"].status == "filled"
    assert kept.book.orders["A"].filled == pytest.approx(2)

    # IOC close-out: short fill, not resting, still terminal and not overfilled.
    ioc = check_trace(
        [
            {"op": "submit", "order_id": "A", "qty": 10},
            {"op": "fill", "order_id": "A", "fill_id": "f1", "qty": 4, "rest": False},
        ]
    )
    assert ioc.ok
    order = ioc.book.orders["A"]
    assert order.status == "filled"
    assert order.filled == pytest.approx(4)
    assert order.filled <= order.qty

    over = check_trace(
        [
            {"op": "submit", "order_id": "A", "qty": 2},
            {"op": "fill", "order_id": "A", "fill_id": "f1", "qty": 3, "rest": False},
        ]
    )
    assert not over.ok
    assert any("exceed" in item for item in over.violations)


def _cfg(tmp_path):
    cfg = load_config("configs/paper.yaml")
    cfg.data.root = tmp_path
    cfg.costs.frictionless = True
    cfg.costs.participation_limit = 1.0
    cfg.risk_gate.max_name = 1.0
    cfg.risk_gate.max_gross = 3.0
    cfg.risk_gate.max_net = 3.0
    cfg.risk_gate.max_order_notional = 1e12
    cfg.risk_gate.max_participation = 1.0
    return cfg


def _order(oid, qty, when, *, limit=None, expire=None, side=OrderSide.BUY):
    return Order(
        order_id=oid,
        security_id="A",
        symbol="A",
        side=side,
        quantity=qty,
        signal_time=when,
        decision_time=when,
        order_time=when,
        status=OrderStatus.NEW,
        limit_price=limit,
        expire_time=expire,
    )


def _assert_both(session: TraceSession) -> None:
    direct = check_trace(session.events)
    assert direct.ok, direct.violations
    extracted = check_trace(events_from_broker(session.broker))
    assert extracted.ok, extracted.violations
    for oid, order in direct.book.orders.items():
        other = extracted.book.orders[oid]
        assert other.status == order.status
        assert other.filled == pytest.approx(order.filled)
        assert other.qty == pytest.approx(order.qty)


def test_broker_traces_match_the_spec(tmp_path) -> None:
    when = datetime(2024, 1, 2, tzinfo=UTC)
    cfg = _cfg(tmp_path)
    broker = SimulatedBroker(config=cfg, initial_cash=1_000_000.0)
    broker.mark({"A": 100.0})
    session = TraceSession(broker)

    # Market fill.
    session.submit(
        _order("mkt", 5, when),
        price=100.0,
        nav=1_000_000.0,
        adv_dollars=1e12,
        sigma=0.0,
    )
    # Reject: missing price.
    session.submit(
        _order("bad", 1, when),
        price=0.0,
        nav=1_000_000.0,
        adv_dollars=1e12,
        sigma=0.0,
    )
    # Rest, partial, complete.
    cfg.costs.participation_limit = 0.1
    session.submit(
        _order("lim", 10, when, limit=99.0),
        price=100.0,
        nav=broker.nav(),
        adv_dollars=4950.0,
        sigma=0.0,
    )
    assert "lim" in session.broker.open_orders
    session.process_bar(
        "A",
        bar_open=100.0,
        bar_high=101.0,
        bar_low=98.0,
        bar_time=when + timedelta(hours=1),
        adv_dollars=4950.0,
    )
    assert session.broker.open_orders["lim"].status is OrderStatus.PARTIAL
    session.amend("lim", quantity=6.0, limit_price=99.0)
    session.process_bar(
        "A",
        bar_open=99.0,
        bar_high=100.0,
        bar_low=98.0,
        bar_time=when + timedelta(hours=2),
        adv_dollars=1e12,
    )
    assert "lim" not in session.broker.open_orders

    # Rest then cancel.
    cfg.costs.participation_limit = 1.0
    session.submit(
        _order("cxl", 2, when, limit=90.0),
        price=100.0,
        nav=session.broker.nav(),
        adv_dollars=1e12,
        sigma=0.0,
    )
    session.cancel("cxl")

    # Expire wins over a touching bar.
    session.submit(
        _order("exp", 2, when, limit=99.0, expire=when + timedelta(hours=4)),
        price=100.0,
        nav=session.broker.nav(),
        adv_dollars=1e12,
        sigma=0.0,
    )
    session.process_bar(
        "A",
        bar_open=100.0,
        bar_high=101.0,
        bar_low=98.0,
        bar_time=when + timedelta(hours=4),
        adv_dollars=1e12,
    )
    assert "exp" not in session.broker.open_orders

    # Durable restart is a stutter on the spec state.
    before = check_trace(session.events)
    session.restart()
    _assert_both(session)
    after = check_trace(session.events)
    assert before.ok and after.ok
    for oid, order in before.book.orders.items():
        assert after.book.orders[oid].status == order.status
        assert after.book.orders[oid].filled == pytest.approx(order.filled)


def test_ioc_partial_session_keeps_the_parent_quantity(tmp_path) -> None:
    cfg = _cfg(tmp_path)
    cfg.costs.participation_limit = 0.1
    cfg.costs.frictionless = True
    broker = SimulatedBroker(config=cfg, initial_cash=10_000_000.0)
    broker.mark({"A": 100.0})
    session = TraceSession(broker)
    when = datetime(2024, 1, 2, tzinfo=UTC)
    session.submit(
        _order("ioc", 40_000, when),
        price=100.0,
        nav=10_000_000.0,
        adv_dollars=100_000.0,
        sigma=0.0,
    )
    checked = check_trace(session.events)
    assert checked.ok, checked.violations
    order = checked.book.orders["ioc"]
    assert order.status == "filled"
    assert order.qty == pytest.approx(40_000)
    assert order.filled == pytest.approx(100)
    assert order.filled < order.qty


def test_restore_dedupes_fill_id_without_moving_cash(tmp_path) -> None:
    cfg = _cfg(tmp_path)
    broker = SimulatedBroker(config=cfg, initial_cash=100_000.0)
    when = datetime(2024, 1, 2, tzinfo=UTC)
    broker.mark({"A": 100.0})
    session = TraceSession(broker)
    session.submit(
        _order("once", 4, when),
        price=50.0,
        nav=100_000.0,
        adv_dollars=1e12,
        sigma=0.0,
    )
    state = session.broker.to_dict()
    fill_rows = [row for row in state["history"] if row["fill"] is not None]
    state["history"] = list(state["history"]) + [fill_rows[-1]]
    state["n_orders"] = len(state["history"])
    restored = SimulatedBroker.from_state(cfg, state)
    assert restored.cash == pytest.approx(session.broker.cash)
    assert len(restored.fills) == len(session.broker.fills)
    replay = list(session.events) + [
        {
            "op": "duplicate_fill",
            "order_id": "once",
            "fill_id": session.broker.fills[-1].fill_id,
        }
    ]
    checked = check_trace(replay)
    assert checked.ok
    assert checked.book.orders["once"].filled == pytest.approx(4)
