"""Regression pins for fail-open broker defects found by seeded property tests.

- A duplicate live ``order_id`` silently overwrote the resting order; a later
  bar could then fill the wrong order object.
- A bar whose open sat outside its own [low, high] range produced fills at
  prices the bar never printed.
- ``target_to_orders`` crashed on NaN marks (pydantic raised on the derived
  quantity) while silently skipping non-positive marks, and minted
  liquidations on +inf marks.
"""

from datetime import UTC, datetime

import pytest

from quant_fund.execution.simulated_broker import SimulatedBroker
from quant_fund.schemas.orders import Order, OrderSide, OrderStatus
from tests.property._books import research_config

T0 = datetime(2024, 1, 2, tzinfo=UTC)


def _broker(**kwargs) -> SimulatedBroker:
    return SimulatedBroker(config=research_config(), **kwargs)


def _order(oid: str, side: OrderSide, qty: float, limit: float | None = None) -> Order:
    return Order(
        order_id=oid,
        security_id="A",
        symbol="A",
        side=side,
        quantity=qty,
        signal_time=T0,
        decision_time=T0,
        order_time=T0,
        status=OrderStatus.NEW,
        limit_price=limit,
    )


def test_duplicate_live_order_id_rejected() -> None:
    broker = _broker()
    broker.submit(
        _order("dup", OrderSide.BUY, 10.0, limit=99.0), price=100.0, nav=1e6, adv_dollars=1e9
    )
    with pytest.raises(ValueError, match="already working"):
        broker.submit(
            _order("dup", OrderSide.SELL, 50.0, limit=101.0), price=100.0, nav=1e6, adv_dollars=1e9
        )
    # the original intent is still the working order
    assert broker.open_orders["dup"].side is OrderSide.BUY


def test_malformed_bar_cannot_print_phantom_fill() -> None:
    broker = _broker()
    broker.submit(
        _order("o1", OrderSide.BUY, 1.0, limit=105.0), price=100.0, nav=1e6, adv_dollars=1e9
    )
    with pytest.raises(ValueError, match="bar ordering"):
        broker.process_bar("A", bar_open=110.0, bar_high=100.0, bar_low=90.0, adv_dollars=1e9)
    with pytest.raises(ValueError, match="bar ordering"):
        broker.submit(
            _order("o2", OrderSide.BUY, 1.0, limit=105.0),
            price=100.0,
            nav=1e6,
            adv_dollars=1e9,
            bar_open=90.0,
            bar_high=110.0,
            bar_low=100.0,
        )
    assert not broker.fills


def test_target_to_orders_nan_mark_skips_not_crashes() -> None:
    broker = _broker()
    orders = broker.target_to_orders(
        {"A": 0.5, "B": 0.5},
        {"A": 100.0, "B": float("nan")},
        signal_time=T0,
        order_time=T0,
    )
    assert [o.security_id for o in orders] == ["A"]
    # +inf marks are equally untradeable — no liquidation orders on garbage
    broker.shares["A"] = 5.0
    orders = broker.target_to_orders(
        {"A": 0.5}, {"A": float("inf")}, signal_time=T0, order_time=T0, nav=1e6
    )
    assert orders == []
    # without an explicit nav, the held-name mark is caught by nav() itself
    with pytest.raises(ValueError, match="valuation marks"):
        broker.target_to_orders({"A": 0.5}, {"A": float("inf")}, signal_time=T0, order_time=T0)
