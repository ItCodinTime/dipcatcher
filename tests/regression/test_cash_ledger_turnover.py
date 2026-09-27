"""Cash-ledger fees must include turnover bps the broker already deducts."""

from datetime import UTC, datetime

import pytest

from quant_fund.config.loader import load_config
from quant_fund.execution.costs import total_cost
from quant_fund.execution.simulated_broker import SimulatedBroker
from quant_fund.paper.ledger import PaperLedger
from quant_fund.schemas.orders import Order, OrderSide, OrderStatus


def _cfg(tmp_path):
    cfg = load_config("configs/paper.yaml")
    cfg.data.root = tmp_path
    cfg.costs.frictionless = False
    cfg.costs.commission_bps = 1.0
    cfg.costs.half_spread_bps = 5.0
    cfg.costs.impact_y = 0.0
    cfg.costs.bps_per_turnover = 10.0
    cfg.costs.participation_limit = 1.0
    cfg.risk_gate.max_name = 1.0
    cfg.risk_gate.max_gross = 2.0
    cfg.risk_gate.max_net = 2.0
    cfg.risk_gate.max_order_notional = 1e12
    cfg.risk_gate.max_participation = 1.0
    return cfg


def _order(oid: str, side: OrderSide, qty: float, when: datetime) -> Order:
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
    )


def test_cash_ledger_fees_include_turnover_bps(tmp_path) -> None:
    """10 bps of turnover on a $1000 buy is $1.

    Before the fix the ledger summed fee+spread+impact ($0.60) and the
    broker cash drop was $1.60. Default configs leave bps_per_turnover at 0,
    so this does not move those runs.
    """
    cfg = _cfg(tmp_path)
    broker = SimulatedBroker(config=cfg, initial_cash=100_000.0)
    when = datetime(2024, 1, 2, tzinfo=UTC)
    broker.mark({"A": 100.0})
    record = broker.submit(
        _order("buy-1", OrderSide.BUY, 10.0, when),
        price=100.0,
        nav=100_000.0,
        adv_dollars=1e12,
        sigma=0.0,
        decision_price=99.0,
    )
    assert record.fill is not None
    costs = total_cost(10.0, 100.0, 1e12, 0.0, cfg.costs)
    assert costs["turnover_bps"] == pytest.approx(1.0)
    assert costs["total"] == pytest.approx(1.6)
    # Adverse drift is recorded and is not a cash charge.
    assert record.fill.slippage == pytest.approx(10.0)
    assert record.fill.turnover_cost == pytest.approx(1.0)
    assert broker.cash == pytest.approx(100_000.0 - 1_000.0 - 1.6)

    ledger = PaperLedger(tmp_path, "turnover-bps")
    ledger.record_orders([record], when)
    event = ledger._cash_events[0]
    assert event["fees"] == pytest.approx(1.6)
    assert event["cash_delta"] == pytest.approx(-(1_000.0 + 1.6))

    sell = broker.submit(
        _order("sell-1", OrderSide.SELL, 10.0, when),
        price=100.0,
        nav=broker.nav(),
        adv_dollars=1e12,
        sigma=0.0,
    )
    assert sell.fill is not None
    ledger.record_orders([sell], when)
    cash_from_ledger = 100_000.0 + sum(row["cash_delta"] for row in ledger._cash_events)
    assert cash_from_ledger == pytest.approx(broker.cash)
