"""Duplicate order-id submission is not idempotent. The spec rejects it."""

from datetime import UTC, datetime

import pytest

from quant_fund.config.loader import load_config
from quant_fund.execution.simulated_broker import SimulatedBroker
from quant_fund.formal.order_lifecycle import check_trace
from quant_fund.schemas.orders import Order, OrderSide, OrderStatus


def test_duplicate_order_id_submit_overfills_and_spec_rejects_it(tmp_path) -> None:
    """A second ``submit`` of the same id applies a second fill.

    The simulated broker does not treat ``order_id`` as an idempotency key.
    Cash and shares move again (10 shares become 20). The lifecycle spec
    rejects the trace because submit is only enabled from ``absent``.

    The broker submission path is intentionally unchanged: paper order ids
    are fresh UUIDs, and gating duplicates would be a submission-policy
    change. This test is the counterexample.
    """
    cfg = load_config("configs/paper.yaml")
    cfg.data.root = tmp_path
    cfg.costs.frictionless = True
    cfg.costs.participation_limit = 1.0
    cfg.risk_gate.max_name = 1.0
    cfg.risk_gate.max_gross = 2.0
    cfg.risk_gate.max_net = 2.0
    cfg.risk_gate.max_order_notional = 1e12
    cfg.risk_gate.max_participation = 1.0
    broker = SimulatedBroker(config=cfg, initial_cash=100_000.0)
    when = datetime(2024, 1, 2, tzinfo=UTC)
    broker.mark({"A": 100.0})
    order = Order(
        order_id="same-id",
        security_id="A",
        symbol="A",
        side=OrderSide.BUY,
        quantity=10.0,
        signal_time=when,
        decision_time=when,
        order_time=when,
        status=OrderStatus.NEW,
    )
    first = broker.submit(order, price=100.0, nav=100_000.0, adv_dollars=1e12, sigma=0.0)
    second = broker.submit(order, price=100.0, nav=broker.nav(), adv_dollars=1e12, sigma=0.0)
    assert first.fill is not None and second.fill is not None
    assert first.fill.fill_id != second.fill.fill_id
    assert broker.shares["A"] == pytest.approx(20.0)
    assert broker.cash == pytest.approx(98_000.0)

    trace = [
        {"op": "submit", "order_id": "same-id", "qty": 10.0},
        {
            "op": "fill",
            "order_id": "same-id",
            "fill_id": first.fill.fill_id,
            "qty": 10.0,
            "rest": False,
        },
        {"op": "submit", "order_id": "same-id", "qty": 10.0},
        {
            "op": "fill",
            "order_id": "same-id",
            "fill_id": second.fill.fill_id,
            "qty": 10.0,
            "rest": False,
        },
    ]
    result = check_trace(trace)
    assert not result.ok
    assert any("submit from filled" in item for item in result.violations)
