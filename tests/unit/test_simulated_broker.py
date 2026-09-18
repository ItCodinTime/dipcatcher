"""Simulated broker order lifecycle, kill switch, risk gate."""

from datetime import UTC, datetime

import pytest

from quant_fund.config.loader import load_config
from quant_fund.execution.simulated_broker import RejectReason, SimulatedBroker
from quant_fund.monitoring.kill_switch import HALT_NEW_ORDERS
from quant_fund.schemas.orders import Order, OrderSide, OrderStatus


def _cfg(tmp_path):
    cfg = load_config("configs/paper.yaml")
    cfg.data.root = tmp_path
    cfg.risk_gate.max_name = 1.0
    cfg.risk_gate.max_net = 1.0
    cfg.risk_gate.max_gross = 2.0
    cfg.risk_gate.max_order_notional = 1e12
    cfg.risk_gate.max_participation = 1.0
    return cfg


def _order(side=OrderSide.BUY, qty=10.0, oid="o1"):
    t = datetime(2024, 1, 2, tzinfo=UTC)
    return Order(
        order_id=oid,
        security_id="A",
        symbol="A",
        side=side,
        quantity=qty,
        signal_time=t,
        decision_time=t,
        order_time=t,
        status=OrderStatus.NEW,
    )


def test_target_rotation_sells_before_buys_to_reuse_cash(tmp_path):
    cfg = _cfg(tmp_path)
    broker = SimulatedBroker(config=cfg, initial_cash=10.0)
    broker.shares["Z"] = 10.0
    prices = {"A": 100.0, "Z": 100.0}
    t = datetime(2024, 1, 2, tzinfo=UTC)

    orders = broker.target_to_orders(
        {"A": 1.0, "Z": 0.0}, prices, signal_time=t, order_time=t, nav=1000.0
    )

    assert [order.side for order in orders] == [OrderSide.SELL, OrderSide.BUY]
    records = [
        broker.submit(order, price=prices[order.security_id], nav=1000.0, adv_dollars=1e9)
        for order in orders
    ]
    assert all(record.order.status is OrderStatus.FILLED for record in records)
    assert broker.shares["A"] == pytest.approx(10.0)
    assert broker.shares["Z"] == pytest.approx(0.0)


def test_order_lifecycle_fill(tmp_path):
    cfg = _cfg(tmp_path)
    broker = SimulatedBroker(config=cfg, initial_cash=100_000.0)
    broker.mark({"A": 100.0})
    rec = broker.submit(_order(), price=100.0, nav=100_000.0, adv_dollars=1e9, sigma=0.02)
    assert rec.order.status is OrderStatus.FILLED
    assert rec.fill is not None
    assert broker.shares["A"] == pytest.approx(10.0)
    assert broker.cash < 100_000.0
    assert len(broker.fills) == 1


def test_kill_switch_halts_new_orders(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.kill_switch.state = HALT_NEW_ORDERS
    broker = SimulatedBroker(config=cfg, initial_cash=100_000.0)
    rec = broker.submit(_order(), price=100.0, nav=100_000.0, adv_dollars=1e9)
    assert rec.order.status is OrderStatus.REJECTED
    assert rec.reject_reason == RejectReason.KILL_SWITCH.value
    assert broker.halt_count == 1
    assert broker.shares.get("A", 0.0) == 0.0


def test_submit_caps_participation_before_risk_gate(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.costs.participation_limit = 0.1
    cfg.risk_gate.max_participation = 0.25
    broker = SimulatedBroker(config=cfg, initial_cash=10_000_000.0)

    rec = broker.submit(
        _order(qty=40_000.0),
        price=100.0,
        nav=10_000_000.0,
        adv_dollars=100_000.0,
        sigma=0.02,
    )

    assert rec.order.status is OrderStatus.FILLED
    assert rec.fill is not None
    assert rec.fill.quantity == pytest.approx(100.0)
    assert rec.fill.is_partial is True


def test_risk_gate_reject(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.risk_gate.max_order_notional = 100.0  # tiny
    broker = SimulatedBroker(config=cfg, initial_cash=100_000.0)
    rec = broker.submit(_order(qty=1000.0), price=100.0, nav=100_000.0, adv_dollars=1e9)
    assert rec.order.status is OrderStatus.REJECTED
    assert rec.reject_reason is not None
    assert rec.reject_reason.startswith(RejectReason.RISK_GATE.value)
    assert broker.reject_count == 1


def test_shadow_no_capital(tmp_path):
    cfg = _cfg(tmp_path)
    shadow = SimulatedBroker(config=cfg, initial_cash=0.0, slot="shadow", allow_capital=False)
    rec = shadow.submit(_order(), price=100.0, nav=1.0, adv_dollars=1e9)
    assert rec.order.status is OrderStatus.ACKED
    assert rec.fill is None
    assert shadow.cash == 0.0
    assert shadow.shares.get("A", 0.0) == 0.0


def test_invalid_liquidity_and_nav_rejected_before_fill(tmp_path):
    cfg = _cfg(tmp_path)
    broker = SimulatedBroker(config=cfg, initial_cash=100_000.0)
    for kwargs in ({"adv_dollars": 0.0}, {"adv_dollars": float("nan")}, {"sigma": -0.1}):
        rec = broker.submit(
            _order(oid=f"bad-{len(broker.history)}"), price=100.0, nav=100_000.0, **kwargs
        )
        assert rec.order.status is OrderStatus.REJECTED
        assert rec.reject_reason == RejectReason.INVALID_MARKET_DATA.value
        assert rec.fill is None
    rec = broker.submit(_order(oid="bad-nav"), price=100.0, nav=0.0, adv_dollars=1e9)
    assert rec.order.status is OrderStatus.REJECTED
    assert rec.reject_reason == RejectReason.INVALID_MARKET_DATA.value


def test_negative_order_quantity_cannot_invert_side(tmp_path):
    cfg = _cfg(tmp_path)
    broker = SimulatedBroker(config=cfg, initial_cash=100_000.0)
    invalid = _order().model_copy(update={"quantity": -10.0})
    rec = broker.submit(invalid, price=100.0, nav=100_000.0, adv_dollars=1e9)
    assert rec.order.status is OrderStatus.REJECTED
    assert rec.reject_reason == RejectReason.ZERO_QTY.value
    assert broker.shares.get("A", 0.0) == 0.0


@pytest.mark.parametrize("price", [0.0, -1.0, float("nan"), float("inf")])
def test_mark_rejects_invalid_market_marks(tmp_path, price):
    broker = SimulatedBroker(config=_cfg(tmp_path), initial_cash=100_000.0)
    with pytest.raises(ValueError, match="market marks"):
        broker.mark({"A": price})


def test_from_state_rejects_nonfinite_cash_and_marks(tmp_path):
    cfg = _cfg(tmp_path)
    with pytest.raises(ValueError, match="cash"):
        SimulatedBroker.from_state(cfg, {"cash": float("nan")})
    with pytest.raises(ValueError, match="market marks"):
        SimulatedBroker.from_state(cfg, {"cash": 1.0, "last_marks": {"A": -1.0}})


def test_from_state_rejects_invalid_cash_and_counters(tmp_path):
    cfg = _cfg(tmp_path)
    with pytest.raises(ValueError, match="initial cash"):
        SimulatedBroker.from_state(cfg, {"initial_cash": -1.0})
    with pytest.raises(ValueError, match="counters"):
        SimulatedBroker.from_state(cfg, {"cash": 1.0, "reject_count": -1})


@pytest.mark.parametrize("kill_state", [None, "NOT_A_KILL_STATE"])
def test_from_state_rejects_invalid_runtime_kill_switch_state(tmp_path, kill_state):
    cfg = _cfg(tmp_path)
    state = {"cash": 100_000.0, "kill_state": kill_state}

    with pytest.raises(ValueError, match="restored kill switch state"):
        SimulatedBroker.from_state(cfg, state)


def test_from_state_preserves_runtime_kill_switch_state(tmp_path):
    cfg = _cfg(tmp_path)
    broker = SimulatedBroker(config=cfg, initial_cash=100_000.0)
    broker.kill.set_state(HALT_NEW_ORDERS)

    restored = SimulatedBroker.from_state(cfg, broker.to_dict())

    assert restored.kill.state == HALT_NEW_ORDERS
    rec = restored.submit(_order(), price=100.0, nav=100_000.0, adv_dollars=1e9)
    assert rec.order.status is OrderStatus.REJECTED
    assert rec.reject_reason == RejectReason.KILL_SWITCH.value
    assert restored.shares.get("A", 0.0) == 0.0


def test_from_state_preserves_legacy_receipt_counter_baselines(tmp_path):
    cfg = _cfg(tmp_path)
    restored = SimulatedBroker.from_state(
        cfg,
        {
            "initial_cash": 100_000.0,
            "cash": 99_000.0,
            "shares": {"A": 10.0},
            "n_orders": 7,
            "n_fills": 3,
        },
    )

    assert restored.history == []
    assert restored.to_dict()["n_orders"] == 7
    assert restored.to_dict()["n_fills"] == 3


def test_from_state_rejects_receipt_history_count_mismatch(tmp_path):
    cfg = _cfg(tmp_path)
    broker = SimulatedBroker(config=cfg, initial_cash=100_000.0)
    state = broker.to_dict()
    state["n_orders"] = 1
    with pytest.raises(ValueError, match="order count"):
        SimulatedBroker.from_state(cfg, state)
