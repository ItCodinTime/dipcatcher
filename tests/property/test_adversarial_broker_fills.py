"""Seeded adversarial attacks on the simulated broker and cost model.

Plain pytest + ``np.random.default_rng`` loops. A fill can only exist on a
real, well-formed bar at a price inside that bar's range; costs are never
negative rebates; cash and positions are conserved exactly.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from quant_fund.execution.costs import total_cost
from quant_fund.execution.simulated_broker import RejectReason, SimulatedBroker
from quant_fund.schemas.orders import Order, OrderSide, OrderStatus
from tests.property._books import research_config

T0 = datetime(2024, 1, 2, tzinfo=UTC)

BAD_MARKS = [None, float("nan"), float("inf"), float("-inf"), 0.0, -5.0]

MALFORMED_BARS = [
    (110.0, 100.0, 90.0),  # open above the printed high
    (90.0, 110.0, 100.0),  # open below the printed low
    (100.0, 90.0, 110.0),  # high below low
    (float("nan"), 100.0, 90.0),
    (100.0, float("nan"), 90.0),
    (100.0, 100.0, float("nan")),
    (float("inf"), 110.0, 90.0),
    (0.0, 100.0, 90.0),
    (-1.0, 100.0, 90.0),
]


def _broker(**kwargs: object) -> SimulatedBroker:
    return SimulatedBroker(config=research_config(), **kwargs)


def _order(
    oid: str = "o1",
    sid: str = "A",
    side: OrderSide = OrderSide.BUY,
    qty: float = 10.0,
    limit: float | None = None,
    t: datetime = T0,
    expire: datetime | None = None,
) -> Order:
    return Order(
        order_id=oid,
        security_id=sid,
        symbol=sid,
        side=side,
        quantity=qty,
        signal_time=t,
        decision_time=t,
        order_time=t,
        status=OrderStatus.NEW,
        limit_price=limit,
        expire_time=expire,
    )


def _fill_cost_total(fill) -> float:
    return float(fill.fee + fill.spread_cost + fill.impact_cost + fill.turnover_cost)


@pytest.mark.parametrize("price", BAD_MARKS)
def test_submit_rejects_bad_execution_marks(price) -> None:
    """Bad marks produce a REJECTED record, never a fill or state change."""
    broker = _broker()
    rec = broker.submit(_order(), price=price, nav=1e6, adv_dollars=1e9, sigma=0.02)
    assert rec.order.status is OrderStatus.REJECTED
    assert rec.reject_reason == RejectReason.MISSING_PRICE.value
    assert rec.fill is None
    assert broker.cash == broker.initial_cash
    assert not broker.fills and not broker.open_orders


def test_submit_rejects_invalid_market_data_seeded() -> None:
    """NaN/non-positive ADV, negative vol, or bad market vol all fail closed."""
    bad = [float("nan"), float("inf"), 0.0, -1.0]
    rng = np.random.default_rng(20240911)
    for i in range(64):
        broker = _broker()
        case = int(rng.integers(0, 3))
        adv = float(rng.choice(bad)) if case == 0 else 1e9
        sig = float(rng.choice([float("nan"), -0.1])) if case == 1 else 0.02
        mv = float(rng.choice([float("nan"), -0.5])) if case == 2 else None
        rec = broker.submit(
            _order(f"m{i}"),
            price=100.0,
            nav=1e6,
            adv_dollars=adv,
            sigma=sig,
            market_predicted_vol=mv,
        )
        assert rec.order.status is OrderStatus.REJECTED
        assert rec.reject_reason == RejectReason.INVALID_MARKET_DATA.value
        assert rec.fill is None and not broker.fills


@pytest.mark.parametrize("qty", [0.0, -5.0, float("nan"), float("inf")])
def test_zero_or_invalid_quantity_rejected_cleanly(qty) -> None:
    """model_copy bypasses Order validation; submit must still refuse."""
    broker = _broker()
    order = _order(qty=10.0).model_copy(update={"quantity": qty})
    rec = broker.submit(order, price=100.0, nav=1e6, adv_dollars=1e9, sigma=0.02)
    assert rec.order.status is OrderStatus.REJECTED
    assert rec.reject_reason == RejectReason.ZERO_QTY.value
    assert broker.cash == broker.initial_cash


def test_duplicate_order_id_fails_closed() -> None:
    """A second live order with the same id must not silently overwrite the first."""
    broker = _broker()
    broker.submit(_order("dup", limit=99.0), price=100.0, nav=1e6, adv_dollars=1e9)
    with pytest.raises(ValueError, match="already working"):
        broker.submit(
            _order("dup", side=OrderSide.SELL, qty=50.0, limit=101.0),
            price=100.0,
            nav=1e6,
            adv_dollars=1e9,
        )
    resting = broker.open_orders["dup"]
    assert resting.side is OrderSide.BUY
    assert resting.limit_price == 99.0


def test_order_id_free_after_fill_or_cancel() -> None:
    """Ids are only reserved while working; completed ids may be reused."""
    broker = _broker()
    broker.submit(_order("x"), price=100.0, nav=1e6, adv_dollars=1e9)
    rec = broker.submit(_order("x"), price=100.0, nav=1e6, adv_dollars=1e9)
    assert rec.order.status is OrderStatus.FILLED

    broker2 = _broker()
    broker2.submit(_order("y", limit=99.0), price=100.0, nav=1e6, adv_dollars=1e9)
    broker2.cancel_order("y")
    broker2.submit(_order("y", limit=98.0), price=100.0, nav=1e6, adv_dollars=1e9)
    assert broker2.open_orders["y"].limit_price == 98.0


def test_limit_fills_never_leave_the_bar_range() -> None:
    """Any fill on a valid bar is inside [low, high] and never worse than the limit."""
    rng = np.random.default_rng(20240902)
    for i in range(500):
        broker = _broker()
        lo, hi = sorted(rng.uniform(5.0, 500.0, 2).tolist())
        op = float(rng.uniform(lo, hi))
        limit = float(rng.uniform(lo * 0.5, hi * 2.0))
        side = OrderSide.BUY if rng.random() < 0.5 else OrderSide.SELL
        broker.shares["A"] = 10.0  # small book keeps the name-size gate clear
        broker.mark({"A": op})
        rec = broker.submit(
            _order(f"o{i}", side=side, qty=1.0, limit=limit),
            price=op,
            nav=1e8,
            adv_dollars=1e12,
            sigma=0.02,
            bar_open=op,
            bar_high=hi,
            bar_low=lo,
        )
        touched = lo <= limit if side is OrderSide.BUY else hi >= limit
        if not touched:
            assert rec.fill is None
            assert broker.open_orders[f"o{i}"].status is OrderStatus.ACKED
            continue
        assert rec.order.status is OrderStatus.FILLED
        fill = rec.fill
        expected = min(op, limit) if side is OrderSide.BUY else max(op, limit)
        assert fill.price == pytest.approx(expected, rel=0, abs=1e-12)
        assert lo <= fill.price <= hi
        if side is OrderSide.BUY:
            assert fill.price <= limit
        else:
            assert fill.price >= limit
        assert _fill_cost_total(fill) >= 0.0


@pytest.mark.parametrize("bar", MALFORMED_BARS)
def test_process_bar_refuses_malformed_bars(bar) -> None:
    """A bar whose open is outside its own range cannot produce a fill."""
    broker = _broker()
    broker.submit(_order("o1", limit=99.0), price=100.0, nav=1e6, adv_dollars=1e9)
    with pytest.raises(ValueError):
        broker.process_bar("A", bar_open=bar[0], bar_high=bar[1], bar_low=bar[2], adv_dollars=1e9)
    assert broker.open_orders["o1"].status is OrderStatus.ACKED
    assert not broker.fills


@pytest.mark.parametrize("bar", MALFORMED_BARS)
def test_submit_limit_refuses_malformed_bars(bar) -> None:
    broker = _broker()
    with pytest.raises(ValueError):
        broker.submit(
            _order("o1", limit=99.0),
            price=100.0,
            nav=1e6,
            adv_dollars=1e9,
            bar_open=bar[0],
            bar_high=bar[1],
            bar_low=bar[2],
        )
    assert not broker.fills


def test_no_fill_without_bar_context() -> None:
    """Limit orders rest when no bar is supplied — never fill blind."""
    broker = _broker()
    rec = broker.submit(_order("o1", limit=99.0), price=100.0, nav=1e6, adv_dollars=1e9)
    assert rec.order.status is OrderStatus.ACKED
    assert rec.fill is None
    assert broker.cash == broker.initial_cash
    # A bar for a different security does not touch the resting order.
    broker.process_bar("B", bar_open=50.0, bar_high=51.0, bar_low=49.0, adv_dollars=1e9)
    assert broker.open_orders["o1"].status is OrderStatus.ACKED
    assert not broker.fills


def test_resting_order_expires_at_expire_time() -> None:
    broker = _broker()
    expire = T0 + timedelta(days=1)
    broker.submit(_order("o1", limit=95.0, expire=expire), price=100.0, nav=1e6, adv_dollars=1e9)
    recs = broker.process_bar(
        "A",
        bar_open=100.0,
        bar_high=101.0,
        bar_low=99.0,
        bar_time=expire + timedelta(seconds=1),
        adv_dollars=1e9,
    )
    assert len(recs) == 1
    assert recs[0].order.status is OrderStatus.CANCELLED
    assert recs[0].reject_reason == "expired"
    assert "o1" not in broker.open_orders


def test_flat_bar_price_ties_fill_at_the_print() -> None:
    """Identical open/high/low: touches resolve to exactly the printed price."""
    for limit, side, expect_fill in [
        (100.0, OrderSide.BUY, True),
        (99.9, OrderSide.BUY, False),
        (100.0, OrderSide.SELL, True),
        (100.1, OrderSide.SELL, False),
    ]:
        broker = _broker()
        broker.shares["A"] = 10.0  # tiny book keeps the name-size gate clear
        broker.submit(
            _order("o1", side=side, qty=1.0, limit=limit),
            price=100.0,
            nav=1e6,
            adv_dollars=1e9,
        )
        recs = broker.process_bar(
            "A", bar_open=100.0, bar_high=100.0, bar_low=100.0, adv_dollars=1e9
        )
        if expect_fill:
            assert len(recs) == 1 and recs[0].fill is not None
            assert recs[0].fill.price == 100.0
        else:
            assert not recs or all(r.fill is None for r in recs)
            assert broker.open_orders["o1"].status is OrderStatus.ACKED


def test_buy_never_spends_more_than_cash() -> None:
    """A buy that cannot clear the cash check is rejected, not financed.

    Reaching the cash check needs max_net > 1: a buy always adds notional to
    net, so with max_net <= 1 the net gate fires first whenever the order
    exceeds cash. With a loose net limit a large mark-valued book clears the
    gate while cash cannot cover the ticket.
    """
    cfg = research_config()
    cfg.risk_gate.max_net = 2.0
    broker = SimulatedBroker(config=cfg, initial_cash=1000.0)
    broker.shares["X"] = 9990.0
    broker.mark({"X": 100.0})  # nav ~= 1e6
    rec = broker.submit(
        _order("o1", sid="A", qty=20.0),
        price=100.0,
        nav=broker.nav(),
        adv_dollars=1e12,
        sigma=0.02,
    )
    assert rec.order.status is OrderStatus.REJECTED
    assert rec.reject_reason == RejectReason.INSUFFICIENT_CASH.value
    assert broker.cash == 1000.0


def test_cash_and_position_conservation_seeded() -> None:
    """cash = cash0 - signed_notional - costs; shares accumulate signed fills."""
    for seed in range(40):
        rng = np.random.default_rng(seed)
        broker = _broker()
        n = int(rng.integers(1, 4))
        expected_cash = float(broker.initial_cash)
        expected_shares: dict[str, float] = {}
        for _ in range(int(rng.integers(3, 15))):
            sid = f"S{int(rng.integers(0, n))}"
            px = float(rng.uniform(5.0, 500.0))
            broker.mark({sid: px})
            qty = float(rng.uniform(0.5, 30.0))
            side = OrderSide.BUY if rng.random() < 0.6 else OrderSide.SELL
            rec = broker.submit(
                _order(f"o{seed}-{sid}-{rng.integers(1e9)}", sid=sid, side=side, qty=qty),
                price=px,
                nav=max(broker.nav(), 1.0),
                adv_dollars=float(rng.uniform(1e8, 1e10)),
                sigma=float(rng.uniform(0.005, 0.5)),
            )
            if rec.fill is None:
                assert rec.order.status in (OrderStatus.REJECTED, OrderStatus.ACKED)
                continue
            signed = float(rec.fill.quantity) * (1.0 if side is OrderSide.BUY else -1.0)
            expected_cash -= signed * float(rec.fill.price) + _fill_cost_total(rec.fill)
            expected_shares[sid] = expected_shares.get(sid, 0.0) + signed
            assert _fill_cost_total(rec.fill) >= 0.0
            assert rec.fill.slippage >= 0.0
        assert broker.cash == pytest.approx(expected_cash, rel=1e-9, abs=1e-9)
        for sid, qty in expected_shares.items():
            assert broker.shares.get(sid, 0.0) == pytest.approx(qty, rel=1e-12, abs=1e-9)
        ident = broker.cash_nav_identity()
        assert ident["residual"] == 0.0
        assert ident["nav"] == pytest.approx(ident["cash"] + ident["position_mv"])


def test_participation_cap_re_rests_residual() -> None:
    """A capped fill leaves a PARTIAL residual working at the same limit."""
    broker = _broker(initial_cash=1e12)
    rec = broker.submit(
        _order("o1", qty=2e7, limit=101.0),
        price=100.0,
        nav=1e12,
        adv_dollars=1e9,
        sigma=0.02,
        bar_open=100.0,
        bar_high=101.0,
        bar_low=99.0,
    )
    assert rec.fill is not None
    assert rec.fill.is_partial
    max_qty = broker.config.costs.participation_limit * (1e9 / 100.0)
    assert rec.fill.quantity == pytest.approx(max_qty)
    residual = broker.open_orders["o1"]
    assert residual.status is OrderStatus.PARTIAL
    assert residual.quantity == pytest.approx(2e7 - rec.fill.quantity)


def test_target_to_orders_skips_untradeable_marks() -> None:
    """Missing, non-finite, or non-positive marks produce no order for that name."""
    broker = _broker()
    targets = {"A": 0.5, "B": 0.5, "C": 0.5, "D": 0.5, "E": 0.5}
    prices = {"A": 100.0, "B": float("nan"), "C": 0.0, "D": -5.0, "E": float("inf")}
    orders = broker.target_to_orders(targets, prices, signal_time=T0, order_time=T0)
    assert {o.security_id for o in orders} == {"A"}
    assert orders[0].quantity > 0


def test_target_to_orders_rejects_non_finite_nav() -> None:
    broker = _broker()
    for bad_nav in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValueError, match="nav must be finite"):
            broker.target_to_orders(
                {"A": 0.5}, {"A": 100.0}, signal_time=T0, order_time=T0, nav=bad_nav
            )


def test_total_cost_never_negative_and_validates_seeded() -> None:
    """No negative-cost rebates on valid inputs; invalid inputs raise."""
    cfg = research_config().costs
    for seed in range(20):
        rng = np.random.default_rng(seed)
        for _ in range(200):
            qty = float(rng.uniform(1e-3, 1e5)) * rng.choice([1.0, -1.0])
            price = float(rng.uniform(0.1, 1e4))
            adv = float(rng.uniform(1e3, 1e12))
            sigma = float(rng.uniform(0.0, 3.0))
            costs = total_cost(qty, price, adv, sigma, cfg)
            for key in ("commission", "spread", "impact", "borrow", "turnover_bps", "total"):
                assert np.isfinite(float(costs[key]))
                assert float(costs[key]) >= 0.0
    bad_inputs = [
        (float("nan"), 100.0, 1e9, 0.02),
        (1.0, 0.0, 1e9, 0.02),
        (1.0, -1.0, 1e9, 0.02),
        (1.0, float("nan"), 1e9, 0.02),
        (1.0, 100.0, 0.0, 0.02),
        (1.0, 100.0, -5.0, 0.02),
        (1.0, 100.0, float("nan"), 0.02),
        (1.0, 100.0, 1e9, -0.1),
        (1.0, 100.0, 1e9, float("nan")),
    ]
    for qty, price, adv, sigma in bad_inputs:
        with pytest.raises(ValueError):
            total_cost(qty, price, adv, sigma, cfg)


def test_costs_scale_monotonic_in_participation() -> None:
    """Impact grows with participation; it cannot give back cash at scale."""
    cfg = research_config().costs
    rng = np.random.default_rng(9)
    prev = 0.0
    for part in np.geomspace(1e-4, 1.0, 25):
        adv = 1e9
        qty = part * adv / 100.0
        imp = float(total_cost(qty, 100.0, adv, 0.02, cfg)["impact"])
        assert imp >= prev >= 0.0
        prev = imp
        _ = rng  # deterministic geomspace; seed kept for symmetry
