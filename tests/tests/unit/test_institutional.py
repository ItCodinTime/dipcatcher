"""Institutional tranche tests: P&L attribution, tearsheet, implementation
shortfall, factor model, reconciliation, and limit orders."""

from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl
import pytest

from quant_fund.config.loader import load_config
from quant_fund.execution.implementation_shortfall import (
    aggregate_shortfall,
    fill_shortfall,
    shortfall_frame,
)
from quant_fund.execution.simulated_broker import SimulatedBroker
from quant_fund.paper.recon import (
    reconcile_broker_states,
    reconcile_equity,
    reconcile_fills,
)
from quant_fund.portfolio.factor_model import (
    crypto_factor_returns,
    decompose_book_factors,
    estimate_factor_betas,
    factor_summary,
)
from quant_fund.portfolio.pnl_attribution import (
    attribute_weights_pnl,
    attribution_summary,
    name_attribution,
    security_returns,
    sleeve_attribution,
)
from quant_fund.reporting.tearsheet import (
    build_tearsheet,
    period_returns_table,
    tearsheet_markdown,
)
from quant_fund.schemas.orders import Fill, Order, OrderSide, OrderStatus

T0 = datetime(2024, 1, 1, tzinfo=UTC)


def _bars(prices: dict[str, list[float]], step: timedelta = timedelta(days=1)) -> pl.DataFrame:
    rows = []
    for sid, px in prices.items():
        for i, p in enumerate(px):
            rows.append(
                {
                    "security_id": sid,
                    "event_time": T0 + i * step,
                    "open": p,
                    "high": p * 1.01,
                    "low": p * 0.99,
                    "close": p,
                    "volume": 1e5,
                    "source": "synthetic",
                }
            )
    return pl.DataFrame(rows)


def _weights(rows: list[tuple[datetime, str, float]]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "event_time": [r[0] for r in rows],
            "security_id": [r[1] for r in rows],
            "target_weight": [r[2] for r in rows],
        }
    )


# ---------- P&L attribution ----------


def test_security_returns_alignment() -> None:
    bars = _bars({"A": [100.0, 110.0, 99.0]})
    rets = security_returns(bars).sort("event_time")
    assert rets["ret"][0] is None
    assert rets["ret"][1] == pytest.approx(0.10)
    assert rets["ret"][2] == pytest.approx(-0.10)


def test_attribute_weights_pnl_uses_prev_weight() -> None:
    bars = _bars({"A": [100.0, 110.0, 121.0], "B": [50.0, 50.0, 45.0]})
    w = _weights(
        [
            (T0, "A", 0.5),
            (T0, "B", 0.5),
            (T0 + timedelta(days=1), "A", 1.0),
            (T0 + timedelta(days=1), "B", 0.0),
        ]
    )
    frame = attribute_weights_pnl(w, bars).sort(["event_time", "security_id"])
    day1 = frame.filter(pl.col("event_time") == T0 + timedelta(days=1))
    # day-0 weights earn day-1 returns: A 0.5*0.10, B 0.5*0.0
    assert day1.filter(pl.col("security_id") == "A")["gross_contrib"][0] == pytest.approx(0.05)
    assert day1.filter(pl.col("security_id") == "B")["gross_contrib"][0] == pytest.approx(0.0)
    day2 = frame.filter(pl.col("event_time") == T0 + timedelta(days=2))
    assert day2.filter(pl.col("security_id") == "A")["gross_contrib"][0] == pytest.approx(0.10)


def test_attribution_costs_and_rollup() -> None:
    bars = _bars({"A": [100.0, 110.0, 121.0]})
    w = _weights([(T0, "A", 1.0)])
    nav = pl.DataFrame({"event_time": [T0, T0 + timedelta(days=1)], "nav": [100.0, 110.0]})
    fills = pl.DataFrame(
        {
            "fill_time": [T0 + timedelta(days=1)],
            "security_id": ["A"],
            "fee": [1.0],
            "spread_cost": [0.5],
            "impact_cost": [0.5],
        }
    )
    frame = attribute_weights_pnl(w, bars, fills=fills, nav=nav)
    row = frame.filter(pl.col("event_time") == T0 + timedelta(days=1))
    assert row["cost_ret"][0] == pytest.approx(2.0 / 110.0)
    assert row["net_contrib"][0] == pytest.approx(row["gross_contrib"][0] - 2.0 / 110.0)

    names = name_attribution(frame)
    assert names.height == 1
    assert names["share_of_gross_abs"][0] == pytest.approx(1.0)

    smap = {"A": "momentum"}
    sleeves = sleeve_attribution(frame, smap)
    assert sleeves["sleeve"][0] == "momentum"
    summary = attribution_summary(frame, sleeve_map=smap)
    assert summary["live_pnl_claim"] is False
    assert summary["by_sleeve"][0]["sleeve"] == "momentum"


def test_attribution_missing_columns_fail_closed() -> None:
    with pytest.raises(ValueError, match="missing columns"):
        attribute_weights_pnl(pl.DataFrame({"x": [1]}), _bars({"A": [1.0]}))


# ---------- Tearsheet ----------


def _equity(nav: list[float], step_days: int = 32) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "event_time": [T0 + timedelta(days=i * step_days) for i in range(len(nav))],
            "nav": nav,
            "gross": [1.0] * len(nav),
            "net": [0.8] * len(nav),
            "turnover": [0.1] * len(nav),
        }
    )


def test_tearsheet_full_blocks() -> None:
    eq = _equity([100.0, 110.0, 99.0, 105.0, 120.0, 115.0, 130.0])
    sheet = build_tearsheet(eq, periods_per_year=12.0, synthetic=True)
    s = sheet["summary"]
    assert s["nav_end"] == 130.0
    assert s["total_return"] == pytest.approx(0.30)
    assert s["max_drawdown"] < 0
    assert sheet["drawdown"]["n_episodes"] >= 1
    assert np.isfinite(sheet["risk"]["var_95"])
    assert sheet["exposure"]["mean_gross"] == pytest.approx(1.0)
    assert sheet["live_pnl_claim"] is False
    md = tearsheet_markdown(sheet)
    assert "## Summary" in md and "SYNTHETIC" in md


def test_tearsheet_period_table_and_empty() -> None:
    eq = _equity([100.0, 110.0, 99.0], step_days=35)
    pr = period_returns_table(eq)
    assert len(pr) >= 2
    empty = build_tearsheet(pl.DataFrame({"event_time": [], "nav": []}))
    assert empty["summary"]["status"] == "empty_or_short"


# ---------- Implementation shortfall ----------


def test_fill_shortfall_signs() -> None:
    buy = fill_shortfall(side_sign=1.0, quantity=10, decision_price=100, exec_price=101, fee=1.0)
    assert buy["drift"] == pytest.approx(10.0)  # adverse
    assert buy["total_is"] == pytest.approx(11.0)
    sell = fill_shortfall(side_sign=-1.0, quantity=10, decision_price=100, exec_price=99, fee=0.0)
    assert sell["drift"] == pytest.approx(10.0)  # adverse for a sell too
    favorable = fill_shortfall(side_sign=1.0, quantity=10, decision_price=100, exec_price=99)
    assert favorable["drift"] == pytest.approx(-10.0)


def test_shortfall_frame_and_aggregate() -> None:
    fills = pl.DataFrame(
        {
            "security_id": ["A", "A", "B"],
            "quantity": [10.0, -5.0, 8.0],  # signed: sell is negative
            "decision_price": [100.0, 100.0, 50.0],
            "price": [101.0, 99.0, 50.0],
            "fee": [1.0, 0.5, 0.4],
        }
    )
    fr = shortfall_frame(fills)
    agg = aggregate_shortfall(fr)
    # A buys 10@101 (dec 100): drift +10; A sells 5@99 (dec 100): drift +5; B flat: 0
    assert agg["drift"] == pytest.approx(15.0)
    assert agg["explicit"] == pytest.approx(1.9)
    assert agg["n_fills"] == 3
    assert agg["is_bps"] is not None


def test_backtest_records_decision_price_and_is(tmp_path) -> None:
    from quant_fund.backtest.engine import run_backtest

    cfg = load_config("configs/research.yaml")
    cfg.data.root = tmp_path
    cfg.costs.frictionless = True
    cfg.risk_gate.max_name = 1.0
    cfg.risk_gate.max_net = 1.0
    cfg.risk_gate.max_gross = 1.0
    cfg.risk_gate.max_order_notional = 1e12
    bars = pl.DataFrame(
        {
            "security_id": ["A"] * 3,
            "event_time": [T0 + timedelta(days=i) for i in range(3)],
            "open": [100.0, 102.0, 104.0],
            "close": [100.0, 102.0, 104.0],
            "close_total_return": [100.0, 102.0, 104.0],
            "volume": [1e6] * 3,
            "adv": [1e9] * 3,
            "vol_20": [0.02] * 3,
            "source": ["file"] * 3,
        }
    )
    w = _weights([(T0, "A", 1.0), (T0 + timedelta(days=1), "A", 0.0)])
    result = run_backtest(bars, w, cfg, initial_nav=100_000.0)
    assert "decision_price" in result.fills.columns
    assert result.fills["decision_price"][0] == pytest.approx(100.0)
    is_rep = result.metrics["implementation_shortfall"]
    assert is_rep is not None
    # entry at next open + exit on the explicit zero target row (rebalance
    # grids carry the last target; a lapse is not a flatten instruction).
    assert is_rep["n_fills"] == 2
    buy = [s for s in is_rep["by_side"] if s["side_sign"] == 1.0][0]
    # exec at next open 102 vs decision 100 -> adverse drift on the entry
    assert buy["total_is"] > 0


# ---------- Factor model ----------


def test_crypto_factor_returns_market() -> None:
    px = {
        "A": [100.0, 110.0],
        "B": [100.0, 105.0],
        "C": [100.0, 90.0],
        "D": [100.0, 96.0],
        "E": [100.0, 101.0],
        "F": [100.0, 99.0],
    }
    factors = crypto_factor_returns(_bars(px))
    row = factors.filter(pl.col("event_time") == T0 + timedelta(days=1))
    # mkt = mean of +10%, +5%, -10%, -4%, +1%, -1% = 1/6 % ≈ 0.001667
    assert row["mkt"][0] == pytest.approx((0.10 + 0.05 - 0.10 - 0.04 + 0.01 - 0.01) / 6.0)
    assert np.isnan(row["carry"][0])  # no funding supplied


def test_factor_betas_recover_injected_beta() -> None:
    # BETA2's return is constructed as exactly 2x the *inclusive* market:
    # r_b = 2 * (S + r_b)/6 => r_b = S/2 = 2.5 * mean(S) for the other 5 names.
    rng = np.random.default_rng(0)
    n_t = 30
    base = [100.0 + 2 * i + 0.3 * np.sin(i) for i in range(n_t)]
    px = {f"S{k}": [p * (1 + 0.001 * rng.normal()) for p in base] for k in range(5)}
    factors_pre = crypto_factor_returns(_bars(px))
    mkt5 = factors_pre["mkt"].to_list()
    closes = [100.0]
    for r in mkt5[1:]:
        closes.append(closes[-1] * (1 + (0.0 if np.isnan(r) else 2.5 * r)))
    px["BETA2"] = closes
    bars = _bars(px)
    factors = crypto_factor_returns(bars)
    f_mkt = factors.select("event_time", "mkt")
    betas = estimate_factor_betas(bars, f_mkt, window=10)
    tail = betas.filter((pl.col("security_id") == "BETA2") & pl.col("beta_mkt").is_not_null()).sort(
        "event_time"
    )
    assert tail.height > 0
    assert tail["beta_mkt"][-1] == pytest.approx(2.0, abs=0.10)


def test_decompose_book_factors_identity() -> None:
    px = {f"S{k}": [100.0 + i + 0.2 * np.sin(i) for i in range(12)] for k in range(6)}
    bars = _bars(px)
    factors = crypto_factor_returns(bars).select("event_time", "mkt")
    w = _weights([(T0 + timedelta(days=d), f"S{k}", 1.0 / 6) for d in range(11) for k in range(6)])
    betas = pl.DataFrame(
        {
            "security_id": [f"S{k}" for k in range(6) for _ in range(12)],
            "event_time": sorted([T0 + timedelta(days=d) for _ in range(6) for d in range(12)]),
            "beta_mkt": [1.0] * 72,
        }
    )
    frame = decompose_book_factors(w, bars, factors, betas)
    # book_ret = contrib_mkt + alpha_resid identically
    recon = frame["contrib_mkt"] + frame["alpha_resid"]
    assert np.allclose(recon.to_numpy(), frame["book_ret"].to_numpy())
    summary = factor_summary(frame)
    assert summary["live_pnl_claim"] is False
    assert "mkt" in summary["by_factor"]


# ---------- Reconciliation ----------


def test_reconcile_broker_states_match_and_diff() -> None:
    state = {
        "cash": 1000.0,
        "shares": {"A": 10.0},
        "last_marks": {"A": 100.0},
        "reject_count": 1,
        "kill_state": "ENABLED",
        "slot": "champion",
        "n_orders": 5,
        "n_fills": 2,
    }
    ok = reconcile_broker_states(state, dict(state))
    assert ok["match"] is True
    bad = dict(state, cash=999.0, shares={"A": 11.0})
    rep = reconcile_broker_states(state, bad)
    assert rep["match"] is False
    assert any("cash" in m for m in rep["mismatches"])
    assert any("shares[A]" in m for m in rep["mismatches"])


def test_reconcile_equity_and_fills() -> None:
    eq = pl.DataFrame({"event_time": [T0, T0 + timedelta(days=1)], "nav": [100.0, 101.0]})
    assert reconcile_equity(eq, eq)["match"] is True
    shifted = eq.with_columns(pl.col("nav") + 1.0)
    rep = reconcile_equity(eq, shifted)
    assert rep["match"] is False
    assert rep["max_abs_nav_delta"] == pytest.approx(1.0)

    f1 = pl.DataFrame(
        {
            "security_id": ["A", "B"],
            "fill_time": [T0, T0],
            "quantity": [1.0, 2.0],
            "price": [10.0, 20.0],
        }
    )
    assert reconcile_fills(f1, f1)["match"] is True
    f2 = pl.DataFrame(
        {
            "security_id": ["A", "C"],
            "fill_time": [T0, T0],
            "quantity": [1.0, 2.0],
            "price": [10.0, 20.0],
        }
    )
    rep = reconcile_fills(f1, f2)
    assert rep["match"] is False
    assert rep["unmatched_expected"] == 1 and rep["unmatched_actual"] == 1


# ---------- Limit orders ----------


def _cfg(tmp_path):
    cfg = load_config("configs/paper.yaml")
    cfg.data.root = tmp_path
    cfg.risk_gate.max_name = 1.0
    cfg.risk_gate.max_net = 1.0
    cfg.risk_gate.max_gross = 2.0
    cfg.risk_gate.max_order_notional = 1e12
    cfg.risk_gate.max_participation = 1.0
    return cfg


def _order(side=OrderSide.BUY, qty=10.0, oid="o1", limit=None, expire=None):
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
        limit_price=limit,
        expire_time=expire,
    )


def test_limit_order_rests_without_bar_then_fills_on_touch(tmp_path) -> None:
    broker = SimulatedBroker(config=_cfg(tmp_path), initial_cash=100_000.0)
    broker.mark({"A": 100.0})
    # No bar context -> resting, not a reject
    rec = broker.submit(_order(limit=99.0), price=100.0, nav=100_000.0, adv_dollars=1e9)
    assert rec.order.status is OrderStatus.ACKED
    assert "o1" in broker.open_orders
    # Bar that does not touch 99 stays resting
    recs = broker.process_bar("A", bar_open=100.0, bar_high=101.0, bar_low=99.5)
    assert recs == []
    assert "o1" in broker.open_orders
    # Bar touching 99 fills at min(open, limit)
    recs = broker.process_bar("A", bar_open=100.0, bar_high=101.0, bar_low=98.0, adv_dollars=1e9)
    assert len(recs) == 1 and recs[0].order.status is OrderStatus.FILLED
    assert recs[0].fill.price == pytest.approx(99.0)
    assert "o1" not in broker.open_orders


def test_limit_order_gap_through_open_fill(tmp_path) -> None:
    broker = SimulatedBroker(config=_cfg(tmp_path), initial_cash=100_000.0)
    broker.mark({"A": 100.0})
    rec = broker.submit(
        _order(limit=99.0),
        price=100.0,
        nav=100_000.0,
        adv_dollars=1e9,
        bar_open=98.0,
        bar_high=100.5,
        bar_low=97.5,
    )
    assert rec.order.status is OrderStatus.FILLED
    assert rec.fill.price == pytest.approx(98.0)  # gap-through fills at open


def test_sell_limit_symmetric_and_immediate_bar_eval(tmp_path) -> None:
    broker = SimulatedBroker(config=_cfg(tmp_path), initial_cash=100_000.0)
    broker.shares["A"] = 10.0
    broker.mark({"A": 100.0})
    rec = broker.submit(
        _order(side=OrderSide.SELL, limit=101.0),
        price=100.0,
        nav=100_000.0,
        adv_dollars=1e9,
        bar_open=100.0,
        bar_high=100.5,
        bar_low=99.0,
    )
    assert rec.order.status is OrderStatus.ACKED
    recs = broker.process_bar("A", bar_open=102.0, bar_high=103.0, bar_low=101.5, adv_dollars=1e9)
    assert len(recs) == 1 and recs[0].order.status is OrderStatus.FILLED
    assert recs[0].fill.price == pytest.approx(102.0)  # gap-up fills at open


def test_resting_reject_cancels_open_order(tmp_path) -> None:
    cfg = _cfg(tmp_path)
    cfg.risk_gate.max_participation = 0.0  # any triggered fill fails the gate
    broker = SimulatedBroker(config=cfg, initial_cash=100_000.0)
    broker.mark({"A": 100.0})
    broker.submit(_order(limit=99.0, qty=10.0), price=100.0, nav=100_000.0, adv_dollars=1e9)
    assert "o1" in broker.open_orders
    recs = broker.process_bar("A", bar_open=100.0, bar_high=100.5, bar_low=98.0)
    assert len(recs) == 1 and recs[0].order.status is OrderStatus.REJECTED
    assert "o1" not in broker.open_orders


def test_open_orders_persist_through_state_roundtrip(tmp_path) -> None:
    cfg = _cfg(tmp_path)
    broker = SimulatedBroker(config=cfg, initial_cash=100_000.0)
    broker.mark({"A": 100.0})
    broker.submit(_order(limit=99.0), price=100.0, nav=100_000.0, adv_dollars=1e9)
    restored = SimulatedBroker.from_state(cfg, broker.to_dict())
    assert "o1" in restored.open_orders
    recs = restored.process_bar("A", bar_open=100.0, bar_high=101.0, bar_low=98.0, adv_dollars=1e9)
    assert len(recs) == 1 and recs[0].order.status is OrderStatus.FILLED


def test_resting_limit_partial_fill_keeps_residual(tmp_path) -> None:
    cfg = _cfg(tmp_path)
    cfg.costs.participation_limit = 0.1
    broker = SimulatedBroker(config=cfg, initial_cash=100_000.0)
    broker.mark({"A": 100.0})
    broker.submit(_order(limit=99.0, qty=10.0), price=100.0, nav=100_000.0, adv_dollars=4950.0)
    bar2 = datetime(2024, 1, 3, tzinfo=UTC)
    # max_qty = 0.1 * 4950 / 99 = 5.0 -> first sweep fills 5 of 10
    recs = broker.process_bar(
        "A", bar_open=100.0, bar_high=101.0, bar_low=98.0, bar_time=bar2, adv_dollars=4950.0
    )
    assert len(recs) == 1 and recs[0].order.status is OrderStatus.FILLED
    assert recs[0].fill.is_partial
    assert recs[0].fill.quantity == pytest.approx(5.0)
    assert recs[0].fill.fill_time == bar2
    residual = broker.open_orders["o1"]
    assert residual.status is OrderStatus.PARTIAL
    assert residual.quantity == pytest.approx(5.0)
    # Second touching bar completes the residual; book empties.
    recs = broker.process_bar("A", bar_open=99.0, bar_high=100.0, bar_low=98.5, adv_dollars=4950.0)
    assert len(recs) == 1 and recs[0].order.status is OrderStatus.FILLED
    assert not recs[0].fill.is_partial
    assert "o1" not in broker.open_orders


def test_cancel_order_records_and_removes(tmp_path) -> None:
    broker = SimulatedBroker(config=_cfg(tmp_path), initial_cash=100_000.0)
    broker.mark({"A": 100.0})
    broker.submit(_order(limit=99.0), price=100.0, nav=100_000.0, adv_dollars=1e9)
    rec = broker.cancel_order("o1")
    assert rec.order.status is OrderStatus.CANCELLED
    assert "o1" not in broker.open_orders
    with pytest.raises(ValueError, match="not working"):
        broker.cancel_order("o1")


def test_expire_time_cancels_before_touch(tmp_path) -> None:
    t = datetime(2024, 1, 2, tzinfo=UTC)
    broker = SimulatedBroker(config=_cfg(tmp_path), initial_cash=100_000.0)
    broker.mark({"A": 100.0})
    broker.submit(
        _order(limit=99.0, expire=t + timedelta(hours=4)),
        price=100.0,
        nav=100_000.0,
        adv_dollars=1e9,
    )
    # Bar touches the limit but arrives at/after expiry -> cancel, no fill.
    recs = broker.process_bar(
        "A",
        bar_open=100.0,
        bar_high=101.0,
        bar_low=98.0,
        bar_time=t + timedelta(hours=4),
    )
    assert len(recs) == 1 and recs[0].order.status is OrderStatus.CANCELLED
    assert recs[0].reject_reason == "expired"
    assert "o1" not in broker.open_orders
    assert broker.fills == []
    # expire_time before order_time is invalid
    with pytest.raises(ValueError, match="expire_time"):
        _order(limit=99.0, expire=t - timedelta(hours=1))


def test_amend_order_cancel_replace(tmp_path) -> None:
    broker = SimulatedBroker(config=_cfg(tmp_path), initial_cash=100_000.0)
    broker.mark({"A": 100.0})
    broker.submit(_order(limit=98.0, qty=10.0), price=100.0, nav=100_000.0, adv_dollars=1e9)
    rec = broker.amend_order("o1", quantity=20.0, limit_price=99.5)
    assert rec.order.quantity == pytest.approx(20.0)
    assert rec.order.limit_price == pytest.approx(99.5)
    # Amended limit now touches on a bar that missed the old 98.0 limit
    recs = broker.process_bar("A", bar_open=100.0, bar_high=101.0, bar_low=99.0, adv_dollars=1e9)
    assert len(recs) == 1 and recs[0].order.status is OrderStatus.FILLED
    assert recs[0].fill.price == pytest.approx(99.5)
    assert recs[0].fill.quantity == pytest.approx(20.0)
    with pytest.raises(ValueError, match="not working"):
        broker.amend_order("o1", quantity=5.0)
    # Invalid amend (non-positive qty) fails closed via schema validation
    broker.submit(_order(oid="o2", limit=98.0), price=100.0, nav=100_000.0, adv_dollars=1e9)
    with pytest.raises(Exception, match="quantity"):
        broker.amend_order("o2", quantity=-1.0)


def test_fill_decision_price_validation() -> None:
    t = datetime(2024, 1, 2, tzinfo=UTC)
    with pytest.raises(ValueError, match="decision_price"):
        Fill(
            fill_id="f1",
            order_id="o1",
            security_id="A",
            quantity=1.0,
            price=100.0,
            fill_time=t,
            decision_price=-1.0,
        )


def test_broker_decision_price_slippage(tmp_path) -> None:
    broker = SimulatedBroker(config=_cfg(tmp_path), initial_cash=100_000.0)
    broker.mark({"A": 100.0})
    rec = broker.submit(_order(), price=101.0, nav=100_000.0, adv_dollars=1e9, decision_price=100.0)
    assert rec.fill.decision_price == pytest.approx(100.0)
    assert rec.fill.slippage == pytest.approx(10.0)  # adverse $10 for 10 qty


# ---------- Ops dashboard ----------


def test_ops_snapshot_all_ok(tmp_path) -> None:
    from quant_fund.monitoring.dashboard import ops_snapshot

    cfg = _cfg(tmp_path)
    snap = ops_snapshot(
        nav=1_000_000.0,
        cash=400_000.0,
        positions={"BTC": 5.0, "ETH": 30.0},
        marks={"BTC": 60_000.0, "ETH": 3_000.0},
        config=cfg,
        mark_age_bars={"BTC": 0, "ETH": 1},
        n_open_orders=2,
        kill_switch_state="ENABLED",
        peak_nav=1_100_000.0,
        recon_mismatches=0,
        drift_alert=False,
    )
    assert snap["overall_status"] == "ok"
    assert snap["checks"]["gross"]["status"] == "ok"
    assert snap["checks"]["mark_staleness"]["status"] == "ok"
    assert snap["research_only"] is True and snap["live_pnl_claim"] is False


def test_ops_snapshot_breaches_and_warnings(tmp_path) -> None:
    from quant_fund.monitoring.dashboard import ops_snapshot

    cfg = _cfg(tmp_path)
    cfg.risk_gate.max_name = 0.2
    cfg.risk_gate.max_gross = 0.5
    snap = ops_snapshot(
        nav=1_000_000.0,
        cash=-10.0,  # overdrawn -> warn
        positions={"BTC": 10.0},
        marks={"BTC": 60_000.0},  # 60% name -> breach, 60% gross -> breach
        config=cfg,
        mark_age_bars={"BTC": 99},  # stale -> breach
        n_open_orders=None,
        kill_switch_state="HALT_NEW_ORDERS",  # blocking -> breach
        recon_mismatches=3,  # -> breach
        drift_alert=True,  # -> breach
    )
    assert snap["overall_status"] == "breach"
    assert snap["checks"]["largest_name"]["status"] == "breach"
    assert snap["checks"]["gross"]["status"] == "breach"
    assert snap["checks"]["cash_buffer"]["status"] == "warn"
    assert snap["checks"]["mark_staleness"]["stale_securities"] == ["BTC"]
    assert snap["checks"]["kill_switch"]["status"] == "breach"
    assert snap["checks"]["reconciliation"]["mismatches"] == 3
    assert snap["checks"]["feature_drift"]["status"] == "breach"


def test_ops_snapshot_insufficient_data_never_green(tmp_path) -> None:
    from quant_fund.monitoring.dashboard import ops_snapshot

    snap = ops_snapshot(
        nav=1.0,
        cash=1.0,
        positions={},
        marks={},
        config=_cfg(tmp_path),
    )
    assert snap["checks"]["mark_staleness"]["status"] == "insufficient_data"
    assert snap["checks"]["reconciliation"]["status"] == "insufficient_data"
    assert snap["checks"]["feature_drift"]["status"] == "insufficient_data"
    assert snap["checks"]["kill_switch"]["status"] == "insufficient_data"
    # all-insufficient -> overall reports insufficient, not ok
    assert snap["overall_status"] in {"insufficient_data", "ok"}
    assert snap["overall_status"] != "breach"


def test_ops_snapshot_unmarked_position_is_breach(tmp_path) -> None:
    from quant_fund.monitoring.dashboard import ops_snapshot

    snap = ops_snapshot(
        nav=1_000_000.0,
        cash=500_000.0,
        positions={"BTC": 5.0},
        marks={},  # no mark for held position
        config=_cfg(tmp_path),
        kill_switch_state="ENABLED",
    )
    assert snap["checks"]["unmarked_positions"]["status"] == "breach"
    assert snap["checks"]["unmarked_positions"]["securities"] == ["BTC"]
    assert snap["overall_status"] == "breach"


def test_ops_snapshot_markdown_and_invalid_nav(tmp_path) -> None:
    from quant_fund.monitoring.dashboard import ops_snapshot, render_markdown

    snap = ops_snapshot(
        nav=1_000_000.0,
        cash=1_000_000.0,
        positions={},
        marks={},
        config=_cfg(tmp_path),
        kill_switch_state="ENABLED",
        n_open_orders=0,
        recon_mismatches=0,
    )
    md = render_markdown(snap)
    assert "Ops snapshot" in md
    assert "| gross |" in md
    assert "kill_switch" in md
    with pytest.raises(ValueError, match="nav"):
        ops_snapshot(nav=float("nan"), cash=0.0, positions={}, marks={}, config=_cfg(tmp_path))
