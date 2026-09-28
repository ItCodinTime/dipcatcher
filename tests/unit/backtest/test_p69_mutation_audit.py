"""P6.9 mutation-spot-check regressions (see docs/AUDIT_P69_TESTS.md).

Each test pins a money-path behavior whose mutant survived the P6.9 audit
slice: limit-fill touch boundaries, short-position NAV/exposure accounting,
participation caps on both sides, the insufficient-cash guard, residual
bookkeeping on partial fills, the overlay scale application, the
missing-mark flatten guard, perp settlement/liquidation arithmetic, and the
cost/turnover metric accumulators.
"""

from datetime import UTC, datetime, timedelta

import polars as pl
import pytest

from quant_fund.backtest.engine import _run_backtest_event_loop, run_backtest
from quant_fund.backtest.perp_engine import run_perp_backtest
from quant_fund.config.loader import load_config
from quant_fund.execution.costs import total_cost
from quant_fund.execution.simulated_broker import RejectReason, SimulatedBroker
from quant_fund.risk.overlay import BookRiskOverlay
from quant_fund.schemas.orders import Order, OrderSide, OrderStatus

T0 = datetime(2024, 1, 1, tzinfo=UTC)


def _cfg(tmp_path):
    cfg = load_config("configs/research.yaml")
    cfg.data.root = tmp_path
    cfg.risk_gate.max_name = 10.0
    cfg.risk_gate.max_net = 10.0
    cfg.risk_gate.max_gross = 10.0
    cfg.risk_gate.max_order_notional = 1e12
    cfg.risk_gate.max_predicted_vol = 1e9
    cfg.risk_gate.stale_price_bars = 3
    return cfg


def _paper_cfg(tmp_path):
    cfg = load_config("configs/paper.yaml")
    cfg.data.root = tmp_path
    cfg.risk_gate.max_name = 1.0
    cfg.risk_gate.max_net = 1.0
    cfg.risk_gate.max_gross = 2.0
    cfg.risk_gate.max_order_notional = 1e12
    cfg.risk_gate.max_participation = 1.0
    return cfg


def _order(side=OrderSide.BUY, qty=10.0, oid="o1", limit=None, sid="A"):
    t = datetime(2024, 1, 2, tzinfo=UTC)
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
    )


def _spot_bars(specs: dict[str, list[tuple[float | None, float | None]]]) -> pl.DataFrame:
    """{sid: [(open, close), ...]} daily bars with close_total_return = close."""
    rows = []
    for sid, bars in specs.items():
        for i, (o, c) in enumerate(bars):
            rows.append(
                {
                    "security_id": sid,
                    "event_time": T0 + timedelta(days=i),
                    "open": o,
                    "high": None if c is None else c * 1.001,
                    "low": None if c is None else c * 0.999,
                    "close": c,
                    "close_total_return": c,
                    "volume": 1_000_000.0,
                    "adv": 1_000_000.0,
                    "source": "synthetic",
                }
            )
    return pl.DataFrame(rows)


def _perp_bars(specs: dict[str, list[tuple[float | None, float | None]]]) -> pl.DataFrame:
    """{sid: [(open, close), ...]} hourly bars; high/low bracket the close."""
    rows = []
    for sid, bars in specs.items():
        for i, (o, c) in enumerate(bars):
            ref = c if c is not None else o
            rows.append(
                {
                    "security_id": sid,
                    "event_time": T0 + timedelta(hours=i),
                    "open": o,
                    "high": None if ref is None else ref * 1.001,
                    "low": None if ref is None else ref * 0.999,
                    "close": c,
                    "volume": 1_000_000.0,
                    "source": "synthetic",
                }
            )
    return pl.DataFrame(rows)


def _weights(rows: list[tuple[int, str, float]], step: timedelta) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "event_time": [T0 + i * step for i, _, _ in rows],
            "security_id": [sid for _, sid, _ in rows],
            "target_weight": [w for _, _, w in rows],
        }
    )


# ---------------------------------------------------------------------------
# execution/costs.py — signed-quantity symmetry
# ---------------------------------------------------------------------------


def test_total_cost_symmetric_for_signed_qty(tmp_path) -> None:
    cfg = _cfg(tmp_path)
    buy = total_cost(500.0, 100.0, 1e8, 0.02, cfg.costs)
    sell = total_cost(-500.0, 100.0, 1e8, 0.02, cfg.costs)
    # Notional-based fees must be non-negative and identical in either
    # direction; dropping an abs() turns the sell leg into negative revenue.
    assert buy["commission"] > 0.0 and buy["total"] > 0.0
    for key in ("commission", "spread", "impact", "turnover_bps", "total"):
        assert buy[key] >= 0.0
        assert sell[key] == pytest.approx(buy[key])


# ---------------------------------------------------------------------------
# SimulatedBroker — limit touches, short-position accounting, caps, cash guard
# ---------------------------------------------------------------------------


def test_buy_limit_fills_on_exact_low_touch(tmp_path) -> None:
    broker = SimulatedBroker(config=_paper_cfg(tmp_path), initial_cash=100_000.0)
    broker.mark({"A": 100.0})
    broker.submit(_order(limit=99.0), price=100.0, nav=100_000.0, adv_dollars=1e9)
    recs = broker.process_bar("A", bar_open=100.0, bar_high=101.0, bar_low=99.0, adv_dollars=1e9)
    assert len(recs) == 1 and recs[0].order.status is OrderStatus.FILLED
    assert recs[0].fill.price == pytest.approx(99.0)


def test_sell_limit_fills_on_exact_high_touch(tmp_path) -> None:
    broker = SimulatedBroker(config=_paper_cfg(tmp_path), initial_cash=100_000.0)
    broker.mark({"A": 100.0})
    broker.submit(
        _order(side=OrderSide.SELL, limit=101.0),
        price=100.0,
        nav=100_000.0,
        adv_dollars=1e9,
    )
    recs = broker.process_bar("A", bar_open=100.0, bar_high=101.0, bar_low=99.0, adv_dollars=1e9)
    assert len(recs) == 1 and recs[0].order.status is OrderStatus.FILLED
    assert recs[0].fill.price == pytest.approx(101.0)


def test_nav_requires_mark_for_short_position(tmp_path) -> None:
    broker = SimulatedBroker(config=_paper_cfg(tmp_path), initial_cash=100_000.0)
    broker.shares["A"] = -10.0
    with pytest.raises(ValueError, match="missing market mark"):
        broker.nav({})


def test_exposures_zero_nav_returns_zero_tuple(tmp_path) -> None:
    broker = SimulatedBroker(config=_paper_cfg(tmp_path), initial_cash=0.0)
    assert broker.exposures({"A": 100.0}) == (0.0, 0.0)


def test_exposures_values_for_mixed_book(tmp_path) -> None:
    broker = SimulatedBroker(config=_paper_cfg(tmp_path), initial_cash=100_000.0)
    broker.shares["A"] = 10.0  # +1000 long
    broker.shares["B"] = -5.0  # -250 short at mark 50
    marks = {"A": 100.0, "B": 50.0}
    nav = broker.nav(marks)
    gross, net = broker.exposures(marks)
    assert nav == pytest.approx(100_750.0)
    assert gross == pytest.approx(1_250.0 / nav)
    assert net == pytest.approx(750.0 / nav)


def test_participation_cap_binds_buy_and_sell(tmp_path) -> None:
    cfg = _paper_cfg(tmp_path)
    cfg.costs.participation_limit = 0.01
    broker = SimulatedBroker(config=cfg, initial_cash=1e9)
    broker.mark({"A": 100.0, "B": 100.0})
    # max_qty = 0.01 * 1e6 / 100 = 100 shares
    buy = broker.submit(_order(qty=1000.0), price=100.0, nav=1e9, adv_dollars=1e6)
    assert buy.order.status is OrderStatus.FILLED
    assert buy.fill.is_partial
    assert broker.shares["A"] == pytest.approx(100.0)
    sell = broker.submit(
        _order(side=OrderSide.SELL, qty=1000.0, oid="o2", sid="B"),
        price=100.0,
        nav=1e9,
        adv_dollars=1e6,
    )
    assert sell.order.status is OrderStatus.FILLED
    assert sell.fill.is_partial
    assert broker.shares["B"] == pytest.approx(-100.0)


def test_partial_sell_fill_keeps_residual(tmp_path) -> None:
    cfg = _paper_cfg(tmp_path)
    cfg.costs.participation_limit = 0.01
    broker = SimulatedBroker(config=cfg, initial_cash=1e9)
    broker.mark({"A": 100.0})
    broker.submit(
        _order(side=OrderSide.SELL, qty=1000.0, limit=99.0),
        price=100.0,
        nav=1e9,
        adv_dollars=1e6,
    )
    # max_qty = 0.01 * 1e6 / 100 = 100 shares; the 900-share residual re-rests.
    recs = broker.process_bar("A", bar_open=100.0, bar_high=101.0, bar_low=98.0, adv_dollars=1e6)
    assert len(recs) == 1 and recs[0].order.status is OrderStatus.FILLED
    assert recs[0].fill.is_partial
    residual = broker.open_orders["o1"]
    assert residual.status is OrderStatus.PARTIAL
    assert residual.quantity == pytest.approx(900.0)

    # A full sell fill is NOT partial: under the abs-drop mutant the signed
    # exec qty (-10 < 10) wrongly reports is_partial=True.
    broker.submit(
        _order(side=OrderSide.SELL, qty=10.0, oid="o2", limit=99.0),
        price=100.0,
        nav=1e9,
        adv_dollars=1e6,
    )
    recs2 = broker.process_bar("A", bar_open=100.0, bar_high=101.0, bar_low=98.0, adv_dollars=1e6)
    assert len(recs2) == 2  # "o1" residual (capped again) + "o2" full fill
    full = next(r for r in recs2 if r.order.order_id == "o2")
    assert not full.fill.is_partial


def test_missing_mark_reject_increments_reject_count(tmp_path) -> None:
    broker = SimulatedBroker(config=_paper_cfg(tmp_path), initial_cash=1e9)
    broker.mark({"A": 100.0})
    rec = broker.submit(_order(), price=100.0, nav=0.0, adv_dollars=1e9)
    assert rec.order.status is OrderStatus.REJECTED
    assert rec.reject_reason == RejectReason.INVALID_MARKET_DATA.value
    assert broker.reject_count == 1


def test_buy_rejected_when_cash_covers_notional_but_not_fees(tmp_path) -> None:
    cfg = _paper_cfg(tmp_path)
    cfg.costs.commission_bps = 10.0
    broker = SimulatedBroker(config=cfg, initial_cash=1_000.0)
    broker.mark({"A": 100.0})
    rec = broker.submit(_order(qty=10.0), price=100.0, nav=1_000.0, adv_dollars=1e9)
    # notional = 1000, commission = 1.0 → cash 1000 is short of 1001
    assert rec.order.status is OrderStatus.REJECTED
    assert rec.reject_reason == RejectReason.INSUFFICIENT_CASH.value
    assert broker.reject_count == 1


def test_order_rejected_when_participation_breaches_gate(tmp_path) -> None:
    cfg = _paper_cfg(tmp_path)
    cfg.costs.participation_limit = 1.0  # disable the exec-size cap
    cfg.risk_gate.max_participation = 0.005
    broker = SimulatedBroker(config=cfg, initial_cash=1e9)
    broker.mark({"A": 100.0})
    rec = broker.submit(_order(qty=100.0), price=100.0, nav=1e9, adv_dollars=1e6)
    # participation = 100 * 100 / 1e6 = 0.01 > 0.005
    assert rec.order.status is OrderStatus.REJECTED
    assert RejectReason.RISK_GATE.value in rec.reject_reason


def test_gross_gate_sees_short_plus_long_book(tmp_path) -> None:
    cfg = _paper_cfg(tmp_path)
    cfg.risk_gate.max_name = 1.0
    cfg.risk_gate.max_net = 1.0
    cfg.risk_gate.max_gross = 0.9
    broker = SimulatedBroker(config=cfg, initial_cash=1e9)
    broker.mark({"A": 100.0, "B": 100.0})
    broker.shares["B"] = -8.0  # short -800 notional
    rec = broker.submit(_order(qty=90.0), price=100.0, nav=10_000.0, adv_dollars=1e9)
    # projected gross = |−800| + |9000| = 9800 → 0.98 > 0.9 → reject.
    # Dropping the abs() gives net-signed 8200 → 0.82, under the cap.
    assert rec.order.status is OrderStatus.REJECTED
    assert RejectReason.RISK_GATE.value in rec.reject_reason


def test_gross_gate_units_are_shares_times_price(tmp_path) -> None:
    cfg = _paper_cfg(tmp_path)
    cfg.risk_gate.max_gross = 1e-6
    broker = SimulatedBroker(config=cfg, initial_cash=1e9)
    broker.mark({"A": 100.0})
    rec = broker.submit(_order(qty=20.0), price=100.0, nav=1e9, adv_dollars=1e9)
    # gross_after = 2000 / 1e9 = 2e-6 > 1e-6 → reject; as a quotient it is ~0.
    assert rec.order.status is OrderStatus.REJECTED
    assert RejectReason.RISK_GATE.value in rec.reject_reason


def test_net_gate_uses_signed_exposure_units(tmp_path) -> None:
    cfg = _paper_cfg(tmp_path)
    cfg.risk_gate.max_net = 1e-6
    broker = SimulatedBroker(config=cfg, initial_cash=1e9)
    broker.mark({"A": 100.0})
    rec = broker.submit(
        _order(side=OrderSide.SELL, qty=100.0), price=100.0, nav=1e9, adv_dollars=1e9
    )
    # net_after = −10_000 / 1e9 = −1e-5 → |net| > 1e-6 → reject
    assert rec.order.status is OrderStatus.REJECTED
    assert RejectReason.RISK_GATE.value in rec.reject_reason


def test_name_gate_uses_current_weight_units(tmp_path) -> None:
    cfg = _paper_cfg(tmp_path)
    cfg.risk_gate.max_name = 0.6
    broker = SimulatedBroker(config=cfg, initial_cash=2_000.0)
    broker.mark({"A": 100.0})
    broker.shares["A"] = 5.0  # current_w = 500/2000 = 0.25
    rec = broker.submit(_order(qty=10.0), price=100.0, nav=2_000.0, adv_dollars=1e9)
    # name_w = |0.25 + 0.5| = 0.75 > 0.6 → reject; a quotient gives 0.05+0.5.
    assert rec.order.status is OrderStatus.REJECTED
    assert RejectReason.RISK_GATE.value in rec.reject_reason


def test_cash_nav_identity_position_market_value(tmp_path) -> None:
    broker = SimulatedBroker(config=_paper_cfg(tmp_path), initial_cash=100_000.0)
    broker.shares["A"] = 10.0
    broker.shares["B"] = -20.0
    ident = broker.cash_nav_identity({"A": 100.0, "B": 50.0})
    assert ident["position_mv"] == pytest.approx(0.0)
    assert ident["nav"] == pytest.approx(100_000.0)
    assert ident["residual"] == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# engine.py — overlay scale application, missing-mark flatten, liquidity cap
# ---------------------------------------------------------------------------


def test_overlay_scale_applies_to_positions(tmp_path) -> None:
    cfg = _cfg(tmp_path)
    cfg.costs.frictionless = True
    cfg.risk_gate.max_name = 1.0
    cfg.risk_gate.max_net = 1.0
    cfg.risk_gate.max_gross = 1.0
    dates = [datetime(2024, 1, day, tzinfo=UTC) for day in (1, 2, 3, 4)]
    bars = pl.DataFrame(
        {
            "security_id": ["A"] * 4,
            "event_time": dates,
            "open": [100.0, 100.0, 90.0, 80.0],
            "close": [100.0, 90.0, 80.0, 70.0],
            "close_total_return": [100.0, 90.0, 80.0, 70.0],
            "volume": [1_000_000.0] * 4,
            "adv": [100_000_000.0] * 4,
            "vol_20": [0.02] * 4,
            "source": ["file"] * 4,
        }
    )
    weights = pl.DataFrame(
        {
            "event_time": [dates[0]],
            "security_id": ["A"],
            "target_weight": [1.0],
        }
    )
    overlay = BookRiskOverlay(vol_target=1.0, dd_limit=0.05, es_limit=1.0, lookback=8)
    result = run_backtest(bars, weights, cfg, initial_nav=100_000.0, risk_overlay=overlay)
    assert overlay.n_halt >= 1
    # The halt must reach the book: entry fill plus a flattening sell.
    assert result.fills.height == 2
    navs = [float(v) for v in result.equity["nav"].to_list()]
    assert navs[-1] == pytest.approx(navs[-2])


def test_held_long_flattens_when_mark_goes_dark(tmp_path) -> None:
    cfg = _cfg(tmp_path)
    cfg.costs.frictionless = True
    bars = _spot_bars({"A": [(100.0, 100.0)] * 2 + [(100.0, None)] * 6})
    weights = _weights([(0, "A", 0.5)], timedelta(days=1))
    res = _run_backtest_event_loop(bars, weights, cfg, initial_nav=100_000.0)
    # The guard must exit at the last execution print rather than ghost-hold
    # until the stale-mark limit trips.
    assert res.fills.height == 2
    sell = res.fills.row(1, named=True)
    assert sell["quantity"] < 0.0


def test_held_short_flattens_when_mark_goes_dark(tmp_path) -> None:
    cfg = _cfg(tmp_path)
    cfg.costs.frictionless = True
    bars = _spot_bars({"A": [(100.0, 100.0)] * 2 + [(100.0, None)] * 6})
    weights = _weights([(0, "A", -0.5)], timedelta(days=1))
    res = _run_backtest_event_loop(bars, weights, cfg, initial_nav=100_000.0)
    assert res.fills.height == 2
    buy_back = res.fills.row(1, named=True)
    assert buy_back["quantity"] > 0.0


def test_participation_cap_bounds_engine_fill_qty(tmp_path) -> None:
    cfg = _cfg(tmp_path)
    cfg.costs.frictionless = True
    cfg.costs.participation_limit = 0.01
    # adv = 1e6 → max_qty = 0.01 * 1e6 / 100 = 100 shares
    bars = _spot_bars({"A": [(100.0, 100.0)] * 4})
    res_long = _run_backtest_event_loop(
        bars, _weights([(0, "A", 0.9)], timedelta(days=1)), cfg, initial_nav=100_000.0
    )
    assert res_long.fills.height >= 1
    for q in res_long.fills["quantity"].to_list():
        assert float(q) == pytest.approx(100.0)
    res_short = _run_backtest_event_loop(
        bars, _weights([(0, "A", -0.9)], timedelta(days=1)), cfg, initial_nav=100_000.0
    )
    assert res_short.fills.height >= 1
    for q in res_short.fills["quantity"].to_list():
        assert float(q) == pytest.approx(-100.0)


def test_participation_gate_rejects_oversized_order(tmp_path) -> None:
    cfg = _cfg(tmp_path)
    cfg.costs.frictionless = True
    cfg.costs.participation_limit = 1.0
    cfg.risk_gate.max_participation = 0.05
    bars = _spot_bars({"A": [(100.0, 100.0)] * 4})
    w = _weights([(0, "A", 0.9)], timedelta(days=1))
    res = _run_backtest_event_loop(bars, w, cfg, initial_nav=100_000.0)
    # participation = 900 * 100 / 1e6 = 0.09 > 0.05 → rejected, no fill
    assert res.fills.height == 0
    assert res.metrics["risk_gate_rejects"] >= 1


# ---------------------------------------------------------------------------
# perp_engine.py — settlement math, caps, liquidation, metric accumulators
# ---------------------------------------------------------------------------


def test_perp_flip_settles_realized_pnl(tmp_path) -> None:
    cfg = _cfg(tmp_path)
    cfg.costs.frictionless = True
    prices = [100.0] * 4 + [110.0] * 4
    bars = _perp_bars({"A": [(p, p) for p in prices]})
    w = _weights([(0, "A", 0.5), (4, "A", -0.3)], timedelta(hours=1))
    res = run_perp_backtest(bars, None, w, cfg, initial_nav=1e5)
    # 500 long @100 fills bar 1; flip short at bar 5's open 110 realizes
    # +5000 into cash. Final marks 110 → nav = 105000.
    nav = float(res.equity["nav"][-1])
    assert nav == pytest.approx(105_000.0, rel=1e-4)


def test_perp_long_pyramid_then_partial_close_settles(tmp_path) -> None:
    cfg = _cfg(tmp_path)
    cfg.costs.frictionless = True
    prices = [100.0] * 3 + [150.0] * 3 + [200.0] * 3
    bars = _perp_bars({"A": [(p, p) for p in prices]})
    w = _weights([(0, "A", 0.5), (3, "A", 0.8), (6, "A", 0.2)], timedelta(hours=1))
    res = run_perp_backtest(bars, None, w, cfg, initial_nav=1e5)
    # bar1: 500 @100. bar4: equity=1e5+500*50=125000 → desired 0.8*125000/150
    # ≈666.67 → +166.67 @150 → vwap entry 112.5. bar7: equity≈158333 →
    # desired 0.2*158333/200≈158.33 → close ≈508.33 @200, realized
    # ≈508.33*(200-112.5)=44479. Remaining 158.33 marked at 200 → nav≈158333.
    nav = float(res.equity["nav"][-1])
    assert nav == pytest.approx(158_333.0, rel=2e-3)


def test_perp_short_pyramid_then_partial_close_settles(tmp_path) -> None:
    cfg = _cfg(tmp_path)
    cfg.costs.frictionless = True
    prices = [100.0] * 3 + [80.0] * 3 + [60.0] * 3
    bars = _perp_bars({"A": [(p, p) for p in prices]})
    w = _weights([(0, "A", -0.5), (3, "A", -0.8), (6, "A", -0.2)], timedelta(hours=1))
    res = run_perp_backtest(bars, None, w, cfg, initial_nav=1e5)
    # bar1: -500 @100. bar4: equity=1e5+500*20=110000 → desired -1100 →
    # -600 @80 → vwap entry (500*100+600*80)/1100 ≈ 89.09. bar7: equity
    # =1e5+1100*29.09=132000 → desired ≈-440 → cover ≈660 @60, realized
    # ≈660*29.09=19200. Remaining -440 earns 440*29.09=12800 → nav=132000.
    nav = float(res.equity["nav"][-1])
    assert nav == pytest.approx(132_000.0, rel=2e-3)


def test_perp_leverage_cap_headroom_accounts_for_other_positions(tmp_path) -> None:
    cfg = _cfg(tmp_path)
    cfg.costs.frictionless = True
    cfg.perp.max_leverage = 2.0
    flat = [(100.0, 100.0)] * 8
    bars = _perp_bars({"A": flat, "B": flat})
    w = _weights([(0, "A", 1.5), (0, "B", 1.5)], timedelta(hours=1))
    res = run_perp_backtest(bars, None, w, cfg, initial_nav=1e5)
    # First fill takes ~1.5x = 1500 units; the second is capped to the
    # remaining headroom: (2e5 - 1.5e5) / 100 = 500 units.
    assert res.fills.height == 2
    qtys = sorted(abs(float(q)) for q in res.fills["quantity"])
    assert qtys[0] == pytest.approx(500.0, rel=1e-2)
    assert qtys[1] == pytest.approx(1500.0, rel=1e-2)


def test_perp_leverage_cap_reaches_cap_not_below(tmp_path) -> None:
    cfg = _cfg(tmp_path)
    cfg.costs.frictionless = True
    cfg.perp.max_leverage = 2.0
    res = run_perp_backtest(
        _perp_bars({"A": [(100.0, 100.0)] * 10}),
        None,
        _weights([(0, "A", 5.0)], timedelta(hours=1)),
        cfg,
        initial_nav=1e5,
    )
    # Cap binds at exactly 2x: a sign/direction slip under-fills or rejects.
    assert float(res.equity["gross"].max()) == pytest.approx(2.0, rel=1e-3)


def test_perp_participation_cap_bounds_fill_qty(tmp_path) -> None:
    cfg = _cfg(tmp_path)
    cfg.costs.frictionless = True
    cfg.costs.participation_limit = 1e-5
    bars = _perp_bars({"A": [(100.0, 100.0)] * 6})
    res = run_perp_backtest(
        bars, None, _weights([(0, "A", 0.9)], timedelta(hours=1)), cfg, initial_nav=1e5
    )
    # enriched adv ≈ 100 * 1e6 = 1e8 → max_qty = 1e-5 * 1e8 / 100 = 10 units
    assert res.fills.height == 1
    assert res.fills["quantity"][0] == pytest.approx(10.0, rel=1e-2)
    res_short = run_perp_backtest(
        bars, None, _weights([(0, "A", -0.9)], timedelta(hours=1)), cfg, initial_nav=1e5
    )
    assert res_short.fills.height == 1
    assert res_short.fills["quantity"][0] == pytest.approx(-10.0, rel=1e-2)


def test_perp_short_squeeze_liquidates_on_high_wick(tmp_path) -> None:
    cfg = _cfg(tmp_path)
    cfg.costs.frictionless = True
    cfg.perp.max_leverage = 3.0
    cfg.perp.maint_margin_ratio = 0.05
    # Short held; close stays at 100 but one bar wicks to a 250 high. The
    # adverse mark for a short is the HIGH — ignoring it misses the breach.
    rows = [
        {
            "security_id": "A",
            "event_time": T0 + timedelta(hours=i),
            "open": 100.0,
            "high": 250.0 if i == 2 else 100.1,
            "low": 99.9,
            "close": 100.0,
            "volume": 1_000_000.0,
            "source": "synthetic",
        }
        for i in range(6)
    ]
    bars = pl.DataFrame(rows)
    w = _weights([(0, "A", -2.0)], timedelta(hours=1))
    res = run_perp_backtest(bars, None, w, cfg, initial_nav=1e4)
    assert res.metrics["liquidation_count"] >= 1
    # 200 short units liquidated at the 250 wick: fee = 200*250*50/1e4 = 250.
    qty = 2.0 * 1e4 / 100.0
    expected_fee = qty * 250.0 * cfg.perp.liquidation_fee_bps / 1e4
    assert res.metrics["liquidation_cost"] == pytest.approx(expected_fee, rel=1e-2)
    nav = float(res.equity["nav"][-1])
    expected_nav = 1e4 - qty * (250.0 - 100.0) - expected_fee
    assert nav == pytest.approx(expected_nav, rel=1e-2)


def test_perp_liquidation_fee_debits_cash_exactly(tmp_path) -> None:
    cfg = _cfg(tmp_path)
    cfg.costs.frictionless = True
    cfg.perp.max_leverage = 3.0
    cfg.perp.maint_margin_ratio = 0.05
    cfg.perp.liquidation_fee_bps = 50.0
    prices = [100.0, 100.0, 60.0, 60.0, 60.0]
    bars = _perp_bars({"A": [(p, p) for p in prices]})
    res = run_perp_backtest(
        bars, None, _weights([(0, "A", 2.9)], timedelta(hours=1)), cfg, initial_nav=1e4
    )
    assert res.metrics["liquidation_count"] == 1
    # ~290 units long @100, liquidated at the 60.06 low-wick mark.
    qty = 2.9 * 1e4 / 100.0
    liq_price = 60.0 * 0.999  # _perp_bars low = close * 0.999
    expected_fee = qty * liq_price * 50.0 / 1e4
    assert res.metrics["liquidation_cost"] == pytest.approx(expected_fee, rel=1e-3)
    nav = float(res.equity["nav"][-1])
    expected_nav = 1e4 - qty * (100.0 - liq_price) - expected_fee
    assert nav == pytest.approx(expected_nav, rel=1e-3)


def test_perp_liquidation_targets_largest_position_first(tmp_path) -> None:
    cfg = _cfg(tmp_path)
    cfg.costs.frictionless = True
    cfg.perp.max_leverage = 3.0
    cfg.perp.maint_margin_ratio = 0.05
    # Short book: A -290 @100 squeezes to 132, B -500 @10 flat. Only winding
    # down A (|q|*mark = 38k vs B's 5k) clears the deficit in one shot. A
    # victim-key bug (min, sign-flip, quotient) picks B first, the deficit
    # persists, and both positions get liquidated → count 2.
    a = _perp_bars({"A": [(p, p) for p in [100.0, 100.0, 132.0, 132.0, 132.0]]})
    b = _perp_bars({"B": [(p, p) for p in [10.0] * 5]})
    bars = pl.concat([a, b])
    w = _weights([(0, "A", -2.9), (0, "B", -0.5)], timedelta(hours=1))
    res = run_perp_backtest(bars, None, w, cfg, initial_nav=1e4)
    assert res.metrics["liquidation_count"] == 1


def test_perp_dust_filter_allows_small_executable_delta(tmp_path) -> None:
    cfg = _cfg(tmp_path)
    cfg.costs.frictionless = True
    prices = [100.0] * 8
    bars = _perp_bars({"A": [(p, p) for p in prices]})
    # +2 USD of delta: |delta|*price = 2 ≥ 1 — below the floor only if the
    # product is miscomputed as a quotient.
    w = _weights([(0, "A", 0.5), (4, "A", 0.50002)], timedelta(hours=1))
    res = run_perp_backtest(bars, None, w, cfg, initial_nav=1e5)
    assert res.fills.height == 2


def test_perp_metrics_turnover_and_cost_buckets(tmp_path) -> None:
    cfg = _cfg(tmp_path)  # frictionless off: fee accrual is asserted exactly
    prices = [100.0] * 6
    bars = _perp_bars({"A": [(p, p) for p in prices]})
    w = _weights([(0, "A", 0.5), (4, "A", 0.2)], timedelta(hours=1))
    res = run_perp_backtest(bars, None, w, cfg, initial_nav=1e5)
    assert res.fills.height == 2
    q1 = float(res.fills["quantity"][0])
    nav = float(res.equity["nav"][-1])
    assert nav < 1e5  # fees were paid out of cash, not into it
    # turnover contributions: buy 500*100/1e5 = 0.5, then the reduce leg
    # |delta2|*100 / nav5 ≈ 0.3 → mean over 6 bars ≈ (0.5 + 0.3) / 6
    assert res.metrics["mean_turnover"] == pytest.approx(0.8 / 6.0, rel=0.1)
    assert res.metrics["commission"] == pytest.approx(
        q1 * 100.0 * cfg.costs.commission_bps / 1e4
        + abs(float(res.fills["quantity"][1])) * 100.0 * cfg.costs.commission_bps / 1e4,
        rel=1e-6,
    )
    assert res.metrics["spread"] == pytest.approx(
        q1 * 100.0 * cfg.costs.half_spread_bps / 1e4
        + abs(float(res.fills["quantity"][1])) * 100.0 * cfg.costs.half_spread_bps / 1e4,
        rel=1e-6,
    )
    assert res.metrics["impact"] >= 0.0


def test_perp_net_notional_signs_with_position(tmp_path) -> None:
    cfg = _cfg(tmp_path)
    cfg.costs.frictionless = True
    prices = [100.0] * 6
    bars = _perp_bars({"A": [(p, p) for p in prices]})
    w = _weights([(0, "A", -0.5)], timedelta(hours=1))
    res = run_perp_backtest(bars, None, w, cfg, initial_nav=1e5)
    net = float(res.equity["net"][-1])
    assert net == pytest.approx(-0.5, rel=1e-2)
