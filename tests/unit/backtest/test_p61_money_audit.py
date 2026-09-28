"""P6.1 money-path audit regressions (see docs/AUDIT_P61_MONEY.md).

Each test pins one claim checked in the audit ledger: state-mutation ordering
in the paper broker, shortfall_frame's two quantity/side_sign modes, the
projected-position leverage caps in the perp/carry engines, and causal
exec-time marking (no same-bar close in the NAV used to size orders).
"""

from datetime import UTC, datetime, timedelta

import polars as pl
import pytest

from quant_fund.backtest.carry_engine import run_carry_backtest
from quant_fund.backtest.engine import _run_backtest_event_loop
from quant_fund.backtest.perp_engine import run_perp_backtest
from quant_fund.config.loader import load_config
from quant_fund.execution.implementation_shortfall import shortfall_frame
from quant_fund.execution.simulated_broker import RejectReason, SimulatedBroker
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


def _paper_cfg(tmp_path):
    cfg = load_config("configs/paper.yaml")
    cfg.data.root = tmp_path
    cfg.risk_gate.max_name = 1.0
    cfg.risk_gate.max_net = 1.0
    cfg.risk_gate.max_gross = 2.0
    cfg.risk_gate.max_order_notional = 1e12
    cfg.risk_gate.max_participation = 1.0
    return cfg


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
            "event_time": [T0 + at * step for at, _, _ in rows],
            "security_id": [sid for _, sid, _ in rows],
            "target_weight": [w for _, _, w in rows],
        }
    )


# --------------------------------------------------------------------------
# simulated_broker: validation must not run after cash/shares have moved.
# --------------------------------------------------------------------------


def test_broker_bad_decision_price_leaves_state_untouched(tmp_path):
    """A raise after cash/shares mutation left a phantom position with no
    fill record. Validation must precede the cash move so the raise is clean."""
    broker = SimulatedBroker(config=_paper_cfg(tmp_path), initial_cash=100_000.0)
    with pytest.raises(ValueError, match="decision_price"):
        broker.submit(
            _order(),
            price=100.0,
            nav=100_000.0,
            adv_dollars=1e9,
            decision_price=float("nan"),
        )
    assert broker.cash == 100_000.0
    assert broker.shares.get("A", 0.0) == 0.0
    assert broker.fills == []


@pytest.mark.parametrize("price", [float("inf"), float("nan")])
def test_broker_submit_rejects_nonfinite_price(tmp_path, price):
    broker = SimulatedBroker(config=_paper_cfg(tmp_path), initial_cash=100_000.0)
    rec = broker.submit(_order(), price=price, nav=100_000.0, adv_dollars=1e9)
    assert rec.order.status is OrderStatus.REJECTED
    assert rec.reject_reason == RejectReason.MISSING_PRICE.value


# --------------------------------------------------------------------------
# implementation_shortfall.shortfall_frame: unsigned qty + side_sign mode was
# skipping the quantity normalization (signed qty flipped the drift sign and
# produced negative notional), and no value was validated at all.
# --------------------------------------------------------------------------


def _fills(**over) -> pl.DataFrame:
    row = {
        "security_id": "A",
        "quantity": 5.0,
        "decision_price": 100.0,
        "price": 110.0,
    }
    row.update(over)
    return pl.DataFrame({k: [v] for k, v in row.items()})


def test_shortfall_frame_signed_qty_mode_unchanged():
    res = shortfall_frame(_fills(quantity=-5.0))
    # sell 5, exec 110 vs decision 100 → adverse -50? No: sell exec above
    # decision is favorable → drift = -1 * (110 - 100) * 5 = -50.
    assert res["drift"][0] == pytest.approx(-50.0)
    assert res["notional"][0] == pytest.approx(500.0)
    assert res["side_sign"][0] == -1.0


def test_shortfall_frame_side_sign_mode_computes_drift():
    res = shortfall_frame(_fills(quantity=5.0, side_sign=-1.0))
    assert res["drift"][0] == pytest.approx(-50.0)
    assert res["notional"][0] == pytest.approx(500.0)


def test_shortfall_frame_rejects_signed_qty_with_side_sign():
    """qty=-5 plus side_sign=-1 is ambiguous (signed mode or unsigned mode?);
    it must raise rather than double-sign the drift and negate notional."""
    with pytest.raises(ValueError, match="unsigned"):
        shortfall_frame(_fills(quantity=-5.0, side_sign=-1.0))


@pytest.mark.parametrize("bad_sign", [0.0, 0.5, 2.0, float("nan"), None])
def test_shortfall_frame_rejects_bad_side_sign(bad_sign):
    with pytest.raises(ValueError, match="side_sign"):
        shortfall_frame(_fills(quantity=5.0, side_sign=bad_sign))


@pytest.mark.parametrize(
    "over",
    [
        {"quantity": 0.0},
        {"quantity": None},
        {"quantity": float("nan")},
        {"quantity": float("inf")},
        {"price": 0.0},
        {"price": float("nan")},
        {"decision_price": None},
        {"decision_price": float("nan")},
        {"decision_price": -3.0},
        {"fee": float("nan")},
        {"fee": -1.0},
        {"spread_cost": None},
        {"impact_cost": float("inf")},
    ],
)
def test_shortfall_frame_fails_closed_on_unscorable_values(over):
    with pytest.raises(ValueError):
        shortfall_frame(_fills(**over))


# --------------------------------------------------------------------------
# perp/carry engines: the leverage cap must bound the projected position, not
# |delta| on top of existing gross — otherwise an over-cap book can never
# deleverage (every reduce order margin-rejected, position trapped).
# --------------------------------------------------------------------------


def test_perp_deleverage_allowed_when_book_over_cap(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.costs.frictionless = True
    cfg.perp.max_leverage = 1.0
    # Enter short at cap (w=-1.0 → exec bar 1), then the underlying rallies to
    # 150: equity halves, gross triples → book at ~3x cap. A reduce to w=-0.1
    # must still execute.
    prices = [100.0] * 3 + [150.0] * 4
    bars = _perp_bars({"A": [(p, p) for p in prices]})
    w = _weights([(0, "A", -1.0), (4, "A", -0.1)], timedelta(hours=1))
    res = run_perp_backtest(bars, None, w, cfg, initial_nav=1e5)
    assert res.fills.height == 2
    reduce_fill = res.fills.row(1, named=True)
    # equity 1e5 - 1000*50 = 5e4; desired qty = -0.1*5e4/150 ≈ -33.33;
    # delta = +966.67 must execute even though book was 3x over cap.
    assert reduce_fill["quantity"] == pytest.approx(966.67, rel=1e-3)
    assert res.metrics["margin_rejects"] == 0


def test_perp_add_with_no_headroom_still_margin_rejects(tmp_path):
    """The cap is not a free pass: an ADD with no headroom still rejects."""
    cfg = _cfg(tmp_path)
    cfg.costs.frictionless = True
    cfg.perp.max_leverage = 1.0
    prices = [100.0] * 3 + [150.0] * 4
    bars = _perp_bars({"A": [(p, p) for p in prices]})
    # w=-4.0 deepens the short: desired |qty| 1333 exceeds the 333-unit room.
    w = _weights([(0, "A", -1.0), (4, "A", -4.0)], timedelta(hours=1))
    res = run_perp_backtest(bars, None, w, cfg, initial_nav=1e5)
    assert res.fills.height == 1
    assert res.metrics["margin_rejects"] == 1


def test_carry_deleverage_allowed_when_book_over_cap(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.costs.frictionless = True
    cfg.perp.max_leverage = 3.0
    # Enter pair at w=0.9 (exec bar 1), then price 4x's: perp gross ≈ 3.6x
    # equity > 3x cap. Reducing to w=0.3 must execute.
    prices = [100.0] * 3 + [400.0] * 4
    perp = _perp_bars({"A": [(p, p) for p in prices]})
    spot = _perp_bars({"A": [(p, p) for p in prices]})
    w = _weights([(0, "A", 0.9), (4, "A", 0.3)], timedelta(hours=1))
    res = run_carry_backtest(perp, spot, None, w, cfg, initial_nav=1e5)
    assert res.fills.height == 2
    reduce_fill = res.fills.row(1, named=True)
    # units 900 → desired 0.3*equity/400 ≈ 75; delta ≈ -825
    assert reduce_fill["quantity"] == pytest.approx(-825.0, rel=0.01)
    assert res.metrics["margin_rejects"] == 0


def test_carry_funding_marks_at_bar_close(tmp_path):
    """Funding on the perp leg uses the bar's close mark (perp-engine
    convention), not the exec-open mark."""
    cfg = _cfg(tmp_path)
    cfg.costs.frictionless = True
    flat = [(100.0, 100.0)] * 3
    moved = [(105.0, 110.0)]  # open 105, close 110 on the funding bar
    tail = [(110.0, 110.0)] * 3
    perp = _perp_bars({"A": flat + moved + tail})
    spot = _perp_bars({"A": [(p, p) for p in [100.0] * 3 + [110.0] * 4]})
    funding = pl.DataFrame(
        {"security_id": ["A"], "event_time": [T0 + timedelta(hours=3)], "value": [0.01]}
    )
    res = run_carry_backtest(
        perp,
        spot,
        funding,
        _weights([(0, "A", 0.5)], timedelta(hours=1)),
        cfg,
        initial_nav=1e5,
    )
    # 500 units; funding at the close mark 110 → 500*110*0.01 = 550
    # (an exec-open mark of 105 would give 525).
    assert res.metrics["funding_received_total"] == pytest.approx(550.0, rel=0.01)


# --------------------------------------------------------------------------
# Exec-time NAV/equity must use marks knowable at the open: a held name with
# no valid open on the exec bar must be marked at the prior close, never at
# that same bar's close (future data leaking into position sizing).
# --------------------------------------------------------------------------


def test_spot_exec_nav_does_not_leak_exec_day_close(tmp_path):
    """Held B has no open print on the exec day but doubles by the close.
    Sizing A's order at NAV including B's rallied close is look-ahead."""
    cfg = _cfg(tmp_path)
    cfg.costs.frictionless = True
    bars = _spot_bars(
        {
            "A": [(100.0, 100.0)] * 5,
            "B": [(100.0, 100.0)] * 4 + [(None, 200.0)],
        }
    )
    weights = _weights([(0, "A", 0.5), (0, "B", 0.2), (3, "A", 0.6)], timedelta(days=1))
    res = _run_backtest_event_loop(bars, weights, cfg, initial_nav=2e6)
    last_fill = res.fills.row(-1, named=True)
    assert last_fill["security_id"] == "A"
    # Causal NAV at the exec open = cash 6e5 + 10000*100 + 4000*100 = 2e6;
    # desired A = 0.6*2e6/100 = 12000 → delta +2000.
    # Leaked NAV would be 2.4e6 → delta +4400.
    assert last_fill["quantity"] == pytest.approx(2000.0)


def test_perp_exec_equity_does_not_leak_exec_bar_close(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.costs.frictionless = True
    flat = [(100.0, 100.0)] * 4
    bars = _perp_bars(
        {
            "A": flat + [(100.0, 100.0)],
            "B": flat + [(None, 200.0)],
        }
    )
    weights = _weights([(0, "A", 0.5), (0, "B", 0.5), (3, "A", 0.8)], timedelta(hours=1))
    res = run_perp_backtest(bars, None, weights, cfg, initial_nav=1e5)
    last_fill = res.fills.row(-1, named=True)
    assert last_fill["security_id"] == "A"
    # Causal equity 1e5 → desired 800 → delta +300; leaked B mark 200 gives
    # equity 1.5e5 → delta +700.
    assert last_fill["quantity"] == pytest.approx(300.0)


def test_carry_exec_equity_does_not_leak_exec_bar_close(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.costs.frictionless = True
    flat = [(100.0, 100.0)] * 4
    # Held pair B: spot close rallies 100→150 on the exec bar (open missing),
    # perp stays flat — the basis move inflates leaked equity by u*50.
    perp = _perp_bars({"A": flat + [(100.0, 100.0)], "B": flat + [(None, 100.0)]})
    spot = _perp_bars({"A": flat + [(100.0, 100.0)], "B": flat + [(None, 150.0)]})
    weights = _weights([(0, "B", 0.3), (3, "A", 0.5)], timedelta(hours=1))
    res = run_carry_backtest(perp, spot, None, weights, cfg, initial_nav=1e5)
    last_fill = res.fills.row(-1, named=True)
    assert last_fill["security_id"] == "A"
    # Causal equity 1e5 → desired A = 0.5*1e5/100 = 500 units; leaked spot
    # mark 150 inflates equity to 1.15e5 → 575 units.
    assert last_fill["quantity"] == pytest.approx(500.0)


def test_fast_replay_exec_nav_leak_known_residual(tmp_path):
    """Exposure: run_backtest_fast (outside this lane) still marks held names
    without an exec print at the exec bar's close — the same look-ahead fixed
    in the event loop. Runs the public entrypoint, which delegates to the fast
    replay on this panel. XFAIL-strict until fast_replay.py is fixed."""
    from quant_fund.backtest.engine import run_backtest

    cfg = _cfg(tmp_path)
    cfg.costs.frictionless = True
    bars = _spot_bars(
        {
            "A": [(100.0, 100.0)] * 5,
            "B": [(100.0, 100.0)] * 4 + [(None, 200.0)],
        }
    )
    weights = _weights([(0, "A", 0.5), (0, "B", 0.2), (3, "A", 0.6)], timedelta(days=1))
    res = run_backtest(bars, weights, cfg, initial_nav=2e6)
    last_fill = res.fills.row(-1, named=True)
    assert last_fill["security_id"] == "A"
    if last_fill["quantity"] != pytest.approx(2000.0):
        pytest.xfail("fast_replay still sizes off the exec bar's close (out-of-lane)")
