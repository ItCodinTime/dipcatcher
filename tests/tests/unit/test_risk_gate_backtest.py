"""Risk gate + kill switch must bind on the backtest fill path."""

from datetime import UTC, datetime

import polars as pl
import pytest

from quant_fund.backtest.engine import run_backtest
from quant_fund.config.loader import load_config
from quant_fund.monitoring.kill_switch import HALT_NEW_ORDERS


def _bars_weights():
    bars = pl.DataFrame(
        {
            "security_id": ["A", "A"],
            "event_time": [
                datetime(2024, 1, 1, tzinfo=UTC),
                datetime(2024, 1, 2, tzinfo=UTC),
            ],
            "open": [100.0, 100.0],
            "close": [100.0, 100.0],
            "close_total_return": [100.0, 100.0],
            "volume": [1_000_000.0] * 2,
            "adv": [100_000_000.0] * 2,
            "vol_20": [0.02] * 2,
            "source": ["file"] * 2,
        }
    )
    weights = pl.DataFrame(
        {
            "event_time": [datetime(2024, 1, 1, tzinfo=UTC)],
            "security_id": ["A"],
            "target_weight": [1.0],
        }
    )
    return bars, weights


def test_risk_gate_rejects_oversized_order(tmp_path) -> None:
    cfg = load_config("configs/research.yaml")
    cfg.data.root = tmp_path
    cfg.costs.frictionless = True
    cfg.risk_gate.max_order_notional = 50.0  # $50; target notional is ~100k
    bars, weights = _bars_weights()

    result = run_backtest(bars, weights, cfg, initial_nav=100_000.0)

    assert result.fills.height == 0
    assert int(result.metrics["risk_gate_rejects"]) >= 1


def test_kill_switch_halts_new_orders(tmp_path) -> None:
    cfg = load_config("configs/research.yaml")
    cfg.data.root = tmp_path
    cfg.costs.frictionless = True
    cfg.kill_switch.state = HALT_NEW_ORDERS
    bars, weights = _bars_weights()

    result = run_backtest(bars, weights, cfg, initial_nav=100_000.0)

    assert result.fills.height == 0
    assert int(result.metrics["kill_switch_halts"]) >= 1


def test_kill_switch_counts_every_blocked_order(tmp_path) -> None:
    """Bugbot regression: each attempted order blocked by a halt is accounted.

    The halt stops the turn, but every order it blocked must surface in
    ``kill_switch_halts`` (mirroring SimulatedBroker's per-order accounting),
    not silently vanish for all remaining names.
    """
    cfg = load_config("configs/research.yaml")
    cfg.data.root = tmp_path
    cfg.costs.frictionless = True
    cfg.kill_switch.state = HALT_NEW_ORDERS
    n_names, n_dates = 4, 3
    bars = pl.DataFrame(
        {
            "security_id": [f"S{j}" for j in range(n_names)] * n_dates,
            "event_time": [
                datetime(2024, 1, 1 + i, tzinfo=UTC) for i in range(n_dates) for _ in range(n_names)
            ],
            "open": [100.0] * (n_names * n_dates),
            "close": [100.0] * (n_names * n_dates),
            "close_total_return": [100.0] * (n_names * n_dates),
            "volume": [1_000_000.0] * (n_names * n_dates),
            "adv": [100_000_000.0] * (n_names * n_dates),
            "vol_20": [0.02] * (n_names * n_dates),
            "source": ["file"] * (n_names * n_dates),
        }
    )
    weights = pl.DataFrame(
        {
            "event_time": [
                datetime(2024, 1, 1 + i, tzinfo=UTC) for i in range(n_dates) for _ in range(n_names)
            ],
            "security_id": [f"S{j}" for _ in range(n_dates) for j in range(n_names)],
            "target_weight": [0.25] * (n_names * n_dates),
        }
    )

    result = run_backtest(bars, weights, cfg, initial_nav=100_000.0)

    assert result.fills.height == 0
    # With fill=next-open, the last decision date has no execution bar; all
    # earlier dates attempt one order per name and each is blocked.
    expected_attempts = n_names * (n_dates - 1)
    assert int(result.metrics["kill_switch_halts"]) == expected_attempts
    assert int(result.metrics["risk_gate_rejects"]) == 0


def test_check_order_unit_rejects_gross() -> None:
    from datetime import UTC, datetime

    from quant_fund.config.models import AppConfig
    from quant_fund.portfolio.risk_gate import check_order
    from quant_fund.schemas.errors import RiskGateRejected
    from quant_fund.schemas.orders import Order, OrderSide

    cfg = AppConfig()
    order = Order(
        order_id="t1",
        security_id="A",
        symbol="A",
        side=OrderSide.BUY,
        quantity=10.0,
        signal_time=datetime(2024, 1, 1, tzinfo=UTC),
        decision_time=datetime(2024, 1, 1, tzinfo=UTC),
        order_time=datetime(2024, 1, 2, tzinfo=UTC),
    )
    with pytest.raises(RiskGateRejected):
        check_order(
            order,
            nav=1_000_000.0,
            price=100.0,
            current_weight=0.0,
            gross_after=cfg.risk_gate.max_gross + 0.5,
            net_after=0.0,
            participation=0.01,
            predicted_vol=0.02,
            config=cfg,
        )
