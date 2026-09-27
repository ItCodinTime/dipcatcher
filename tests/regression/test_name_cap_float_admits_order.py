"""A target sized exactly to the name cap must not die on a float ulp."""

from __future__ import annotations

from datetime import UTC, datetime

import polars as pl
import pytest

from quant_fund.backtest.engine import run_backtest
from quant_fund.backtest.fast_replay import _exceeds_nb, run_backtest_fast
from quant_fund.config.models import AppConfig
from quant_fund.portfolio.risk_gate import check_order, exceeds_limit
from quant_fund.schemas.errors import RiskGateRejected
from quant_fund.schemas.orders import Order, OrderSide


def _config() -> AppConfig:
    cfg = AppConfig()
    cfg.data.root = "/tmp/dipcatcher-name-cap-ulp"
    cfg.costs.frictionless = True
    cfg.costs.borrow_bps_per_year = 0.0
    cfg.costs.participation_limit = 1.0
    cfg.risk_gate.max_name = 1.0
    cfg.risk_gate.max_net = 1.0
    cfg.risk_gate.max_gross = 1.0
    cfg.risk_gate.max_order_notional = 1e15
    cfg.risk_gate.max_participation = 1.0
    cfg.risk_gate.max_predicted_vol = 100.0
    return cfg


def _bars() -> pl.DataFrame:
    # 1_000_000 / 102.34375 * 102.34375 is 1 ulp above 1.0.
    t0 = datetime(2024, 1, 2, tzinfo=UTC)
    t1 = datetime(2024, 1, 3, tzinfo=UTC)
    t2 = datetime(2024, 1, 4, tzinfo=UTC)
    px = [100.0, 102.34375, 103.94287109375]
    return pl.DataFrame(
        {
            "security_id": ["S0", "S0", "S0"],
            "event_time": [t0, t1, t2],
            "open": px,
            "close": px,
            "close_total_return": px,
            "volume": [1_000_000.0, 1_000_000.0, 1_000_000.0],
            "adv": [1e12, 1e12, 1e12],
            "vol_20": [0.02, 0.02, 0.02],
            "source": ["file", "file", "file"],
        }
    ).with_columns(pl.col("event_time").cast(pl.Datetime("us", "UTC")))


def _weights() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "event_time": [datetime(2024, 1, 2, tzinfo=UTC)],
            "security_id": ["S0"],
            "target_weight": [1.0],
        }
    ).with_columns(pl.col("event_time").cast(pl.Datetime("us", "UTC")))


def test_full_investment_at_name_cap_is_filled() -> None:
    initial = 1_000_000.0
    ref = run_backtest(_bars(), _weights(), _config(), initial_nav=initial)
    fast = run_backtest_fast(_bars(), _weights(), _config(), initial_nav=initial)
    assert ref.metrics["risk_gate_rejects"] == 0
    assert ref.metrics["cash_rejects"] == 0
    assert ref.fills.height == 1
    assert float(ref.equity["nav"][0]) == pytest.approx(initial)
    # Held through the next total-return close: 103.94287109375 / 102.34375.
    assert float(ref.equity["nav"][1]) == pytest.approx(initial * 1.015625)
    assert fast.equity["nav"].to_list() == pytest.approx(ref.equity["nav"].to_list())
    assert fast.metrics["risk_gate_rejects"] == ref.metrics["risk_gate_rejects"]


def test_gate_admits_one_ulp_and_rejects_a_real_breach() -> None:
    cfg = _config()
    now = datetime(2024, 1, 2, tzinfo=UTC)
    order = Order(
        order_id="o",
        security_id="S0",
        symbol="S0",
        side=OrderSide.BUY,
        quantity=1.0,
        signal_time=now,
        decision_time=now,
        order_time=now,
    )
    noisy = 1.0000000000000002
    check_order(
        order,
        nav=1.0,
        price=1.0,
        current_weight=0.0,
        gross_after=noisy,
        net_after=noisy,
        participation=0.01,
        predicted_vol=0.02,
        config=cfg,
    )
    over = order.model_copy(update={"quantity": 1.0 + 1e-6})
    with pytest.raises(RiskGateRejected, match="name weight"):
        check_order(
            over,
            nav=1.0,
            price=1.0,
            current_weight=0.0,
            gross_after=1.0,
            net_after=1.0,
            participation=0.01,
            predicted_vol=0.02,
            config=cfg,
        )
    assert exceeds_limit(noisy, 1.0) is False
    assert exceeds_limit(1.0 + 1e-6, 1.0) is True
    assert _exceeds_nb(noisy, 1.0) is False
    assert _exceeds_nb(1.0 + 1e-6, 1.0) is True
