"""Execution assumptions that can reverse a backtest's apparent capacity."""

from datetime import UTC, datetime, timedelta

import polars as pl
import pytest

from quant_fund.backtest.engine import capacity_sensitivity, run_backtest
from quant_fund.backtest.fast_replay import run_backtest_fast
from quant_fund.config.loader import load_config


def _config(tmp_path):
    cfg = load_config("configs/research.yaml")
    cfg.data.root = tmp_path
    cfg.risk_gate.max_name = 1.0
    cfg.risk_gate.max_net = 1.0
    cfg.risk_gate.max_gross = 1.0
    cfg.risk_gate.max_order_notional = 1e12
    cfg.risk_gate.max_participation = 1.0
    cfg.risk_gate.max_predicted_vol = 1.0
    cfg.costs.participation_limit = 0.1
    return cfg


def _bars(signal_adv, fill_adv, *, signal_vol=0.02, fill_vol=0.9):
    dates = [datetime(2024, 1, 1, tzinfo=UTC) + timedelta(days=i) for i in range(3)]
    return pl.DataFrame(
        {
            "security_id": ["A"] * 3,
            "event_time": dates,
            "open": [100.0] * 3,
            "close": [100.0] * 3,
            "close_total_return": [100.0] * 3,
            "volume": [10.0] * 3,
            "adv": [signal_adv, fill_adv, 1_000_000.0],
            "vol_20": [signal_vol, fill_vol, 0.9],
            "source": ["file"] * 3,
        }
    ), pl.DataFrame({"event_time": [dates[0]], "security_id": ["A"], "target_weight": [1.0]})


@pytest.mark.parametrize("engine", [run_backtest, run_backtest_fast])
def test_next_open_uses_prior_close_liquidity_and_volatility(tmp_path, engine):
    cfg = _config(tmp_path)
    bars, weights = _bars(1_000.0, 1_000_000_000.0)
    result = engine(bars, weights, cfg, initial_nav=100_000)
    assert result.fills.height >= 1
    assert result.fills["quantity"][0] == pytest.approx(1.0)
    # Impact for a $100 trade against $1000 ADV and 2% forecast volatility.
    assert result.fills["impact_cost"][0] == pytest.approx(100 * 0.1 * 0.02 * 0.1**0.5)
    changed = bars.with_columns(
        pl.when(pl.col("event_time") == bars["event_time"][1])
        .then(pl.lit(1.0))
        .otherwise(pl.col("adv"))
        .alias("adv"),
        pl.when(pl.col("event_time") == bars["event_time"][1])
        .then(pl.lit(0.99))
        .otherwise(pl.col("vol_20"))
        .alias("vol_20"),
    )
    result_changed = engine(changed, weights, cfg, initial_nav=100_000)
    assert result_changed.fills["quantity"][0] == pytest.approx(result.fills["quantity"][0])
    assert result_changed.fills["impact_cost"][0] == pytest.approx(result.fills["impact_cost"][0])


@pytest.mark.parametrize("engine", [run_backtest, run_backtest_fast])
def test_missing_asof_adv_cannot_use_future_volume(tmp_path, engine):
    cfg = _config(tmp_path)
    bars, weights = _bars(None, 1_000_000_000.0)
    result = engine(bars.slice(0, 2), weights, cfg, initial_nav=100_000)
    assert result.fills.is_empty()
    assert result.metrics["risk_gate_rejects"] == 1


@pytest.mark.parametrize("engine", [run_backtest, run_backtest_fast])
def test_runtime_zero_participation_never_opens_book(tmp_path, engine):
    cfg = _config(tmp_path)
    # Assignment can bypass CostConfig's construction-time Pydantic validator.
    cfg.costs.participation_limit = 0
    bars, weights = _bars(1_000_000.0, 1_000_000.0)
    result = engine(bars, weights, cfg, initial_nav=100_000)
    assert result.fills.is_empty()
    assert result.metrics["risk_gate_rejects"] == 2


def test_capacity_sensitivity_replays_partial_fills_at_each_scale(tmp_path):
    cfg = _config(tmp_path)
    bars, weights = _bars(1_000.0, 1_000_000_000.0)
    report = capacity_sensitivity(
        bars.slice(0, 2), weights, cfg, nav_levels=(1_000.0, 10_000.0), adv_haircuts=(1.0, 0.5)
    )
    assert report["research_only"] is True
    cases = report["cases"]
    assert isinstance(cases, list)
    assert len(cases) == 4
    assert [case["filled_orders"] for case in cases] == [1, 1, 1, 1]
    assert [case["executed_notional_fraction"] for case in cases] == pytest.approx(
        [0.1, 0.05, 0.01, 0.005]
    )
    for case in cases:
        assert case["total_return"] == pytest.approx(case["end_nav"] / case["initial_nav"] - 1)


@pytest.mark.parametrize("haircut", [0.0, 1.1])
def test_capacity_sensitivity_rejects_invalid_stresses(tmp_path, haircut):
    cfg = _config(tmp_path)
    bars, weights = _bars(1_000.0, 1_000.0)
    with pytest.raises(ValueError, match="adv_haircuts"):
        capacity_sensitivity(bars, weights, cfg, adv_haircuts=(haircut,))
