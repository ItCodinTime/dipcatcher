"""Adapter over ``quant_fund.backtest.engine.run_backtest`` (NEXT_OPEN fills).

The parity config relaxes the pre-trade risk gate (the strategies are
long-only and sized to leave a cash buffer) and disables the sqrt-impact /
spread / borrow frictions so comparisons isolate *convention* differences —
fill timing, sizing NAV basis, rounding — rather than cost-model choices.
A flat ``commission_bps`` can be layered on for the costed scenario.
"""

from __future__ import annotations

import polars as pl

from quant_fund.backtest.engine import run_backtest
from quant_fund.config.models import (
    AppConfig,
    CostConfig,
    ExecutionConfig,
    FillConvention,
    RiskGateConfig,
)
from quant_fund.research.parity.canonical import (
    EngineRun,
    marks_from_bars,
    positions_from_trades,
)
from quant_fund.research.parity.strategies import StrategyPlan

ENGINE = "native"


def parity_config(*, commission_bps: float = 0.0) -> AppConfig:
    """NEXT_OPEN, no spread/impact/borrow, gate wide enough for a 95% book."""
    return AppConfig(
        costs=CostConfig(
            commission_bps=commission_bps,
            half_spread_bps=0.0,
            impact_y=0.0,
            bps_per_turnover=0.0,
            borrow_bps_per_year=0.0,
            frictionless=commission_bps == 0.0,
            participation_limit=1.0,
        ),
        risk_gate=RiskGateConfig(
            max_order_notional=1e15,
            max_gross=1e6,
            max_net=1e6,
            max_name=1.0,
            max_participation=1.0,
            max_predicted_vol=1e9,
            stale_price_bars=1_000_000,
            stale_model_hours=1e9,
        ),
        execution=ExecutionConfig(fill=FillConvention.NEXT_OPEN),
    )


def run(
    panel: pl.DataFrame,
    plan: StrategyPlan,
    *,
    initial_nav: float = 1_000_000.0,
    commission_bps: float = 0.0,
) -> EngineRun:
    cfg = parity_config(commission_bps=commission_bps)
    result = run_backtest(panel, plan.weights, cfg, initial_nav=initial_nav)
    fills = result.fills
    if fills.height:
        trades = fills.select(
            pl.lit(ENGINE).alias("engine"),
            pl.lit(plan.name).alias("strategy"),
            "security_id",
            "fill_time",
            pl.col("quantity").sign().alias("side"),
            pl.col("quantity").abs().alias("quantity"),
            "price",
            "fee",
        ).cast(
            {
                "engine": pl.String,
                "strategy": pl.String,
                "security_id": pl.String,
                "fill_time": pl.Datetime("us", "UTC"),
                "side": pl.Float64,
                "quantity": pl.Float64,
                "price": pl.Float64,
                "fee": pl.Float64,
            }
        )
    else:
        from quant_fund.research.parity.canonical import empty_trades

        trades = empty_trades()
    positions = positions_from_trades(
        trades, marks_from_bars(panel), engine=ENGINE, strategy=plan.name
    )
    pos_mv = (
        positions.group_by("event_time").agg(pl.col("market_value").sum().alias("mv"))
        if positions.height
        else pl.DataFrame(
            schema={"event_time": pl.Datetime("us", "UTC"), "mv": pl.Float64}
        )
    )
    equity = (
        result.equity.select("event_time", "nav")
        .join(pos_mv, on="event_time", how="left")
        .with_columns(pl.col("mv").fill_null(0.0))
        .select(
            pl.lit(ENGINE).alias("engine"),
            pl.lit(plan.name).alias("strategy"),
            "event_time",
            "nav",
            (pl.col("nav") - pl.col("mv")).alias("cash"),
        )
        .cast(
            {
                "engine": pl.String,
                "strategy": pl.String,
                "event_time": pl.Datetime("us", "UTC"),
                "nav": pl.Float64,
                "cash": pl.Float64,
            }
        )
    )
    notes = [
        f"risk_gate_rejects={result.metrics.get('risk_gate_rejects')}",
        f"cash_rejects={result.metrics.get('cash_rejects')}",
        f"kill_switch_halts={result.metrics.get('kill_switch_halts')}",
        "label=SYNTHETIC research-only",
    ]
    return EngineRun(
        engine=ENGINE,
        strategy=plan.name,
        status="ok",
        trades=trades,
        positions=positions,
        equity=equity,
        notes=notes,
        metrics=dict(result.metrics),
    )
