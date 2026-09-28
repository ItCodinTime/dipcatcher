"""Latency and impact grid for the execution simulator.

Execution diagnostic only. Net P&L and Sharpe in the table describe how the
simulated path degrades as latency and impact change. They are not research
headlines and not a live P&L claim.
"""

from __future__ import annotations

import math
from datetime import timedelta

import polars as pl

from quant_fund.backtest.event_sim.simulator import EventSimSpec, run_event_backtest
from quant_fund.config.models import AppConfig


def _cell(
    bars: pl.DataFrame,
    weights: pl.DataFrame,
    config: AppConfig,
    *,
    signal_bars: int,
    exchange_bars: int,
    impact_multiplier: float,
    initial_nav: float,
    seed: int,
) -> dict[str, object]:
    scenario = config.model_copy(deep=True)
    scenario.costs.frictionless = False
    scenario.costs.impact_y = float(config.costs.impact_y) * float(impact_multiplier)
    spec = EventSimSpec(
        signal_to_order_bars=signal_bars,
        order_to_exchange_bars=exchange_bars,
        seed=seed,
        fee_schedule="bps",
        pdt_mode="warn",
        gfv_mode="warn",
    )
    out = run_event_backtest(bars, weights, scenario, spec, initial_nav=initial_nav)
    metrics = out.result.metrics
    end_nav = (
        float(out.result.equity.get_column("nav")[-1])
        if out.result.equity.height
        else float(initial_nav)
    )
    net_pnl = end_nav - float(initial_nav)
    sharpe = metrics.get("sharpe", float("nan"))
    sharpe_value = float(sharpe) if isinstance(sharpe, (int, float)) else float("nan")
    costs = [
        float(value)
        for name in ("commission", "spread", "impact")
        if isinstance((value := metrics.get(name, 0.0)), (int, float))
    ]
    return {
        "signal_to_order_bars": signal_bars,
        "order_to_exchange_bars": exchange_bars,
        "signal_to_order": str(timedelta(0)),
        "order_to_exchange": str(timedelta(0)),
        "impact_multiplier": float(impact_multiplier),
        "net_pnl": float(net_pnl),
        "sharpe_diagnostic": sharpe_value,
        "total_return": metrics.get("total_return", 0.0),
        "total_cost": float(sum(costs)),
        "end_equity": end_nav,
        "source": out.result.source_note,
        "claim": "execution_diagnostic_only",
        "research_only": True,
        "live_pnl_claim": False,
    }


def execution_sensitivity(
    bars: pl.DataFrame,
    weights: pl.DataFrame,
    config: AppConfig,
    *,
    signal_latencies: tuple[int, ...] = (0, 1),
    exchange_latencies: tuple[int, ...] = (0,),
    impact_multipliers: tuple[float, ...] = (1.0, 2.0),
    initial_nav: float = 1_000_000.0,
    seed: int = 0,
) -> dict[str, object]:
    """Rerun one strategy across latency and square-root impact settings.

    The baseline cell is zero signal latency, zero exchange latency, and the
    smallest impact multiplier. Deltas are that cell minus the scenario
    (positive means the scenario made less).
    """
    if not signal_latencies or any(int(item) < 0 for item in signal_latencies):
        raise ValueError("signal_latencies must contain non-negative bar counts")
    if not exchange_latencies or any(int(item) < 0 for item in exchange_latencies):
        raise ValueError("exchange_latencies must contain non-negative bar counts")
    if not impact_multipliers or any(
        not math.isfinite(item) or item < 0.0 for item in impact_multipliers
    ):
        raise ValueError("impact_multipliers must contain finite non-negative values")
    rows: list[dict[str, object]] = []
    for signal_bars in signal_latencies:
        for exchange_bars in exchange_latencies:
            for multiplier in impact_multipliers:
                rows.append(
                    _cell(
                        bars,
                        weights,
                        config,
                        signal_bars=int(signal_bars),
                        exchange_bars=int(exchange_bars),
                        impact_multiplier=float(multiplier),
                        initial_nav=initial_nav,
                        seed=seed,
                    )
                )

    def _cell_float(row: dict[str, object], key: str) -> float:
        value = row[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{key} must be numeric")
        return float(value)

    baseline = min(
        rows,
        key=lambda row: (
            _cell_float(row, "signal_to_order_bars"),
            _cell_float(row, "order_to_exchange_bars"),
            _cell_float(row, "impact_multiplier"),
        ),
    )
    base_pnl = _cell_float(baseline, "net_pnl")
    base_sharpe = _cell_float(baseline, "sharpe_diagnostic")
    for row in rows:
        row["net_pnl_delta"] = _cell_float(row, "net_pnl") - base_pnl
        sharpe = _cell_float(row, "sharpe_diagnostic")
        row["sharpe_delta"] = (
            sharpe - base_sharpe
            if math.isfinite(sharpe) and math.isfinite(base_sharpe)
            else float("nan")
        )
    return {
        "claim": "execution_diagnostic_only",
        "research_only": True,
        "live_pnl_claim": False,
        "role": "execution_diagnostic",
        "baseline_impact_y": float(config.costs.impact_y),
        "initial_nav": float(initial_nav),
        "seed": int(seed),
        "rows": rows,
    }


def format_sensitivity_table(report: dict[str, object]) -> str:
    """Text table. Sharpe and net P&L are execution diagnostics, not headlines."""
    rows = report.get("rows")
    if not isinstance(rows, list):
        raise ValueError("report is missing rows")
    header = (
        "EXECUTION DIAGNOSTIC ONLY — not a research headline and not a live P&L claim.\n"
        f"research_only={report.get('research_only')} live_pnl_claim={report.get('live_pnl_claim')}\n"
    )
    columns = (
        f"{'sig_lat':>7} {'ex_lat':>7} {'impact_x':>8} {'net_pnl':>14} "
        f"{'sharpe':>10} {'pnl_delta':>14} {'sharpe_delta':>12} {'total_cost':>12}"
    )
    lines = [header.rstrip(), columns]
    for row in rows:
        if not isinstance(row, dict):
            continue
        sharpe = float(row["sharpe_diagnostic"])
        sharpe_delta = float(row["sharpe_delta"])
        lines.append(
            f"{int(row['signal_to_order_bars']):7d} "
            f"{int(row['order_to_exchange_bars']):7d} "
            f"{float(row['impact_multiplier']):8.2f} "
            f"{float(row['net_pnl']):14.4f} "
            f"{sharpe:10.4f} "
            f"{float(row['net_pnl_delta']):14.4f} "
            f"{sharpe_delta:12.4f} "
            f"{float(row['total_cost']):12.4f}"
        )
    return "\n".join(lines) + "\n"
