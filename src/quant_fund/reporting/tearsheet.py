"""Performance tearsheet: assembles the repo's metric primitives into a
single institutional-style report (dict + markdown).

Consumes an equity frame (``event_time, nav`` — the backtester's output,
optionally with ``gross``/``net``/``turnover`` columns), optional fills and
target weights, and renders:

- summary statistics (CAGR, vol, Sharpe/Sortino/Calmar, hit rate, moments)
- drawdown analysis (episodes, depth, duration, time-underwater share)
- risk block (historical VaR/ES at 95/99)
- execution block (turnover, cost totals, fill count)
- exposure block (mean/max gross and net, when the equity frame carries them)
- period returns table (year-month buckets for datetime stamps)
- optional per-name / per-sleeve P&L attribution (weights + bars required)

All outputs are research diagnostics: ``live_pnl_claim`` is always false,
synthetic inputs are labeled, and empty inputs fail closed to honest NaN.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from quant_fund.metrics.analytics import drawdown_duration_stats, equity_curve_analytics
from quant_fund.metrics.returns import (
    annualized_vol,
    cagr,
    calmar_ratio,
    max_drawdown,
    sharpe_ratio,
    sortino_ratio,
)
from quant_fund.metrics.risk import historical_es, historical_var


def _nav_and_returns(equity: pl.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    if "nav" not in equity.columns:
        raise ValueError("equity frame requires a 'nav' column")
    nav = equity["nav"].to_numpy().astype(float)
    nav = nav[np.isfinite(nav)]
    if nav.size < 2:
        return nav, np.array([], dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        rets = nav[1:] / nav[:-1] - 1.0
    rets = rets[np.isfinite(rets)]
    return nav, rets


def period_returns_table(equity: pl.DataFrame) -> dict[str, float]:
    """Compound returns per calendar year-month (datetime event_time only).

    Irregular/missing months are skipped (the compounding covers whatever
    bars each bucket contains). Returns ``{"YYYY-MM": ret}`` ordered by key.
    """
    if not {"event_time", "nav"} <= set(equity.columns) or equity.height < 2:
        return {}
    if not equity.schema["event_time"].is_temporal():
        return {}
    monthly = (
        equity.sort("event_time")
        .with_columns(
            pl.col("event_time").dt.strftime("%Y-%m").alias("bucket"),
        )
        .group_by("bucket")
        .agg(
            (pl.col("nav").last() / pl.col("nav").first() - 1.0).alias("ret"),
        )
        .sort("bucket")
    )
    return {str(r["bucket"]): float(r["ret"]) for r in monthly.iter_rows(named=True)}


def build_tearsheet(
    equity: pl.DataFrame,
    *,
    fills: pl.DataFrame | None = None,
    weights: pl.DataFrame | None = None,
    bars: pl.DataFrame | None = None,
    sleeve_map: dict[str, str] | pl.DataFrame | None = None,
    periods_per_year: float = 252.0,
    label: str = "BACKTEST_SIM",
    synthetic: bool = False,
    top_n: int = 10,
) -> dict[str, Any]:
    """Assemble the full tearsheet dict. Empty equity → fail-closed stub."""
    nav, rets = _nav_and_returns(equity)
    sheet: dict[str, Any] = {
        "label": label,
        "synthetic": bool(synthetic),
        "periods_per_year": float(periods_per_year),
        "live_pnl_claim": False,
        "research_only": True,
    }
    if nav.size < 2 or rets.size == 0:
        sheet["summary"] = {"n_bars": int(nav.size), "status": "empty_or_short"}
        return sheet

    sr = sharpe_ratio(rets, periods_per_year=periods_per_year)
    sheet["summary"] = {
        "n_bars": int(nav.size),
        "nav_start": float(nav[0]),
        "nav_end": float(nav[-1]),
        "total_return": float(nav[-1] / nav[0] - 1.0),
        "cagr": cagr(rets, periods_per_year),
        "vol_ann": annualized_vol(rets, periods_per_year),
        "sharpe": sr["sharpe"],
        "sortino": sortino_ratio(rets, periods_per_year=periods_per_year),
        "calmar": calmar_ratio(rets, periods_per_year),
        "max_drawdown": max_drawdown(rets),
        "hit_rate": float(np.mean(rets > 0.0)),
        "mean_ret": float(np.mean(rets)),
        "skew": float(_safe_moment(rets, 3)),
        "excess_kurt": float(_safe_moment(rets, 4) - 3.0),
        "flag_high_sharpe": bool(sr["flag_high_sharpe"]),
    }

    # Drawdown episodes, with event_time resolved where available.
    dd = drawdown_duration_stats(nav=nav)
    times = (
        equity["event_time"].to_list()
        if "event_time" in equity.columns and equity.height == nav.size
        else None
    )
    episodes = []
    for ep in dd["episodes"]:
        row = dict(ep)
        if times is not None:
            row["start"] = str(times[int(ep["start_idx"])])
            row["trough_time"] = str(times[int(ep["trough_idx"])])
            row["end"] = str(times[int(ep["end_idx"])])
        episodes.append(row)
    episodes.sort(key=lambda e: float(e["trough"]))
    sheet["drawdown"] = {
        "n_episodes": dd["n_episodes"],
        "max_underwater_duration": dd["max_underwater_duration"],
        "time_underwater_frac": dd["time_underwater_frac"],
        "current_underwater_duration": dd["current_underwater_duration"],
        "deepest_trough": dd["deepest_trough"],
        "worst_episodes": episodes[:5],
    }

    losses = -rets
    sheet["risk"] = {
        "var_95": historical_var(losses, 0.95),
        "es_95": historical_es(losses, 0.95),
        "var_99": historical_var(losses, 0.99),
        "es_99": historical_es(losses, 0.99),
        "worst_bar": float(np.min(rets)),
        "best_bar": float(np.max(rets)),
    }

    if fills is not None and fills.height > 0:
        cost_cols = [
            c for c in ("fee", "spread_cost", "impact_cost", "slippage") if c in fills.columns
        ]
        sheet["execution"] = {
            "n_fills": int(fills.height),
            "cost_totals": {
                c: float(fills[c].sum())
                for c in cost_cols  # type: ignore[index]
            },
            "total_cost": float(
                sum(fills[c].sum() for c in cost_cols)  # type: ignore[arg-type]
            ),
        }
    if "turnover" in equity.columns:
        sheet.setdefault("execution", {})["mean_turnover"] = float(
            np.nanmean(equity["turnover"].to_numpy().astype(float))
        )

    if {"gross", "net"} <= set(equity.columns):
        g = equity["gross"].to_numpy().astype(float)
        n = equity["net"].to_numpy().astype(float)
        sheet["exposure"] = {
            "mean_gross": float(np.nanmean(g)),
            "max_gross": float(np.nanmax(g)),
            "mean_net": float(np.nanmean(n)),
            "mean_abs_net": float(np.nanmean(np.abs(n))),
        }

    sheet["period_returns"] = period_returns_table(equity)
    sheet["equity_curve"] = equity_curve_analytics(nav, periods_per_year=periods_per_year)

    if weights is not None and bars is not None and weights.height > 0 and bars.height > 0:
        from quant_fund.portfolio.pnl_attribution import (
            attribute_weights_pnl,
            attribution_summary,
        )

        frame = attribute_weights_pnl(weights, bars, fills=fills, nav=equity)
        sheet["attribution"] = attribution_summary(frame, sleeve_map=sleeve_map, top_n=top_n)
    return sheet


def _safe_moment(rets: np.ndarray, k: int) -> float:
    """Central moment normalized by sigma^k (NaN on degenerate input)."""
    if rets.size < 3:
        return float("nan")
    sd = float(np.std(rets, ddof=1))
    if sd <= 0 or not np.isfinite(sd):
        return float("nan")
    return float(np.mean((rets - np.mean(rets)) ** k) / sd**k)


def tearsheet_markdown(sheet: dict[str, Any]) -> str:
    """Render a tearsheet dict as a compact markdown report."""
    lines = [f"# Tearsheet — {sheet.get('label', 'book')}", ""]
    if sheet.get("synthetic"):
        lines += ["> SYNTHETIC DATA. Not evidence of live profitability.", ""]
    lines.append("> Research diagnostic only; not a live P&L claim.")
    lines.append("")

    def _fmt(v: Any) -> str:
        if isinstance(v, float):
            if np.isnan(v):
                return "n/a"
            return f"{v:.6g}"
        return str(v)

    summary = sheet.get("summary", {})
    if summary:
        lines.append("## Summary")
        for k, v in summary.items():
            lines.append(f"- {k}: {_fmt(v)}")
        lines.append("")
    for name in ("drawdown", "risk", "execution", "exposure", "equity_curve"):
        block = sheet.get(name)
        if not block:
            continue
        lines.append(f"## {name.replace('_', ' ').title()}")
        for k, v in block.items():
            if k == "worst_episodes":
                lines.append("- worst_episodes:")
                for ep in v:
                    lines.append(
                        f"  - trough {_fmt(ep['trough'])} over {ep['duration']} bars"
                        f" ({ep.get('start', ep['start_idx'])} → {ep.get('end', ep['end_idx'])})"
                    )
            elif isinstance(v, dict):
                for kk, vv in v.items():
                    lines.append(f"- {k}.{kk}: {_fmt(vv)}")
            else:
                lines.append(f"- {k}: {_fmt(v)}")
        lines.append("")
    pr = sheet.get("period_returns") or {}
    if pr:
        lines.append("## Period Returns (year-month)")
        lines.append("")
        lines.append("| period | return |")
        lines.append("|---|---|")
        for k, v in pr.items():
            lines.append(f"| {k} | {_fmt(v)} |")
        lines.append("")
    attr = sheet.get("attribution")
    if attr:
        lines.append("## Attribution (weight × return approximation)")
        lines.append(f"- total_gross_pnl: {_fmt(attr.get('total_gross_pnl'))}")
        lines.append(f"- total_cost: {_fmt(attr.get('total_cost'))}")
        lines.append(f"- total_net_pnl: {_fmt(attr.get('total_net_pnl'))}")
        lines.append("- top contributors:")
        for row in attr.get("top_contributors", [])[:5]:
            lines.append(
                f"  - {row['security_id']}: net {_fmt(row['net_pnl'])}"
                f" (gross {_fmt(row['gross_pnl'])})"
            )
        lines.append("- bottom contributors:")
        for row in attr.get("bottom_contributors", [])[:5]:
            lines.append(
                f"  - {row['security_id']}: net {_fmt(row['net_pnl'])}"
                f" (gross {_fmt(row['gross_pnl'])})"
            )
        for row in attr.get("by_sleeve", []):
            lines.append(f"- sleeve {row['sleeve']}: net {_fmt(row['net_pnl'])}")
        lines.append("")
    return "\n".join(lines)


def write_tearsheet_md(path: Path, sheet: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(tearsheet_markdown(sheet))
    return path
