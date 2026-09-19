"""Event-driven daily backtester. Default fill = next open.

Fills pass through the kill switch and ``check_order`` risk gate before cash
moves. Rejected orders are skipped (not silently unconstrained).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import numpy as np
import polars as pl

from quant_fund.config.models import AppConfig, FillConvention
from quant_fund.execution.costs import total_cost
from quant_fund.metrics.analytics import (
    ANALYTICS_SCHEMA_KEYS,
    analytics_export_digest,
    book_diagnostics,
    export_analytics_dict,
)
from quant_fund.metrics.returns import max_drawdown, sharpe_ratio
from quant_fund.monitoring.kill_switch import KillSwitch
from quant_fund.pipeline.forecast import (
    MARKET_RISK_OVERLAY_GARCH,
    MARKET_RISK_OVERLAY_REALIZED_GARCH,
    market_risk_overlay_asof,
)
from quant_fund.portfolio.risk_gate import check_order
from quant_fund.risk.overlay import BookRiskOverlay
from quant_fund.schemas.errors import KillSwitchActive, RiskGateRejected
from quant_fund.schemas.orders import Order, OrderSide, OrderStatus


class StaleValuationError(RuntimeError):
    """Backtest cannot value a held position within the configured freshness window."""


@dataclass
class BacktestResult:
    equity: pl.DataFrame
    fills: pl.DataFrame
    metrics: dict[str, float | str | int | bool | dict]
    frictionless: bool
    source_note: str


@dataclass
class Book:
    cash: float
    shares: dict[str, float] = field(default_factory=dict)

    def nav(self, prices: dict[str, float]) -> float:
        pos = sum(self.shares.get(s, 0.0) * prices.get(s, 0.0) for s in self.shares)
        return self.cash + pos


def _valid_price(value: object) -> float | None:
    if value is None:
        return None
    try:
        price = float(str(value))
    except (TypeError, ValueError):
        return None
    return price if np.isfinite(price) and price > 0 else None


def _target_weight_map(rows: pl.DataFrame) -> dict[str, float]:
    """Return validated target weights for one decision timestamp.

    Duplicate ``(event_time, security_id)`` rows are rejected rather than
    silently depending on input row order. Security identifiers are normalized
    to strings at this boundary so downstream execution keys are stable.
    """
    required = {"event_time", "security_id", "target_weight"}
    missing = required.difference(rows.columns)
    if missing:
        raise ValueError(f"target weights missing required columns: {sorted(missing)}")
    duplicates = (
        rows.group_by(["event_time", "security_id"])
        .agg(pl.len().alias("_n"))
        .filter(pl.col("_n") > 1)
    )
    if duplicates.height:
        raise ValueError("duplicate target weights for event_time/security_id")
    targets: dict[str, float] = {}
    for row in rows.iter_rows(named=True):
        weight = float(row["target_weight"])
        if not np.isfinite(weight):
            raise ValueError(
                f"target_weight must be finite for {row['security_id']!r} at {row['event_time']!r}"
            )
        targets[str(row["security_id"])] = weight
    return targets


def _validate_target_weight_panel(weights: pl.DataFrame) -> None:
    """Reject duplicate target keys before a backtest starts."""
    _target_weight_map(weights)


def _projected_exposures(
    book: Book,
    prices: dict[str, float],
    sid: str,
    delta: float,
    nav: float,
) -> tuple[float, float, float]:
    """Return (current_weight, gross_after, net_after) after applying ``delta`` shares."""
    nav_safe = max(nav, 1e-12)
    current_shares = book.shares.get(sid, 0.0)
    current_weight = (current_shares * prices.get(sid, 0.0)) / nav_safe
    projected: dict[str, float] = dict(book.shares)
    projected[sid] = current_shares + delta
    all_ids = sorted(set(projected) | set(prices))
    gross = sum(abs(projected.get(s, 0.0) * prices.get(s, 0.0)) for s in all_ids)
    net = sum(projected.get(s, 0.0) * prices.get(s, 0.0) for s in all_ids)
    return current_weight, gross / nav_safe, net / nav_safe


def _make_order(
    *,
    sid: str,
    delta: float,
    signal_time: datetime,
    order_time: datetime,
    order_seq: int,
) -> Order:
    side = OrderSide.BUY if delta > 0 else OrderSide.SELL
    return Order(
        order_id=f"bt-{order_seq}",
        security_id=sid,
        symbol=sid,
        side=side,
        quantity=abs(float(delta)),
        signal_time=signal_time,
        decision_time=signal_time,
        order_time=order_time,
        status=OrderStatus.NEW,
    )


def run_backtest(
    bars: pl.DataFrame,
    weights: pl.DataFrame,
    config: AppConfig,
    *,
    initial_nav: float = 1_000_000.0,
    risk_overlay: BookRiskOverlay | None = None,
) -> BacktestResult:
    """`weights` columns: event_time, security_id, target_weight.

    Target computed from close t is executed at next open (unless close auction enabled).
    Weight rows are a rebalance grid: the last target is held until the next
    row. The optional ``risk_overlay`` may scale or flatten those carried
    weights using prior-close NAV only. Names that do not mark today are
    targeted to 0 so the book can exit while a last print still exists.
    """
    _validate_target_weight_panel(weights)
    px = bars.select(
        "security_id",
        "event_time",
        "open",
        "close",
        "close_total_return",
        "volume",
        pl.col("adv")
        if "adv" in bars.columns
        else (pl.col("close") * pl.col("volume")).alias("adv"),
        pl.col("vol_20") if "vol_20" in bars.columns else pl.lit(0.02).alias("vol_20"),
        "source",
    )
    dates = sorted(px["event_time"].unique().to_list())
    book = Book(cash=initial_nav)
    navs: list[dict] = []
    fill_rows: list[dict] = []
    cost_sum = {"commission": 0.0, "spread": 0.0, "impact": 0.0}
    last_marks: dict[str, float] = {}
    mark_ages: dict[str, int] = {}
    synthetic = (
        "synthetic" in set(px["source"].drop_nulls().to_list()) if "source" in px.columns else False
    )
    kill = KillSwitch(config.kill_switch)
    reject_count = 0
    cash_reject_count = 0
    halt_count = 0
    order_seq = 0
    garch_overlay_dates = 0
    realized_garch_overlay_dates = 0
    last_target_w: dict[str, float] = {}

    use_next_open = (
        config.execution.fill is FillConvention.NEXT_OPEN
        and not config.execution.allow_close_auction
    )

    for i, dt in enumerate(dates[:-1] if use_next_open else dates):
        exec_dt = dates[i + 1] if use_next_open else dt
        day_px = px.filter(pl.col("event_time") == exec_dt)
        day_rows = day_px.iter_rows(named=True)
        exec_mark: dict[str, float] = {}
        close_mark = dict(last_marks)
        next_mark_ages = dict(mark_ages)
        marked_today: set[str] = set()
        advs: dict[str, float] = {}
        vols: dict[str, float] = {}
        for row in day_rows:
            sid = str(row["security_id"])
            raw_exec = _valid_price(row["open"] if use_next_open else row["close"])
            if raw_exec is not None:
                exec_mark[sid] = raw_exec
            total_return_mark = _valid_price(row["close_total_return"])
            fallback_mark = _valid_price(row["close"])
            if total_return_mark is not None:
                close_mark[sid] = total_return_mark
                next_mark_ages[sid] = 0
                marked_today.add(sid)
            elif fallback_mark is not None:
                close_mark[sid] = fallback_mark
                next_mark_ages[sid] = 0
                marked_today.add(sid)
            advs[sid] = _valid_price(row["adv"]) or 1.0
            vols[sid] = _valid_price(row["vol_20"]) or 0.02
        for sid in close_mark:
            if sid not in marked_today:
                next_mark_ages[sid] = next_mark_ages.get(sid, 0) + 1
        last_marks = dict(close_mark)
        mark_ages = next_mark_ages
        stale_held = {
            sid: mark_ages.get(sid)
            for sid, shares in book.shares.items()
            if abs(shares) > 1e-12 and (sid not in close_mark or sid not in mark_ages)
        }
        stale_held.update(
            {
                sid: mark_ages[sid]
                for sid, shares in book.shares.items()
                if abs(shares) > 1e-12
                and sid in mark_ages
                and mark_ages[sid] > config.risk_gate.stale_price_bars
            }
        )
        if stale_held:
            details = ", ".join(f"{sid}={age if age is not None else 'unknown'}" for sid, age in stale_held.items())
            raise StaleValuationError(
                "held position valuation is stale beyond the configured limit: " + details
            )
        tgt_rows = weights.filter(pl.col("event_time") == dt)
        # Value held names without an execution bar at the last close rather
        # than at 0.0: a missing open must not understate NAV / exposures and
        # silently let the risk gate admit orders.
        nav_prices = {**last_marks, **exec_mark}
        nav = book.nav(nav_prices)
        if nav <= 0:
            break
        # Sparse rebalance panels must hold until the next decision date.
        # Missing rows are not a flatten-to-cash instruction.
        if tgt_rows.height > 0:
            last_target_w = _target_weight_map(tgt_rows)
        target_w = dict(last_target_w)
        if risk_overlay is not None:
            overlay_scale = float(risk_overlay.preview_scale())
            if overlay_scale != 1.0:
                target_w = {key: float(value) * overlay_scale for key, value in target_w.items()}
        # A missing mark is an exit, not a ghost hold. Flatten those names
        # while a last execution print may still exist.
        for sid in list(target_w):
            if sid not in marked_today:
                target_w[sid] = 0.0
        for sid, shares in book.shares.items():
            if abs(shares) > 1e-12 and sid not in marked_today:
                target_w[sid] = 0.0
        ids = set(exec_mark) | set(book.shares) | set(target_w)
        traded_turn = 0.0
        market_vol, overlay_source = market_risk_overlay_asof(config, bars, dt)
        if overlay_source == MARKET_RISK_OVERLAY_REALIZED_GARCH:
            realized_garch_overlay_dates += 1
        elif overlay_source == MARKET_RISK_OVERLAY_GARCH:
            garch_overlay_dates += 1
        for sid in sorted(ids):
            price = exec_mark.get(sid)
            if price is None:
                # A missing execution bar is not an executable zero price.
                continue
            tw = target_w.get(sid, 0.0)
            desired = tw * nav / price
            current = book.shares.get(sid, 0.0)
            delta = desired - current
            if abs(delta) * price < 1.0:
                continue
            costs = total_cost(delta, price, advs.get(sid, 1.0), vols.get(sid, 0.02), config.costs)
            # participation cap
            max_qty = config.costs.participation_limit * (advs.get(sid, 1.0) / price)
            if abs(delta) > max_qty > 0:
                delta = np.sign(delta) * max_qty
                costs = total_cost(
                    delta, price, advs.get(sid, 1.0), vols.get(sid, 0.02), config.costs
                )
            try:
                kill.assert_new_orders_allowed()
            except KillSwitchActive:
                # Every attempted order blocked by the halt is counted, mirroring
                # SimulatedBroker's per-order reject accounting. No halt flag:
                # later turns' orders must also reach the gate and be counted,
                # not silently vanish behind an early break.
                halt_count += 1
                continue
            current_w, gross_after, net_after = _projected_exposures(
                book, nav_prices, sid, delta, nav
            )
            participation = abs(delta) * price / max(advs.get(sid, 1.0), 1e-12)
            order_seq += 1
            order = _make_order(
                sid=sid,
                delta=delta,
                signal_time=dt,
                order_time=exec_dt,
                order_seq=order_seq,
            )
            try:
                check_order(
                    order,
                    nav=nav,
                    price=price,
                    current_weight=current_w,
                    gross_after=gross_after,
                    net_after=net_after,
                    participation=participation,
                    predicted_vol=vols.get(sid, 0.02),
                    config=config,
                    market_predicted_vol=market_vol,
                )
            except RiskGateRejected:
                reject_count += 1
                continue
            notional = delta * price
            total_trade_cost = float(costs["total"])
            # Match live/paper execution semantics: a buy is rejected rather
            # than allowing the research book to enter an impossible overdraft.
            if delta > 0 and book.cash < notional + total_trade_cost:
                cash_reject_count += 1
                continue
            book.cash -= notional + total_trade_cost
            book.shares[sid] = current + delta
            # Turnover is based on executed notional, not the requested target
            # change; participation caps can make those materially different.
            traded_turn += abs(notional) / max(nav, 1e-12)
            for k in ("commission", "spread", "impact"):
                cost_sum[k] += float(costs[k])
            fill_rows.append(
                {
                    "fill_time": exec_dt,
                    "signal_time": dt,
                    "security_id": sid,
                    "quantity": delta,
                    "price": price,
                    "fee": costs["commission"],
                    "spread_cost": costs["spread"],
                    "impact_cost": costs["impact"],
                }
            )
        # mark to close
        nav_close = book.nav(close_mark)
        # borrow on shorts
        short_notional = sum(
            abs(min(book.shares.get(s, 0.0), 0.0)) * close_mark.get(s, 0.0) for s in book.shares
        )
        borrow = short_notional * (config.costs.borrow_bps_per_year / 1e4) / 252.0
        if not config.costs.frictionless:
            book.cash -= borrow
            nav_close -= borrow
        navs.append(
            {
                "event_time": exec_dt,
                "nav": nav_close,
                "gross": sum(
                    abs(book.shares.get(s, 0.0) * close_mark.get(s, 0.0)) for s in book.shares
                )
                / max(nav_close, 1e-12),
                "net": sum(book.shares.get(s, 0.0) * close_mark.get(s, 0.0) for s in book.shares)
                / max(nav_close, 1e-12),
                "turnover": traded_turn,
            }
        )
        if risk_overlay is not None:
            risk_overlay.observe(float(nav_close))
    eq = pl.DataFrame(navs) if navs else pl.DataFrame({"event_time": [], "nav": []})
    if eq.height >= 2:
        rets = eq["nav"].pct_change().drop_nulls().to_numpy()
        sr = sharpe_ratio(rets)
        navs_list = [float(v) for v in eq["nav"].to_list()]
        turns = [float(v) for v in eq["turnover"].to_list()] if "turnover" in eq.columns else [0.0]
        gross_s = eq["gross"].to_numpy() if "gross" in eq.columns else None
        net_s = eq["net"].to_numpy() if "net" in eq.columns else None
        diag = book_diagnostics(
            rets,
            gross_exposure=gross_s,
            net_exposure=net_s,
            turnover=np.asarray(turns, dtype=float),
            label="BACKTEST_SIM",
            data_source="SYNTHETIC" if synthetic else "file",
        )
        analytics_export = export_analytics_dict(
            diag,
            extra={
                "total_return": navs_list[-1] / float(initial_nav) - 1.0,
                "sharpe": sr["sharpe"],
                "n": sr["n"],
                "max_drawdown": max_drawdown(rets),
                "commission": cost_sum["commission"],
                "spread": cost_sum["spread"],
                "impact": cost_sum["impact"],
                "flag_high_sharpe": sr["flag_high_sharpe"],
                "risk_gate_rejects": reject_count,
                "cash_rejects": cash_reject_count,
                "kill_switch_halts": halt_count,
            },
        )
        metrics: dict[str, float | str | int | bool | dict] = {
            "total_return": navs_list[-1] / float(initial_nav) - 1.0,
            "sharpe": sr["sharpe"],
            "n": sr["n"],
            "max_drawdown": max_drawdown(rets),
            "mean_turnover": float(np.mean(turns)),
            "commission": cost_sum["commission"],
            "spread": cost_sum["spread"],
            "impact": cost_sum["impact"],
            "flag_high_sharpe": sr["flag_high_sharpe"],
            "risk_gate_rejects": reject_count,
            "cash_rejects": cash_reject_count,
            "kill_switch_halts": halt_count,
            "garch_risk_overlay_dates": garch_overlay_dates,
            "realized_garch_risk_overlay_dates": realized_garch_overlay_dates,
            "analytics": diag,
            "analytics_export": analytics_export,
            "research_only": True,
            "live_pnl_claim": False,
        }
    else:
        # Empty / short panel: no equity path long enough for returns.
        # Still force research-only labeling (never a live P&L claim).
        metrics = {
            "total_return": 0.0,
            "sharpe": float("nan"),
            "n": 0,
            "risk_gate_rejects": reject_count,
            "cash_rejects": cash_reject_count,
            "kill_switch_halts": halt_count,
            "garch_risk_overlay_dates": garch_overlay_dates,
            "realized_garch_risk_overlay_dates": realized_garch_overlay_dates,
            "research_only": True,
            "live_pnl_claim": False,
        }
    if risk_overlay is not None:
        metrics["book_risk_overlay"] = risk_overlay.snapshot()
    if config.costs.frictionless:
        metrics["label"] = "FRICTIONLESS RESEARCH ONLY"
    note = "SYNTHETIC" if synthetic else "file"
    metrics["data_source"] = note
    return BacktestResult(
        equity=eq,
        fills=pl.DataFrame(fill_rows) if fill_rows else pl.DataFrame(),
        metrics=metrics,
        frictionless=config.costs.frictionless,
        source_note=note,
    )


def cost_sensitivity(
    bars: pl.DataFrame,
    weights: pl.DataFrame,
    config: AppConfig,
    *,
    impact_multipliers: tuple[float, ...] = (1.0, 2.0),
    initial_nav: float = 1_000_000.0,
) -> dict[str, object]:
    """Run matched cost scenarios for execution robustness.

    This is deliberately separate from scientific research families. It
    answers whether the simulated economic diagnostic survives configured and
    doubled square-root impact; it does not establish live profitability.

    Output never includes Sharpe / flag_high_sharpe keys (execution diagnostic
    only; ``live_pnl_claim=False``).
    """
    if not impact_multipliers or any(float(m) <= 0.0 for m in impact_multipliers):
        raise ValueError("impact_multipliers must contain positive values")
    required_w = {"event_time", "security_id", "target_weight"}
    missing_w = required_w - set(weights.columns)
    if missing_w:
        raise ValueError(f"weights missing columns: {sorted(missing_w)}")
    required_b = {
        "event_time",
        "security_id",
        "open",
        "close",
        "close_total_return",
        "volume",
        "source",
    }
    missing_b = required_b - set(bars.columns)
    if missing_b:
        raise ValueError(f"bars missing columns: {sorted(missing_b)}")
    baseline = float(config.costs.impact_y)
    if bars.height == 0:
        return {
            "scenarios": {},
            "baseline_impact_y": baseline,
            "claim": "execution_diagnostic_only",
            "empty": True,
            "research_only": True,
            "live_pnl_claim": False,
        }
    scenarios: dict[str, dict[str, object]] = {}
    for multiplier in impact_multipliers:
        scenario = config.model_copy(deep=True)
        scenario.costs.impact_y = baseline * float(multiplier)
        result = run_backtest(bars, weights, scenario, initial_nav=initial_nav)
        costs = [
            float(value)
            for name in ("commission", "spread", "impact")
            if isinstance((value := result.metrics.get(name, 0.0)), (int, float))
        ]
        # Deliberately omit sharpe / flag_high_sharpe — not a research headline.
        scenarios[f"impact_{float(multiplier):g}x"] = {
            "impact_multiplier": float(multiplier),
            "total_return": result.metrics.get("total_return"),
            "total_cost": sum(costs),
            "impact_cost": result.metrics.get("impact", 0.0),
            "risk_gate_rejects": result.metrics.get("risk_gate_rejects", 0),
            "source": result.source_note,
            "claim": "execution_diagnostic_only",
        }
    return {
        "scenarios": scenarios,
        "baseline_impact_y": baseline,
        "claim": "execution_diagnostic_only",
        "research_only": True,
        "live_pnl_claim": False,
    }


def _atomic_write_text(path: Path, content: str) -> None:
    """Publish a text artifact atomically so readers never see partial JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def export_backtest_metrics_json(
    result: BacktestResult,
    path: Path | str,
    *,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Write backtest metrics JSON aligned with paper ledger analytics schema.

    Prefer ``result.metrics["analytics_export"]`` when present; otherwise project
    ``analytics`` / top-level metrics through ``export_analytics_dict``. Always
    forces research-only labeling (never a live P&L claim).
    """
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    metrics = dict(result.metrics)
    blob = metrics.get("analytics_export")
    if not isinstance(blob, dict):
        diag = metrics.get("analytics")
        if not isinstance(diag, dict):
            diag = {}
        top_extra: dict[str, Any] = {
            str(k): metrics[k]
            for k in (
                "total_return",
                "sharpe",
                "n",
                "max_drawdown",
                "commission",
                "spread",
                "impact",
                "flag_high_sharpe",
                "risk_gate_rejects",
                "kill_switch_halts",
                "mean_turnover",
            )
            if k in metrics
        }
        if extra:
            top_extra.update(extra)
        blob = export_analytics_dict(diag, extra=top_extra)
    elif extra:
        blob = dict(blob)
        blob.update(extra)
    else:
        blob = dict(blob)
    # Always force research-only labeling after any merge (never a live P&L claim).
    blob["research_only"] = True
    blob["live_pnl_claim"] = False
    blob["frictionless"] = bool(result.frictionless)
    blob["source_note"] = str(result.source_note)
    blob["aligned_schema_keys"] = list(ANALYTICS_SCHEMA_KEYS)
    blob["analytics_export_sha256"] = analytics_export_digest(blob)
    _atomic_write_text(dest, json.dumps(blob, indent=2, default=str))
    return dest
