"""Event-driven USDT-M perpetual backtester. Default fill = next bar open.

Differs from ``run_backtest`` (the spot book) in the economics it simulates:

- Positions are netted signed quantities; notional is *not* paid from cash.
  ``equity = cash + Σ qty·(mark − vwap_entry)`` — wallet balance plus
  unrealized PnL, where ``cash`` starts at ``initial_nav``.
- Funding is a cashflow, not a transaction cost: at each funding timestamp the
  wallet moves ``−qty·mark·rate`` (positive rate → longs pay shorts).
- Leverage is capped: gross notional ≤ ``max_leverage · equity``.
- Maintenance margin: when ``equity < maint_margin_ratio · gross_notional``
  positions are force-closed largest-first at the adverse bar excursion with a
  liquidation fee. ``liquidation_on_wick=True`` (default) tests the bar's
  adverse extreme — the paranoid convention, since intra-bar ordering of the
  funding debit vs the excursion is unknowable from OHLCV.
- Annualization is inferred from median bar spacing on a 365.25-day crypto
  calendar — ``sharpe_ratio``'s 252-periods default is wrong for 1h/4h bars.

Every result carries ``research_only=True`` and ``live_pnl_claim=False``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Protocol

import numpy as np
import polars as pl

from quant_fund.backtest.engine import (
    BacktestResult,
    StaleValuationError,
    _make_order,
    _target_weight_map,
    _valid_price,
)
from quant_fund.config.models import AppConfig
from quant_fund.execution.costs import total_cost
from quant_fund.metrics.analytics import book_diagnostics
from quant_fund.metrics.returns import cagr, max_drawdown, sharpe_ratio
from quant_fund.monitoring.kill_switch import KillSwitch
from quant_fund.portfolio.risk_gate import check_order
from quant_fund.schemas.errors import KillSwitchActive, RiskGateRejected

SECONDS_PER_YEAR_CRYPTO = 365.25 * 86400.0


class OverlayScaler(Protocol):
    """Causal equity-path-aware target scaler (vol targeting, DD governor).

    The engine calls ``observe(dt, nav)`` after each close marking and
    ``scale(dt, targets)`` before queueing orders — the scaler therefore only
    ever sees the realized past, never the future.
    """

    def observe(self, dt: datetime, nav: float) -> None: ...
    def scale(self, dt: datetime, targets: dict[str, float]) -> dict[str, float]: ...


@dataclass
class PerpBook:
    """Margin book: cash wallet + signed netted positions with vwap entries."""

    cash: float
    qty: dict[str, float] = field(default_factory=dict)
    entry: dict[str, float] = field(default_factory=dict)

    def upnl(self, marks: dict[str, float]) -> float:
        return sum(
            self.qty[s] * (marks[s] - self.entry.get(s, 0.0))
            for s in self.qty
            if abs(self.qty[s]) > 1e-12 and s in marks
        )

    def equity(self, marks: dict[str, float]) -> float:
        return self.cash + self.upnl(marks)

    def gross_notional(self, marks: dict[str, float]) -> float:
        return sum(abs(self.qty[s]) * marks[s] for s in self.qty if s in marks)

    def net_notional(self, marks: dict[str, float]) -> float:
        return sum(self.qty[s] * marks[s] for s in self.qty if s in marks)


def infer_periods_per_year(times: list[datetime], hint_seconds: float | None = None) -> float:
    """Bar-spacing annualization on a 365.25-day calendar.

    Median spacing is robust to isolated missing bars; a hint wins only when
    the panel is too short to infer from (<3 timestamps).
    """
    if len(times) >= 3:
        diffs = np.diff(np.asarray([t.timestamp() for t in sorted(times)], dtype=float))
        diffs = diffs[diffs > 0]
        if diffs.size:
            return float(SECONDS_PER_YEAR_CRYPTO / np.median(diffs))
    if hint_seconds is not None and hint_seconds > 0:
        return float(SECONDS_PER_YEAR_CRYPTO / hint_seconds)
    raise ValueError("cannot infer bar spacing: need >=3 distinct timestamps or bar_seconds_hint")


def _bars_by_time(bars: pl.DataFrame) -> dict[datetime, dict[str, dict[str, float]]]:
    """Partition a bar panel into {event_time: {sid: row}} for the event loop."""
    required = {"event_time", "security_id", "open", "high", "low", "close", "volume"}
    missing = required - set(bars.columns)
    if missing:
        raise ValueError(f"perp bars missing columns: {sorted(missing)}")
    out: dict[datetime, dict[str, dict[str, float]]] = {}
    for row in bars.iter_rows(named=True):
        sid = str(row["security_id"])
        slot = out.setdefault(row["event_time"], {})
        if sid in slot:
            raise ValueError(f"duplicate bar for {sid} at {row['event_time']}")
        slot[sid] = row
    return out


def _funding_by_time(
    funding: pl.DataFrame | None,
    *,
    multiplier: float = 1.0,
) -> dict[datetime, list[tuple[str, float]]]:
    """Group funding events by timestamp; ``multiplier`` stress-scales rates."""
    out: dict[datetime, list[tuple[str, float]]] = {}
    if funding is None or funding.height == 0:
        return out
    required = {"event_time", "security_id", "value"}
    missing = required - set(funding.columns)
    if missing:
        raise ValueError(f"funding frame missing columns: {sorted(missing)}")
    for row in funding.iter_rows(named=True):
        rate = float(row["value"]) * multiplier
        if not np.isfinite(rate):
            raise ValueError("funding rate must be finite")
        out.setdefault(row["event_time"], []).append((str(row["security_id"]), rate))
    return out


def _bar_enrichment(bars: pl.DataFrame) -> pl.DataFrame:
    """Causal ADV/σ columns per symbol (shift-1 rolling stats) for cost sizing."""
    lagged = [
        (pl.col("close") * pl.col("volume"))
        .rolling_mean(20, min_samples=1)
        .shift(1)
        .over("security_id")
        .alias("adv"),
        pl.col("close")
        .pct_change()
        .rolling_std(20, min_samples=2)
        .shift(1)
        .over("security_id")
        .alias("vol_20"),
    ]
    return bars.with_columns(lagged)


def run_perp_backtest(
    bars: pl.DataFrame,
    funding: pl.DataFrame | None,
    weights: pl.DataFrame,
    config: AppConfig,
    *,
    initial_nav: float = 1_000_000.0,
    scaler: OverlayScaler | None = None,
) -> BacktestResult:
    """`weights` columns: event_time, security_id, target_weight (of equity).

    Targets decided on bar t's close execute at bar t+1+fill_delay_bars open.
    A funding event at timestamp f is applied to the bar whose window
    ``(open, close]`` contains f (funding at 08:00 lands on the 07:00 bar for
    1h data), marked at that bar's close.
    """
    if weights.height:
        _ = _target_weight_map(weights)
    # Pre-group target rows by event_time once — a per-bar frame filter is
    # O(N) per timestamp and dominates runtime on hourly panels.
    wmap_by_time: dict[datetime, dict[str, float]] = {}
    for wrow in weights.select("event_time", "security_id", "target_weight").iter_rows(named=True):
        wmap_by_time.setdefault(wrow["event_time"], {})[str(wrow["security_id"])] = float(
            wrow["target_weight"]
        )
    enriched = _bar_enrichment(bars)
    by_time = _bars_by_time(enriched)
    times = sorted(by_time)
    perp = config.perp
    ppy = (
        float(perp.periods_per_year_override)
        if perp.periods_per_year_override is not None
        else infer_periods_per_year(times, perp.bar_seconds_hint)
    )
    fund_map = (
        _funding_by_time(funding, multiplier=perp.funding_spike_multiplier)
        if perp.funding_enabled
        else {}
    )
    synthetic = (
        "synthetic" in set(bars["source"].drop_nulls().to_list())
        if "source" in bars.columns
        else False
    )
    book = PerpBook(cash=float(initial_nav))
    pending: dict[str, tuple[float, datetime, int]] = {}  # sid -> (weight, signal_t, exec_idx)
    pending_exec_at: dict[str, int] = {}
    navs: list[dict] = []
    fill_rows: list[dict] = []
    liq_rows: list[dict] = []
    cost_sum = {"commission": 0.0, "spread": 0.0, "impact": 0.0}
    funding_received = 0.0
    funding_paid = 0.0
    funding_events_applied = 0
    # Funding events whose timestamp never equals a bar time are silently
    # skipped by ``fund_map.get(dt)`` — count them so feed misalignment is
    # visible instead of fabricating zero-funding receipts.
    bar_times = set(times)
    funding_events_dropped = sum(
        len(events) for ts, events in fund_map.items() if ts not in bar_times
    )
    liquidation_count = 0
    liquidation_cost = 0.0
    margin_reject_count = 0
    reject_count = 0
    halt_count = 0
    order_seq = 0
    last_marks: dict[str, float] = {}
    mark_ages: dict[str, int] = {}
    kill = KillSwitch(config.kill_switch)
    ruined = False

    for i, dt in enumerate(times):
        slot = by_time[dt]
        # --- marks for this bar ---
        exec_mark: dict[str, float] = {}
        close_mark = dict(last_marks)
        next_ages = dict(mark_ages)
        marked: set[str] = set()
        advs: dict[str, float] = {}
        vols: dict[str, float] = {}
        for sid, row in slot.items():
            o, c = _valid_price(row["open"]), _valid_price(row["close"])
            if o is not None:
                exec_mark[sid] = o
            if c is not None:
                close_mark[sid] = c
                next_ages[sid] = 0
                marked.add(sid)
            advs[sid] = _valid_price(row.get("adv")) or 1.0
            vols[sid] = _valid_price(row.get("vol_20")) or 0.02
        for sid in close_mark:
            if sid not in marked:
                next_ages[sid] = next_ages.get(sid, 0) + 1
        last_marks = close_mark
        mark_ages = next_ages
        stale_held = {
            sid: mark_ages.get(sid)
            for sid, q in book.qty.items()
            if abs(q) > 1e-12 and mark_ages.get(sid, 10**9) > config.risk_gate.stale_price_bars
        }
        if stale_held:
            details = ", ".join(f"{s}={a}" for s, a in stale_held.items())
            raise StaleValuationError("held perp position valuation stale beyond limit: " + details)

        # --- execute pending orders at this bar's open ---
        nav_prices = {**last_marks, **exec_mark}
        # No early equity<=0 break here: the wick-liquidation pass below must
        # first model the forced unwind; ruin is declared after close marking.
        traded_turn = 0.0
        due = [sid for sid, idx in pending_exec_at.items() if idx <= i]
        for sid in sorted(due):
            target_w, signal_t, _ = pending.pop(sid)
            del pending_exec_at[sid]
            price = exec_mark.get(sid)
            if price is None:
                continue  # no bar at this time → order lapses (missed fill, honest)
            equity = book.equity(nav_prices)
            desired_qty = target_w * equity / price
            current = book.qty.get(sid, 0.0)
            delta = desired_qty - current
            if abs(delta) * price < 1.0:
                continue
            # Hard leverage cap: scale back orders that would breach it.
            gross_now = book.gross_notional(nav_prices)
            cap = perp.max_leverage * equity
            if gross_now + abs(delta) * price > cap and cap > gross_now:
                delta = np.sign(delta) * max(0.0, (cap - gross_now)) / price
            elif gross_now + abs(delta) * price > cap:
                margin_reject_count += 1
                continue
            if abs(delta) * price < 1.0:
                continue
            costs = total_cost(delta, price, advs.get(sid, 1.0), vols.get(sid, 0.02), config.costs)
            max_qty = config.costs.participation_limit * (advs.get(sid, 1.0) / price)
            if abs(delta) > max_qty > 0:
                delta = np.sign(delta) * max_qty
                costs = total_cost(
                    delta, price, advs.get(sid, 1.0), vols.get(sid, 0.02), config.costs
                )
            try:
                kill.assert_new_orders_allowed()
            except KillSwitchActive:
                halt_count += 1
                continue
            nav_safe = max(equity, 1e-12)
            current_w = current * price / nav_safe
            projected = dict(book.qty)
            projected[sid] = current + delta
            all_ids = sorted(set(projected) | set(nav_prices))
            gross_after = (
                sum(abs(projected.get(s, 0.0) * nav_prices.get(s, 0.0)) for s in all_ids) / nav_safe
            )
            net_after = (
                sum(projected.get(s, 0.0) * nav_prices.get(s, 0.0) for s in all_ids) / nav_safe
            )
            participation = abs(delta) * price / max(advs.get(sid, 1.0), 1e-12)
            order_seq += 1
            order = _make_order(
                sid=sid, delta=delta, signal_time=signal_t, order_time=dt, order_seq=order_seq
            )
            try:
                check_order(
                    order,
                    nav=equity,
                    price=price,
                    current_weight=current_w,
                    gross_after=gross_after,
                    net_after=net_after,
                    participation=participation,
                    predicted_vol=vols.get(sid, 0.02),
                    config=config,
                )
            except RiskGateRejected:
                reject_count += 1
                continue
            fee = float(costs["total"])
            notional = delta * price
            # Signed-position accounting: closing quantity realizes
            # q_closed·(price − entry) into cash; the remainder keeps its entry
            # (or re-anchors at this fill when adding or flipping).
            new_qty = current + delta
            old_entry = book.entry.get(sid, price)
            if abs(current) < 1e-12 or np.sign(new_qty) != np.sign(current):
                book.cash += current * (price - old_entry)
                book.entry[sid] = price
            elif abs(new_qty) > abs(current):
                book.entry[sid] = (abs(current) * old_entry + abs(delta) * price) / abs(new_qty)
            else:
                book.cash += (current - new_qty) * (price - old_entry)
            if abs(new_qty) < 1e-12:
                book.qty.pop(sid, None)
                book.entry.pop(sid, None)
            else:
                book.qty[sid] = new_qty
            book.cash -= fee
            traded_turn += abs(notional) / nav_safe
            for k in ("commission", "spread", "impact"):
                cost_sum[k] += float(costs[k])
            fill_rows.append(
                {
                    "fill_time": dt,
                    "signal_time": signal_t,
                    "security_id": sid,
                    "quantity": delta,
                    "price": price,
                    "fee": costs["commission"],
                    "spread_cost": costs["spread"],
                    "impact_cost": costs["impact"],
                }
            )

        # --- funding events inside this bar window (prev_close, this_close] ---
        for sid, rate in fund_map.get(dt, []):
            q = book.qty.get(sid, 0.0)
            if abs(q) < 1e-12:
                continue
            mark = last_marks.get(sid)
            if mark is None:
                continue
            flow = -q * mark * rate
            book.cash += flow
            if flow >= 0:
                funding_received += flow
            else:
                funding_paid += -flow
            funding_events_applied += 1

        # --- liquidation check (wick-paranoid by default) ---
        def _mmr_deficit(marks: dict[str, float]) -> float:
            return perp.maint_margin_ratio * book.gross_notional(marks) - book.equity(marks)

        adverse: dict[str, float] = {}
        if perp.liquidation_on_wick:
            for sid, row in slot.items():
                q = book.qty.get(sid, 0.0)
                if abs(q) < 1e-12:
                    continue
                if q > 0:
                    adverse[sid] = _valid_price(row["low"]) or last_marks.get(sid, 0.0)
                else:
                    adverse[sid] = _valid_price(row["high"]) or last_marks.get(sid, 0.0)
            liq_marks = {**last_marks, **adverse}
        else:
            liq_marks = dict(last_marks)
        while _mmr_deficit(liq_marks) > 0 and book.gross_notional(liq_marks) > 0:
            victim = max(
                (s for s in book.qty if abs(book.qty[s]) > 1e-12),
                key=lambda s: abs(book.qty[s]) * liq_marks.get(s, 0.0),
                default=None,
            )
            if victim is None:
                break
            q = book.qty.pop(victim)
            liq_price = liq_marks.get(victim, book.entry.get(victim, 0.0))
            # realized pnl + liquidation fee on notional
            book.cash += q * (liq_price - book.entry.get(victim, liq_price))
            liq_fee = abs(q) * liq_price * perp.liquidation_fee_bps / 1e4
            book.cash -= liq_fee
            liquidation_cost += liq_fee
            liquidation_count += 1
            liq_rows.append(
                {
                    "event_time": dt,
                    "security_id": victim,
                    "quantity": q,
                    "liq_price": liq_price,
                    "fee": liq_fee,
                }
            )
            book.entry.pop(victim, None)

        # --- mark to close, queue next targets ---
        nav_close = book.equity(last_marks)
        navs.append(
            {
                "event_time": dt,
                "nav": nav_close,
                "gross": book.gross_notional(last_marks) / max(abs(nav_close), 1e-12),
                "net": book.net_notional(last_marks) / max(abs(nav_close), 1e-12),
                "turnover": traded_turn,
            }
        )
        if scaler is not None:
            scaler.observe(dt, nav_close)
        if nav_close <= 0:
            ruined = True
            break
        if i + 1 + perp.fill_delay_bars < len(times):
            targets = wmap_by_time.get(dt)
            if targets:
                if scaler is not None:
                    targets = scaler.scale(dt, dict(targets))
                exec_idx = i + 1 + perp.fill_delay_bars
                for sid, w in targets.items():
                    pending[sid] = (w, dt, exec_idx)
                    pending_exec_at[sid] = exec_idx

    eq = pl.DataFrame(navs) if navs else pl.DataFrame({"event_time": [], "nav": []})
    if eq.height >= 2:
        rets = eq["nav"].pct_change().drop_nulls().to_numpy()
        sr = sharpe_ratio(rets, periods_per_year=ppy)
        navs_list = [float(v) for v in eq["nav"].to_list()]
        turns = [float(v) for v in eq["turnover"].to_list()]
        diag = book_diagnostics(
            rets,
            gross_exposure=eq["gross"].to_numpy(),
            net_exposure=eq["net"].to_numpy(),
            turnover=np.asarray(turns, dtype=float),
            label="PERP_BACKTEST_SIM",
            data_source="SYNTHETIC" if synthetic else "file",
        )
        metrics: dict[str, float | str | int | bool | dict] = {
            "total_return": navs_list[-1] / float(initial_nav) - 1.0,
            "cagr": cagr(rets, periods_per_year=ppy),
            "sharpe": sr["sharpe"],
            "n": sr["n"],
            "periods_per_year": ppy,
            "max_drawdown": max_drawdown(rets),
            "mean_turnover": float(np.mean(turns)),
            "commission": cost_sum["commission"],
            "spread": cost_sum["spread"],
            "impact": cost_sum["impact"],
            "funding_paid_total": funding_paid,
            "funding_received_total": funding_received,
            "funding_net": funding_received - funding_paid,
            "funding_events_applied": funding_events_applied,
            "funding_events_dropped": funding_events_dropped,
            "liquidation_count": liquidation_count,
            "liquidation_cost": liquidation_cost,
            "margin_rejects": margin_reject_count,
            "ruined": ruined,
            "flag_high_sharpe": sr["flag_high_sharpe"],
            "risk_gate_rejects": reject_count,
            "kill_switch_halts": halt_count,
            "analytics": diag,
            "book_type": "usdtm_perp",
            "research_only": True,
            "live_pnl_claim": False,
        }
    else:
        metrics = {
            "total_return": 0.0,
            "sharpe": float("nan"),
            "n": 0,
            "periods_per_year": ppy,
            "ruined": ruined,
            "liquidation_count": liquidation_count,
            "funding_events_applied": funding_events_applied,
            "funding_events_dropped": funding_events_dropped,
            "book_type": "usdtm_perp",
            "research_only": True,
            "live_pnl_claim": False,
        }
    if config.costs.frictionless:
        metrics["label"] = "FRICTIONLESS RESEARCH ONLY"
    metrics["data_source"] = "SYNTHETIC" if synthetic else "file"
    return BacktestResult(
        equity=eq,
        fills=pl.DataFrame(fill_rows) if fill_rows else pl.DataFrame(),
        metrics=metrics,
        frictionless=config.costs.frictionless,
        source_note="SYNTHETIC" if synthetic else "file",
    )
