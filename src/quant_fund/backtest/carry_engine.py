"""Delta-neutral funding-carry backtester: short perp + long spot per unit.

This is the honest version of the funding-carry trade — not a directional bet
on funding rates but the hedged book desks actually run:

- A *pair unit* for symbol s is ``long 1 coin spot + short 1 coin perp``.
  Directional PnL cancels: spot leg gains what the perp leg loses. Residual
  PnL is the basis move ``(spot − perp)`` plus funding receipts.
- Funding is cash: at each settlement, ``cash += units · perp_mark · rate``
  (short perp receives positive funding). Only positive-funding harvesting is
  modeled — negative funding would need spot borrow, which this book does not
  assume.
- The spot leg is fully paid from cash (no free leverage): order deltas are
  clamped so ``cash`` never drops below a margin buffer.
- The perp leg still carries maintenance margin: the wick-paranoid check marks
  the short at the bar's HIGH and the spot unwind at the bar's LOW, then
  force-unwinds pairs largest-first — the adverse ordering, since intra-bar
  sequencing is unknowable from OHLCV.

Same conventions as ``run_perp_backtest``: decisions on bar t fill at
t+1+fill_delay open, costs via ``total_cost`` on both legs, kill switch and
risk gate apply, and every result is ``research_only``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import polars as pl

from quant_fund.backtest.engine import (
    BacktestResult,
    StaleValuationError,
    _make_order,
    _target_weight_map,
    _valid_price,
)
from quant_fund.backtest.perp_engine import (
    OverlayScaler,
    _bar_enrichment,
    _funding_by_time,
    infer_periods_per_year,
)
from quant_fund.config.models import AppConfig
from quant_fund.execution.costs import total_cost
from quant_fund.metrics.analytics import book_diagnostics
from quant_fund.metrics.returns import cagr, max_drawdown, sharpe_ratio
from quant_fund.monitoring.kill_switch import KillSwitch
from quant_fund.portfolio.risk_gate import check_order
from quant_fund.schemas.errors import KillSwitchActive, RiskGateRejected

_CARRY_COLS = ("event_time", "security_id", "target_weight")


@dataclass
class CarryBook:
    """Cash wallet + paired positions.

    ``units[s]`` > 0 means: long ``units`` coins spot, short ``units`` coins
    perp at vwap ``perp_entry[s]``. Spot has no entry — its PnL flows through
    cash at trade time.
    """

    cash: float
    units: dict[str, float] = field(default_factory=dict)
    perp_entry: dict[str, float] = field(default_factory=dict)

    def equity(self, spot_marks: dict[str, float], perp_marks: dict[str, float]) -> float:
        upnl = sum(
            -u * (perp_marks[s] - self.perp_entry.get(s, perp_marks[s]))
            for s, u in self.units.items()
            if abs(u) > 1e-12 and s in perp_marks
        )
        spot_val = sum(
            u * spot_marks[s] for s, u in self.units.items() if abs(u) > 1e-12 and s in spot_marks
        )
        return self.cash + spot_val + upnl

    def perp_gross(self, perp_marks: dict[str, float]) -> float:
        return sum(
            abs(u) * perp_marks[s]
            for s, u in self.units.items()
            if abs(u) > 1e-12 and s in perp_marks
        )


def _pair_bars(
    perp_bars: pl.DataFrame, spot_bars: pl.DataFrame
) -> dict[datetime, dict[str, dict[str, float]]]:
    """Inner-join perp and spot bars on (event_time, security_id).

    A pair can only be marked or traded when both venues printed a bar in the
    same window — no cross-venue stale-price hedging.
    """
    p = _bar_enrichment(perp_bars).rename(
        {"open": "po", "high": "ph", "low": "pl", "close": "pc", "volume": "pv"}
    )
    s = _bar_enrichment(spot_bars).rename(
        {"open": "so", "high": "sh", "low": "sl", "close": "sc", "volume": "sv"}
    )
    keep_p = ["event_time", "security_id", "po", "ph", "pl", "pc", "adv", "vol_20"]
    keep_s = [
        "event_time",
        "security_id",
        "so",
        "sh",
        "sl",
        "sc",
        pl.col("adv").alias("sadv"),
        pl.col("vol_20").alias("svol"),
    ]
    joined = p.select(keep_p).join(s.select(keep_s), on=["event_time", "security_id"], how="inner")
    out: dict[datetime, dict[str, dict[str, float]]] = {}
    for row in joined.iter_rows(named=True):
        slot = out.setdefault(row["event_time"], {})
        sid = str(row["security_id"])
        if sid in slot:
            raise ValueError(f"duplicate paired bar for {sid} at {row['event_time']}")
        slot[sid] = row
    return out


def run_carry_backtest(
    perp_bars: pl.DataFrame,
    spot_bars: pl.DataFrame,
    funding: pl.DataFrame | None,
    weights: pl.DataFrame,
    config: AppConfig,
    *,
    initial_nav: float = 1_000_000.0,
    scaler: OverlayScaler | None = None,
    cash_buffer_frac: float = 0.02,
) -> BacktestResult:
    """`weights` = target fraction of equity per pair (target_weight ≥ 0 only).

    Negative weights are rejected — negative-funding harvest would need spot
    borrow this book does not model.
    """
    if weights.height:
        _ = _target_weight_map(weights)
    if (weights["target_weight"] < 0).any() if weights.height else False:
        raise ValueError("carry book is long-spot/short-perp only: target_weight must be >= 0")

    wmap_by_time: dict[datetime, dict[str, float]] = {}
    for wrow in weights.select(_CARRY_COLS).iter_rows(named=True):
        wmap_by_time.setdefault(wrow["event_time"], {})[str(wrow["security_id"])] = float(
            wrow["target_weight"]
        )

    by_time = _pair_bars(perp_bars, spot_bars)
    times = sorted(by_time)
    perp_cfg = config.perp
    ppy = (
        float(perp_cfg.periods_per_year_override)
        if perp_cfg.periods_per_year_override is not None
        else infer_periods_per_year(times, perp_cfg.bar_seconds_hint)
    )
    fund_map = (
        _funding_by_time(funding, multiplier=perp_cfg.funding_spike_multiplier)
        if perp_cfg.funding_enabled
        else {}
    )

    book = CarryBook(cash=float(initial_nav))
    pending: dict[str, tuple[float, datetime, int]] = {}
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
    last_spot: dict[str, float] = {}
    last_perp: dict[str, float] = {}
    mark_ages: dict[str, int] = {}
    kill = KillSwitch(config.kill_switch)
    ruined = False
    # Per-symbol P&L attribution: spot vwap cost lets every fill/liquidation
    # be decomposed into realized pair P&L vs funding vs fees, so a receipt
    # can show exactly where equity went instead of a bare NAV path.
    spot_vwap: dict[str, float] = {}
    attr: dict[str, dict[str, float]] = {}

    def _attr(sid: str) -> dict[str, float]:
        return attr.setdefault(
            sid, {"funding": 0.0, "realized": 0.0, "fees": 0.0, "liq_realized": 0.0, "liq_fee": 0.0}
        )

    for i, dt in enumerate(times):
        slot = by_time[dt]
        exec_spot: dict[str, float] = {}
        exec_perp: dict[str, float] = {}
        close_spot = dict(last_spot)
        close_perp = dict(last_perp)
        next_ages = dict(mark_ages)
        advs_p: dict[str, float] = {}
        vols_p: dict[str, float] = {}
        advs_s: dict[str, float] = {}
        vols_s: dict[str, float] = {}
        marked: set[str] = set()
        for sid, row in slot.items():
            po, pc = _valid_price(row["po"]), _valid_price(row["pc"])
            so, sc = _valid_price(row["so"]), _valid_price(row["sc"])
            if po is not None:
                exec_perp[sid] = po
            if so is not None:
                exec_spot[sid] = so
            if pc is not None and sc is not None:
                close_perp[sid] = pc
                close_spot[sid] = sc
                next_ages[sid] = 0
                marked.add(sid)
            advs_p[sid] = _valid_price(row.get("adv")) or 1.0
            vols_p[sid] = _valid_price(row.get("vol_20")) or 0.02
            advs_s[sid] = _valid_price(row.get("sadv")) or 1.0
            vols_s[sid] = _valid_price(row.get("svol")) or 0.02
        for sid in set(close_perp) & set(close_spot):
            if sid not in marked:
                next_ages[sid] = next_ages.get(sid, 0) + 1
        last_spot, last_perp, mark_ages = close_spot, close_perp, next_ages
        stale_held = {
            sid: mark_ages.get(sid)
            for sid, u in book.units.items()
            if abs(u) > 1e-12 and mark_ages.get(sid, 10**9) > config.risk_gate.stale_price_bars
        }
        if stale_held:
            details = ", ".join(f"{s}={a}" for s, a in stale_held.items())
            raise StaleValuationError("held carry pair valuation stale beyond limit: " + details)

        # --- execute pending pair orders at this bar's opens ---
        mark_p = {**last_perp, **exec_perp}
        mark_s = {**last_spot, **exec_spot}
        traded_turn = 0.0
        due = [sid for sid, idx in pending_exec_at.items() if idx <= i]
        for sid in sorted(due):
            target_w, signal_t, _ = pending.pop(sid)
            del pending_exec_at[sid]
            po = exec_perp.get(sid)
            so = exec_spot.get(sid)
            if po is None or so is None:
                continue  # cannot hedge without both prints — order lapses
            equity = book.equity(mark_s, mark_p)
            desired_units = max(0.0, target_w * equity / so)
            current = book.units.get(sid, 0.0)
            delta = desired_units - current
            if abs(delta) * so < 1.0:
                continue
            if delta > 0:
                # spot leg is paid from cash — clamp to liquidity minus buffer
                affordable = max(0.0, (book.cash - cash_buffer_frac * equity) / so)
                delta = min(delta, affordable)
                if delta * so < 1.0:
                    continue
            # leverage cap on the perp leg (gross notional vs equity)
            gross_now = book.perp_gross(mark_p)
            cap = perp_cfg.max_leverage * equity
            if gross_now + abs(delta) * po > cap and cap > gross_now:
                delta = np.sign(delta) * max(0.0, (cap - gross_now)) / po
            elif gross_now + abs(delta) * po > cap:
                margin_reject_count += 1
                continue
            if abs(delta) * po < 1.0:
                continue
            costs_p = total_cost(
                delta, po, advs_p.get(sid, 1.0), vols_p.get(sid, 0.02), config.costs
            )
            costs_s = total_cost(
                delta, so, advs_s.get(sid, 1.0), vols_s.get(sid, 0.02), config.costs
            )
            max_qty_p = config.costs.participation_limit * (advs_p.get(sid, 1.0) / po)
            max_qty_s = config.costs.participation_limit * (advs_s.get(sid, 1.0) / so)
            max_qty = min(max_qty_p, max_qty_s)
            if abs(delta) > max_qty > 0:
                delta = np.sign(delta) * max_qty
                costs_p = total_cost(
                    delta, po, advs_p.get(sid, 1.0), vols_p.get(sid, 0.02), config.costs
                )
                costs_s = total_cost(
                    delta, so, advs_s.get(sid, 1.0), vols_s.get(sid, 0.02), config.costs
                )
            try:
                kill.assert_new_orders_allowed()
            except KillSwitchActive:
                halt_count += 1
                continue
            nav_safe = max(equity, 1e-12)
            current_w = current * so / nav_safe
            order_seq += 1
            order = _make_order(
                sid=sid, delta=delta, signal_time=signal_t, order_time=dt, order_seq=order_seq
            )
            try:
                check_order(
                    order,
                    nav=equity,
                    price=so,
                    current_weight=current_w,
                    gross_after=0.0,  # pair book: net market exposure ≈ 0 by construction
                    net_after=0.0,
                    participation=abs(delta) * po / max(advs_p.get(sid, 1.0), 1e-12),
                    predicted_vol=vols_p.get(sid, 0.02),
                    config=config,
                )
            except RiskGateRejected:
                reject_count += 1
                continue
            fee = float(costs_p["total"]) + float(costs_s["total"])
            new_units = current + delta
            # cash: spot leg pays/receives full notional; perp leg only settles
            # realized PnL on quantity reductions (same convention as perp engine)
            book.cash -= delta * so
            old_entry = book.perp_entry.get(sid, po)
            old_spot = spot_vwap.get(sid, so)
            if delta < 0:
                book.cash += (-delta) * (old_entry - po)  # buy back short: realize
                _attr(sid)["realized"] += (-delta) * (so - old_spot) + (-delta) * (old_entry - po)
            if abs(current) < 1e-12:
                book.perp_entry[sid] = po
                spot_vwap[sid] = so
            elif abs(new_units) > abs(current):
                book.perp_entry[sid] = (current * old_entry + delta * po) / new_units
                spot_vwap[sid] = (current * old_spot + delta * so) / new_units
            elif abs(new_units) < 1e-12:
                book.perp_entry.pop(sid, None)
                spot_vwap.pop(sid, None)
            if abs(new_units) < 1e-12:
                book.units.pop(sid, None)
            else:
                book.units[sid] = new_units
            book.cash -= fee
            _attr(sid)["fees"] += fee
            traded_turn += abs(delta) * so / nav_safe
            for k in ("commission", "spread", "impact"):
                cost_sum[k] += float(costs_p[k]) + float(costs_s[k])
            fill_rows.append(
                {
                    "fill_time": dt,
                    "signal_time": signal_t,
                    "security_id": sid,
                    "quantity": delta,
                    "spot_price": so,
                    "perp_price": po,
                    "fee": float(costs_p["commission"]) + float(costs_s["commission"]),
                    "spread_cost": float(costs_p["spread"]) + float(costs_s["spread"]),
                    "impact_cost": float(costs_p["impact"]) + float(costs_s["impact"]),
                }
            )

        # --- funding on the perp leg ---
        for sid, rate in fund_map.get(dt, []):
            u = book.units.get(sid, 0.0)
            if abs(u) < 1e-12:
                continue
            mark = mark_p.get(sid) or last_perp.get(sid)
            if mark is None:
                continue
            flow = u * mark * rate  # short perp: positive rate → receive
            book.cash += flow
            _attr(sid)["funding"] += flow
            if flow >= 0:
                funding_received += flow
            else:
                funding_paid += -flow
            funding_events_applied += 1

        # --- wick-paranoid liquidation on the perp leg ---
        # The venue can tag the SHORT at the bar's adverse excursion (perp
        # high). The hedge leg is ours to unwind, so it marks at the bar's
        # close — using the same bar's spot LOW while the perp sits at its
        # HIGH assumes two opposite extremes filled simultaneously on two
        # venues, fabricating a cross-venue basis that never traded.
        liq_perp = dict(last_perp)
        liq_spot = dict(last_spot)
        if perp_cfg.liquidation_on_wick:
            for sid, row in slot.items():
                if abs(book.units.get(sid, 0.0)) < 1e-12:
                    continue
                liq_perp[sid] = _valid_price(row["ph"]) or last_perp.get(sid, 0.0)

        def _mmr_deficit(lp: dict[str, float] = liq_perp, ls: dict[str, float] = liq_spot) -> float:
            return perp_cfg.maint_margin_ratio * book.perp_gross(lp) - book.equity(ls, lp)

        while _mmr_deficit() > 0 and book.perp_gross(liq_perp) > 0:
            victim = max(
                (s for s in book.units if abs(book.units[s]) > 1e-12),
                key=lambda s: abs(book.units[s]) * liq_perp.get(s, 0.0),
                default=None,
            )
            if victim is None:
                break
            u = book.units.pop(victim)
            entry = book.perp_entry.pop(victim, liq_perp.get(victim, 0.0))
            s_vwap = spot_vwap.pop(victim, liq_spot.get(victim, 0.0))
            liq_s = liq_spot.get(victim, 0.0)
            liq_p = liq_perp.get(victim, entry)
            # venue force-covers the short at the adverse liq price; the
            # orphaned hedge is unwound at market (bar close) with normal
            # costs, plus the liquidation fee on the perp leg
            book.cash += u * liq_s  # dump spot
            book.cash += u * (entry - liq_p)  # cover short
            liq_spot_costs = total_cost(
                u, liq_s, advs_s.get(victim, 1.0), vols_s.get(victim, 0.02), config.costs
            )
            book.cash -= float(liq_spot_costs["total"])
            liq_fee = u * liq_perp.get(victim, 0.0) * perp_cfg.liquidation_fee_bps / 1e4
            book.cash -= liq_fee
            liq_realized = (
                u * (liq_s - s_vwap) + u * (entry - liq_p) - float(liq_spot_costs["total"])
            )
            _attr(victim)["liq_realized"] += liq_realized
            _attr(victim)["liq_fee"] += liq_fee
            for k in ("commission", "spread", "impact"):
                cost_sum[k] += float(liq_spot_costs[k])
            liquidation_cost += liq_fee
            liquidation_count += 1
            liq_rows.append(
                {
                    "event_time": dt,
                    "security_id": victim,
                    "quantity": u,
                    "liq_perp_price": liq_perp.get(victim),
                    "liq_spot_price": liq_spot.get(victim),
                    "fee": liq_fee,
                    "realized_pnl": liq_realized,
                    "perp_entry": entry,
                    "spot_vwap": s_vwap,
                }
            )

        # --- mark close, queue targets ---
        nav_close = book.equity(last_spot, last_perp)
        gross_p = book.perp_gross(last_perp) / max(abs(nav_close), 1e-12)
        navs.append(
            {
                "event_time": dt,
                "nav": nav_close,
                "gross": gross_p,
                "net": gross_p,  # pair book: gross==net exposure metric is the perp leg
                "turnover": traded_turn,
            }
        )
        if scaler is not None:
            scaler.observe(dt, nav_close)
        if nav_close <= 0:
            ruined = True
            break
        if i + 1 + perp_cfg.fill_delay_bars < len(times):
            targets = wmap_by_time.get(dt)
            if targets:
                if scaler is not None:
                    targets = scaler.scale(dt, dict(targets))
                exec_idx = i + 1 + perp_cfg.fill_delay_bars
                for sid, w in targets.items():
                    pending[sid] = (w, dt, exec_idx)
                    pending_exec_at[sid] = exec_idx

    eq = pl.DataFrame(navs) if navs else pl.DataFrame({"event_time": [], "nav": []})
    # Per-symbol attribution: unrealized for still-open pairs marks at last
    # close vs vwap entries; realized/funding/fees/liq were tracked at events.
    for sid, u in book.units.items():
        if abs(u) < 1e-12:
            continue
        s_now = last_spot.get(sid)
        p_now = last_perp.get(sid)
        if s_now is None or p_now is None:
            continue
        _attr(sid)["unrealized"] = u * (s_now - spot_vwap.get(sid, s_now)) + u * (
            book.perp_entry.get(sid, p_now) - p_now
        )
    attr_totals = {
        k: sum(row.get(k, 0.0) for row in attr.values())
        for k in ("funding", "realized", "fees", "liq_realized", "liq_fee", "unrealized")
    }
    final_equity = float(navs[-1]["nav"]) if navs else float(initial_nav)
    conservation_error = (
        final_equity
        - float(initial_nav)
        - (
            attr_totals["funding"]
            + attr_totals["realized"]
            + attr_totals["liq_realized"]
            - attr_totals["fees"]
            - attr_totals["liq_fee"]
            + attr_totals["unrealized"]
        )
    )
    pnl_attribution = {
        "by_symbol": {sid: row for sid, row in sorted(attr.items())},
        "totals": attr_totals,
        "conservation_error": conservation_error,
        "liquidation_events": liq_rows,
    }
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
            label="CARRY_BACKTEST_SIM",
            data_source="file",
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
            "funding_received_total": funding_received,
            "funding_paid_total": funding_paid,
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
            "book_type": "delta_neutral_carry",
            "pnl_attribution": pnl_attribution,
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
            "book_type": "delta_neutral_carry",
            "pnl_attribution": pnl_attribution,
            "research_only": True,
            "live_pnl_claim": False,
        }
    if config.costs.frictionless:
        metrics["label"] = "FRICTIONLESS RESEARCH ONLY"
    metrics["data_source"] = "file"
    return BacktestResult(
        equity=eq,
        fills=pl.DataFrame(fill_rows) if fill_rows else pl.DataFrame(),
        metrics=metrics,
        frictionless=config.costs.frictionless,
        source_note="delta-neutral carry: short perp + long spot, funding harvest",
    )
