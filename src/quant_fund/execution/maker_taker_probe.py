"""Maker-vs-taker execution probe on a pair of ``SimulatedBroker`` books.

Lives in ``execution`` (not ``research``) because it drives the simulated
broker directly — the layering gate forbids ``research`` modules importing
``execution.simulated_broker``. Consumed by ``research/cost_surface`` for
the P5.4 maker/taker fee-delta leg.
"""

from __future__ import annotations

from typing import Any

import polars as pl

from quant_fund.config.models import AppConfig, CostConfig
from quant_fund.execution.simulated_broker import SimulatedBroker

_EPS = 1e-12


def maker_taker_fee_delta(
    bars: pl.DataFrame,
    targets: pl.DataFrame,
    costs: CostConfig,
) -> dict[str, float]:
    """Taker vs maker fee delta for one book on a SimulatedBroker pair.

    Each bar: taker path submits marketable orders at the open; the maker
    path rests limits pegged at the previous close and sweeps them against
    the bar's range. Reports fees per filled dollar, fill rate and the
    unfilled residual — the fill-risk leg of the maker/taker trade-off.
    """
    cfg = AppConfig(costs=costs)
    taker = SimulatedBroker(config=cfg, id_factory=None)
    maker = SimulatedBroker(config=cfg, id_factory=None)
    taker_notional = 0.0
    taker_fees = 0.0
    submitted_notional = 0.0
    by_date: dict[Any, list[dict[str, Any]]] = {}
    for row in bars.iter_rows(named=True):
        by_date.setdefault(row["event_time"], []).append(row)
    tgt_by_date: dict[Any, dict[str, float]] = {}
    for row in targets.iter_rows(named=True):
        tgt_by_date.setdefault(row["event_time"], {})[row["security_id"]] = float(
            row["target_weight"]
        )
    prev_close: dict[str, float] = {}
    for dt in sorted(by_date):
        day = by_date[dt]
        marks = {r["security_id"]: float(r["close"]) for r in day}
        opens = {r["security_id"]: float(r["open"]) for r in day}
        tw = tgt_by_date.get(dt, {})
        advs = {r["security_id"]: float(r["adv"]) for r in day}
        # Taker: marketable orders at this bar's open.
        for order in taker.target_to_orders(tw, opens, signal_time=dt, order_time=dt):
            rec = taker.submit(
                order,
                price=opens[order.security_id],
                adv_dollars=advs[order.security_id],
            )
            if rec.fill is not None:
                taker_fees += rec.fill.fee + rec.fill.spread_cost + rec.fill.impact_cost
                taker_notional += rec.fill.quantity * rec.fill.price
        # Maker: limits pegged at the prior close rest until a bar touches.
        for order in maker.target_to_orders(tw, prev_close or opens, signal_time=dt, order_time=dt):
            peg = prev_close.get(order.security_id, opens[order.security_id])
            order = order.model_copy(update={"limit_price": peg})
            submitted_notional += order.quantity * peg
            maker.submit(
                order,
                price=peg,
                adv_dollars=advs[order.security_id],
            )
        for r in day:
            maker.process_bar(
                r["security_id"],
                bar_open=float(r["open"]),
                bar_high=float(r["high"]),
                bar_low=float(r["low"]),
                bar_time=dt,
                adv_dollars=float(r["adv"]),
            )
        prev_close = marks
    maker_fees = float(sum(f.fee + f.spread_cost + f.impact_cost for f in maker.fills))
    maker_notional = float(sum(f.quantity * f.price for f in maker.fills))
    fill_rate = maker_notional / max(submitted_notional, _EPS)
    return {
        "taker_fees": float(taker_fees),
        "maker_fees": float(maker_fees),
        "taker_fee_bps_per_filled": 1e4 * taker_fees / max(taker_notional, _EPS),
        "maker_fee_bps_per_filled": 1e4 * maker_fees / max(maker_notional, _EPS),
        "maker_fill_rate": float(fill_rate),
        "unfilled_notional_frac": float(1.0 - min(fill_rate, 1.0)),
    }
