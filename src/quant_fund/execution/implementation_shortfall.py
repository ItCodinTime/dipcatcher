"""Implementation shortfall (Perold) decomposition.

Splits each executed fill's cost into the signed price drift between the
decision price and the execution price plus explicit costs (fee, spread,
impact). The signed convention: for a BUY, ``(exec - decision) * qty`` is
adverse when positive; for a SELL, ``(decision - exec) * qty``. Negative
drift is favorable execution and is reported, not clipped.

Research / TCA diagnostic only — never a live P&L claim.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl


def fill_shortfall(
    *,
    side_sign: float,
    quantity: float,
    decision_price: float,
    exec_price: float,
    fee: float = 0.0,
    spread_cost: float = 0.0,
    impact_cost: float = 0.0,
) -> dict[str, float]:
    """Signed implementation-shortfall components for one fill, in dollars.

    ``side_sign`` is +1 for buys, -1 for sells (the signed trade direction).
    All quantities must be finite; prices strictly positive; quantity
    positive; explicit costs non-negative.
    """
    q = float(quantity)
    dec = float(decision_price)
    exe = float(exec_price)
    sgn = float(side_sign)
    if not np.isfinite(q) or q <= 0:
        raise ValueError("quantity must be finite and strictly positive")
    if not (np.isfinite(dec) and dec > 0 and np.isfinite(exe) and exe > 0):
        raise ValueError("decision and execution prices must be finite and positive")
    if sgn not in (-1.0, 1.0):
        raise ValueError("side_sign must be +1 (buy) or -1 (sell)")
    explicit = {}
    for name, value in (("fee", fee), ("spread_cost", spread_cost), ("impact_cost", impact_cost)):
        v = float(value)
        if not np.isfinite(v) or v < 0:
            raise ValueError(f"{name} must be finite and non-negative")
        explicit[name] = v
    drift = sgn * (exe - dec) * q
    notional = q * dec
    total = drift + explicit["fee"] + explicit["spread_cost"] + explicit["impact_cost"]
    return {
        "drift": float(drift),
        "explicit": float(sum(explicit.values())),
        **explicit,
        "total_is": float(total),
        "notional": float(notional),
        "is_bps": float(total / notional * 1e4),
        "drift_bps": float(drift / notional * 1e4),
    }


def shortfall_frame(fills: pl.DataFrame) -> pl.DataFrame:
    """Per-fill IS decomposition for a fills frame.

    Required columns: ``security_id, quantity, decision_price, price``.
    ``quantity`` may be signed (sell < 0) or unsigned with an explicit
    ``side_sign`` (+1 buy / -1 sell) column — the frame is normalized to
    unsigned quantity + side sign internally. Optional explicit-cost
    columns: ``fee, spread_cost, impact_cost``. Adds ``drift, explicit,
    total_is, notional, is_bps, drift_bps`` columns.
    """
    required = {"security_id", "quantity", "decision_price", "price"}
    missing = required - set(fills.columns)
    if missing:
        raise ValueError(f"fills frame missing columns: {sorted(missing)}")
    if "side_sign" not in fills.columns:
        fills = fills.with_columns(
            pl.when(pl.col("quantity") >= 0).then(1.0).otherwise(-1.0).alias("side_sign"),
            pl.col("quantity").abs(),
        )
    if fills.height == 0:
        return fills.with_columns(
            [
                pl.lit(None, dtype=pl.Float64).alias(c)
                for c in ("drift", "explicit", "total_is", "notional", "is_bps", "drift_bps")
            ]
        )
    fee = pl.col("fee").fill_null(0.0) if "fee" in fills.columns else pl.lit(0.0)
    spread = pl.col("spread_cost").fill_null(0.0) if "spread_cost" in fills.columns else pl.lit(0.0)
    impact = pl.col("impact_cost").fill_null(0.0) if "impact_cost" in fills.columns else pl.lit(0.0)
    return fills.with_columns(
        (
            pl.col("side_sign") * (pl.col("price") - pl.col("decision_price")) * pl.col("quantity")
        ).alias("drift"),
        (fee + spread + impact).alias("explicit"),
        (pl.col("quantity") * pl.col("decision_price")).alias("notional"),
    ).with_columns(
        (pl.col("drift") + pl.col("explicit")).alias("total_is"),
        ((pl.col("drift") + pl.col("explicit")) / pl.col("notional") * 1e4).alias("is_bps"),
        (pl.col("drift") / pl.col("notional") * 1e4).alias("drift_bps"),
    )


def aggregate_shortfall(frame: pl.DataFrame) -> dict[str, Any]:
    """Aggregate a ``shortfall_frame`` output into a TCA summary."""
    if frame.height == 0:
        return {
            "n_fills": 0,
            "total_is": 0.0,
            "drift": 0.0,
            "explicit": 0.0,
            "notional": 0.0,
            "is_bps": None,
            "live_pnl_claim": False,
            "research_only": True,
        }
    _require = {"drift", "explicit", "total_is", "notional", "side_sign", "security_id"}
    missing = _require - set(frame.columns)
    if missing:
        raise ValueError(f"shortfall frame missing columns: {sorted(missing)}")
    total_is = float(frame["total_is"].sum())
    drift = float(frame["drift"].sum())
    explicit = float(frame["explicit"].sum())
    notional = float(frame["notional"].sum())
    by_side = (
        frame.group_by("side_sign")
        .agg(
            pl.col("total_is").sum().alias("total_is"),
            pl.col("notional").sum().alias("notional"),
            pl.len().alias("n_fills"),
        )
        .sort("side_sign", descending=True)
        .to_dicts()
    )
    by_name = (
        frame.group_by("security_id")
        .agg(
            pl.col("total_is").sum().alias("total_is"),
            pl.col("notional").sum().alias("notional"),
            pl.len().alias("n_fills"),
        )
        .sort("total_is", descending=True)
        .to_dicts()
    )
    favorable = float(frame.filter(pl.col("total_is") < 0)["total_is"].sum() or 0.0)
    return {
        "n_fills": int(frame.height),
        "total_is": total_is,
        "drift": drift,
        "explicit": explicit,
        "favorable_is": favorable,
        "notional": notional,
        "is_bps": (total_is / notional * 1e4) if notional > 0 else None,
        "drift_bps": (drift / notional * 1e4) if notional > 0 else None,
        "by_side": by_side,
        "top_cost_names": by_name[:10],
        "convention": "signed drift + explicit costs; negative = favorable",
        "live_pnl_claim": False,
        "research_only": True,
    }
