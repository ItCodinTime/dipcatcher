"""Per-name and per-sleeve P&L attribution.

Decomposes book returns into per-security contributions using the causal
weight convention of the backtester: a target weight decided at bar ``t``
is executed at the next bar and earns the close-to-close return realized
*on* bar ``t + 1``. Contribution of name ``i`` at bar ``t`` is therefore
``w_{i,t-1} * r_{i,t}``; cost attribution adds the executed fill costs of
bar ``t`` as a NAV fraction.

Outputs are research diagnostics only — ``live_pnl_claim`` is always
false. Weight-return attribution is the standard approximation: it does
not model intrabar path, partial fills, or cash drag.
"""

from __future__ import annotations

import numpy as np
import polars as pl

_WEIGHT_COLS = {"event_time", "security_id", "target_weight"}
_BAR_COLS = {"event_time", "security_id", "close"}
_COST_COLS = ("fee", "spread_cost", "impact_cost")


def _require(frame: pl.DataFrame, cols: set[str], name: str) -> None:
    missing = cols - set(frame.columns)
    if missing:
        raise ValueError(f"{name} missing columns: {sorted(missing)}")


def security_returns(bars: pl.DataFrame) -> pl.DataFrame:
    """Close-to-close simple returns per security, aligned to the bar's own
    ``event_time`` (the return realized on that bar)."""
    _require(bars, _BAR_COLS, "bars")
    return (
        bars.select("security_id", "event_time", "close")
        .sort(["security_id", "event_time"])
        .with_columns(
            (pl.col("close") / pl.col("close").shift(1) - 1.0)
            .over("security_id")
            .alias("ret")
        )
        .select("security_id", "event_time", "ret")
    )


def attribute_weights_pnl(
    weights: pl.DataFrame,
    bars: pl.DataFrame,
    fills: pl.DataFrame | None = None,
    nav: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Per-(event_time, security_id) P&L attribution frame.

    Columns: ``event_time, security_id, weight_prev, ret, gross_contrib,
    cost_ret, net_contrib``. ``gross_contrib = w_{i,t-1} * r_{i,t}``;
    ``cost_ret`` is executed fill cost at ``t`` divided by NAV at ``t``
    (zero when no fills/nav are supplied); ``net_contrib`` is the
    difference. Rows where the prior weight is zero are dropped.
    """
    _require(weights, _WEIGHT_COLS, "weights")
    _require(bars, _BAR_COLS, "bars")

    rets = security_returns(bars)
    # Effective weight during bar t = last declared target at t-1 or earlier
    # (a target decided at t executes at t+1, matching the engine). Asof the
    # declarations onto the bar grid, then shift one bar per security.
    grid = rets.select("security_id", "event_time").sort(["security_id", "event_time"])
    declared = weights.select("security_id", "event_time", "target_weight").sort(
        ["security_id", "event_time"]
    )
    w_prev = (
        grid.join_asof(declared, on="event_time", by="security_id", strategy="backward")
        .with_columns(pl.col("target_weight").shift(1).over("security_id").alias("weight_prev"))
        .select("security_id", "event_time", "weight_prev")
    )
    frame = rets.join(w_prev, on=["security_id", "event_time"], how="inner").with_columns(
        (pl.col("weight_prev") * pl.col("ret")).alias("gross_contrib")
    )

    if fills is not None and fills.height > 0:
        _require(fills, {"fill_time", "security_id"}, "fills")
        present = [c for c in _COST_COLS if c in fills.columns]
        if not present:
            raise ValueError("fills carry none of the cost columns fee/spread_cost/impact_cost")
        cost_expr = pl.sum_horizontal(
            [pl.col(c).fill_null(0.0).sum() for c in present]
        )
        cost_by = (
            fills.with_columns(pl.col("fill_time").alias("event_time"))
            .group_by(["security_id", "event_time"])
            .agg(cost_expr.alias("cost_dollars"))
        )
        if nav is not None and nav.height > 0:
            _require(nav, {"event_time", "nav"}, "nav")
            cost_by = cost_by.join(nav.select("event_time", "nav"), on="event_time", how="left")
            cost_by = cost_by.with_columns(
                pl.when(pl.col("nav") > 0)
                .then(pl.col("cost_dollars") / pl.col("nav"))
                .otherwise(0.0)
                .alias("cost_ret")
            ).select("security_id", "event_time", "cost_ret")
        else:
            cost_by = cost_by.with_columns(pl.lit(0.0).alias("cost_ret")).select(
                "security_id", "event_time", "cost_ret"
            )
        frame = frame.join(cost_by, on=["security_id", "event_time"], how="left")
    if "cost_ret" not in frame.columns:
        frame = frame.with_columns(pl.lit(0.0).alias("cost_ret"))
    else:
        frame = frame.with_columns(pl.col("cost_ret").fill_null(0.0))

    frame = frame.with_columns(
        (pl.col("gross_contrib") - pl.col("cost_ret")).alias("net_contrib")
    )
    return frame.filter(pl.col("weight_prev") != 0.0).sort(["event_time", "security_id"])


def name_attribution(frame: pl.DataFrame) -> pl.DataFrame:
    """Aggregate the attribution frame per security.

    Columns: ``security_id, gross_pnl, cost, net_pnl, n_bars, mean_ret,
    share_of_gross_abs`` (share of the book's absolute gross contribution —
    measures where P&L mass concentrates, sums to 1).
    """
    _require(
        frame,
        {"security_id", "gross_contrib", "cost_ret", "net_contrib", "ret"},
        "attribution frame",
    )
    out = frame.group_by("security_id").agg(
        pl.col("gross_contrib").sum().alias("gross_pnl"),
        pl.col("cost_ret").sum().alias("cost"),
        pl.col("net_contrib").sum().alias("net_pnl"),
        pl.len().alias("n_bars"),
        pl.col("ret").mean().alias("mean_ret"),
    )
    denom = float(np.abs(out["gross_pnl"].to_numpy()).sum()) if out.height else 0.0
    if denom > 0:
        out = out.with_columns((pl.col("gross_pnl").abs() / denom).alias("share_of_gross_abs"))
    else:
        out = out.with_columns(pl.lit(None).alias("share_of_gross_abs"))
    return out.sort("net_pnl", descending=True)


def sleeve_attribution(
    frame: pl.DataFrame,
    sleeve_map: dict[str, str] | pl.DataFrame,
) -> pl.DataFrame:
    """Aggregate attribution by sleeve label.

    ``sleeve_map`` is a ``{security_id: sleeve}`` dict or a frame with
    ``security_id, sleeve`` columns. Names not present in the map are
    grouped under ``"unmapped"``.
    """
    _require(frame, {"security_id", "gross_contrib", "cost_ret", "net_contrib"}, "frame")
    if isinstance(sleeve_map, dict):
        smap = pl.DataFrame(
            {
                "security_id": list(sleeve_map.keys()),
                "sleeve": list(sleeve_map.values()),
            }
        )
    else:
        _require(sleeve_map, {"security_id", "sleeve"}, "sleeve_map")
        smap = sleeve_map.select("security_id", "sleeve")
    joined = frame.join(smap, on="security_id", how="left").with_columns(
        pl.col("sleeve").fill_null("unmapped")
    )
    out = joined.group_by("sleeve").agg(
        pl.col("gross_contrib").sum().alias("gross_pnl"),
        pl.col("cost_ret").sum().alias("cost"),
        pl.col("net_contrib").sum().alias("net_pnl"),
        pl.len().alias("n_bars"),
    )
    denom = float(np.abs(out["gross_pnl"].to_numpy()).sum()) if out.height else 0.0
    if denom > 0:
        out = out.with_columns((pl.col("gross_pnl").abs() / denom).alias("share_of_gross_abs"))
    else:
        out = out.with_columns(pl.lit(None).alias("share_of_gross_abs"))
    return out.sort("net_pnl", descending=True)


def attribution_summary(
    frame: pl.DataFrame,
    *,
    sleeve_map: dict[str, str] | pl.DataFrame | None = None,
    top_n: int = 10,
) -> dict[str, object]:
    """Roll the attribution frame into a receipt-ready summary dict."""
    names = name_attribution(frame)
    total_gross = float(frame["gross_contrib"].sum()) if frame.height else 0.0
    total_cost = float(frame["cost_ret"].sum()) if frame.height else 0.0
    total_net = float(frame["net_contrib"].sum()) if frame.height else 0.0
    top = names.head(max(int(top_n), 0)) if names.height else names
    bottom = names.tail(max(int(top_n), 0)) if names.height else names
    out: dict[str, object] = {
        "total_gross_pnl": total_gross,
        "total_cost": total_cost,
        "total_net_pnl": total_net,
        "n_names": int(names.height),
        "cost_share_of_gross": (total_cost / abs(total_gross)) if abs(total_gross) > 0 else None,
        "top_contributors": top.to_dicts(),
        "bottom_contributors": bottom.to_dicts(),
        "convention": "weight_prev_x_ret approximation",
        "live_pnl_claim": False,
        "research_only": True,
    }
    if sleeve_map is not None:
        out["by_sleeve"] = sleeve_attribution(frame, sleeve_map).to_dicts()
    return out
