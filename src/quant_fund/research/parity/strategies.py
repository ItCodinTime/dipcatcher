"""Reference strategies as target-weight panels.

A strategy emits sparse ``(event_time, security_id, target_weight)`` rows on
its decision dates — the native engine carries the last complete target book
forward between rows and re-targets it at every execution bar. Adapters must
reproduce that carry-forward explicitly; ``carried_weight_maps`` is the shared
helper that materializes the dense per-bar target book every engine replays.

All strategies are long-only with total weight <= ``total_frac`` (< 1.0) so a
cash buffer survives order sequencing in every engine — otherwise identical
strategies diverge for a trivial reason (buy fills before the matching sell
frees cash), which is a documented convention diff, not an alpha claim.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable

import numpy as np
import polars as pl

TOTAL_WEIGHT = 0.95


@dataclass
class StrategyPlan:
    """A named strategy + its sparse signal-date weight rows."""

    name: str
    weights: pl.DataFrame  # columns: event_time, security_id, target_weight
    params: dict[str, object] = field(default_factory=dict)


def _frame(rows: list[dict]) -> pl.DataFrame:
    if not rows:
        return pl.DataFrame(
            schema={
                "event_time": pl.Datetime("us", "UTC"),
                "security_id": pl.String,
                "target_weight": pl.Float64,
            }
        )
    return pl.DataFrame(rows).with_columns(
        pl.col("event_time").cast(pl.Datetime("us", "UTC")),
        pl.col("security_id").cast(pl.String),
        pl.col("target_weight").cast(pl.Float64),
    )


def carried_weight_maps(
    weights: pl.DataFrame,
    dates: list[datetime],
) -> list[dict[str, float]]:
    """Dense per-bar target book: last complete row set, carried forward."""
    by_date: dict[datetime, dict[str, float]] = {}
    for row in weights.iter_rows(named=True):
        by_date.setdefault(row["event_time"], {})[str(row["security_id"])] = float(
            row["target_weight"]
        )
    out: list[dict[str, float]] = []
    last: dict[str, float] = {}
    for dt in dates:
        if dt in by_date:
            last = by_date[dt]
        out.append(dict(last))
    return out


def buy_hold_equal_weight(panel: pl.DataFrame, total_frac: float = TOTAL_WEIGHT) -> StrategyPlan:
    """Equal-weight all names once, at the first bar's close.

    Executed at the second bar's open (NEXT_OPEN convention); the carried
    target is then re-applied every bar — i.e. daily-rebalanced equal weight,
    matching the native engine's re-target semantics exactly.
    """
    dates = sorted(panel["event_time"].unique().to_list())
    sids = sorted(panel["security_id"].unique().to_list())
    if not dates or not sids:
        raise ValueError("empty panel")
    w = total_frac / len(sids)
    rows = [
        {"event_time": dates[0], "security_id": s, "target_weight": w} for s in sids
    ]
    return StrategyPlan(
        name="buy_hold_equal",
        weights=_frame(rows),
        params={"total_frac": total_frac, "n_names": len(sids)},
    )


def sma_crossover(
    panel: pl.DataFrame,
    fast: int = 5,
    slow: int = 20,
    total_frac: float = TOTAL_WEIGHT,
) -> StrategyPlan:
    """Per-name trend filter: weight = total_frac/N while SMA(fast) > SMA(slow).

    Emits a complete target book on every bar once the slow window is defined
    (each decision row is a full book replacement in the native engine).
    """
    if not 1 <= fast < slow:
        raise ValueError("need 1 <= fast < slow")
    dates = sorted(panel["event_time"].unique().to_list())
    sids = sorted(panel["security_id"].unique().to_list())
    px = (
        panel.select("event_time", "security_id", "close")
        .sort("security_id", "event_time")
        .pivot(on="security_id", index="event_time", values="close")
        .sort("event_time")
    )
    close = px.select(sids).to_numpy()
    n = len(sids)
    w = total_frac / n
    rows: list[dict] = []
    for i in range(slow - 1, len(dates)):
        for j, s in enumerate(sids):
            fma = float(np.mean(close[i - fast + 1 : i + 1, j]))
            sma = float(np.mean(close[i - slow + 1 : i + 1, j]))
            rows.append(
                {
                    "event_time": dates[i],
                    "security_id": s,
                    "target_weight": w if fma > sma else 0.0,
                }
            )
    return StrategyPlan(
        name="sma_crossover",
        weights=_frame(rows),
        params={"fast": fast, "slow": slow, "total_frac": total_frac},
    )


def momentum_topk(
    panel: pl.DataFrame,
    lookback: int = 20,
    top_k: int = 5,
    rebalance: int = 21,
    total_frac: float = TOTAL_WEIGHT,
) -> StrategyPlan:
    """Fixed-fraction momentum: total_frac/K to each of the top-K by trailing return.

    Sparse rows only on rebalance dates; the engine carries them between
    rebalances (``carried_weight_maps`` makes that explicit for adapters).
    """
    if lookback < 1 or top_k < 1 or rebalance < 1:
        raise ValueError("lookback, top_k, rebalance must be >= 1")
    dates = sorted(panel["event_time"].unique().to_list())
    sids = sorted(panel["security_id"].unique().to_list())
    px = (
        panel.select("event_time", "security_id", "close")
        .sort("security_id", "event_time")
        .pivot(on="security_id", index="event_time", values="close")
        .sort("event_time")
    )
    close = px.select(sids).to_numpy()
    w = total_frac / min(top_k, len(sids))
    rows: list[dict] = []
    for i in range(lookback, len(dates)):
        if (i - lookback) % rebalance != 0:
            continue
        rets = close[i, :] / close[i - lookback, :] - 1.0
        order = np.argsort(-rets)
        winners = set(order[: min(top_k, len(sids))].tolist())
        for j, s in enumerate(sids):
            rows.append(
                {
                    "event_time": dates[i],
                    "security_id": s,
                    "target_weight": w if j in winners else 0.0,
                }
            )
    return StrategyPlan(
        name="momentum_topk",
        weights=_frame(rows),
        params={
            "lookback": lookback,
            "top_k": top_k,
            "rebalance": rebalance,
            "total_frac": total_frac,
        },
    )


STRATEGIES: dict[str, Callable[..., StrategyPlan]] = {
    "buy_hold_equal": buy_hold_equal_weight,
    "sma_crossover": sma_crossover,
    "momentum_topk": momentum_topk,
}


def build_plan(name: str, panel: pl.DataFrame, **kwargs: object) -> StrategyPlan:
    if name not in STRATEGIES:
        raise KeyError(f"unknown strategy {name!r}; have {sorted(STRATEGIES)}")
    return STRATEGIES[name](panel, **kwargs)  # type: ignore[arg-type]
