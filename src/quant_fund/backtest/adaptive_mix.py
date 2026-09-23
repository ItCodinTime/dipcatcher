"""Causal allocation across independently simulated, costed signal sleeves.

Inputs are research backtest outputs. The allocator observes each paper sleeve's
NAV at a completed close and emits weights for the following open.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import polars as pl


def dense_targets(bars: pl.DataFrame, targets: pl.DataFrame) -> pl.DataFrame:
    """Interpret missing per-bar proposals as flat, with explicit zero exits.

    This is for sleeves that state a desired position at every bar. It must not
    be used for sparse event-driven strategies where absence means "hold".
    """
    keys = ["event_time", "security_id"]
    if not set(keys).issubset(bars.columns) or not set(keys + ["target_weight"]).issubset(
        targets.columns
    ):
        raise ValueError("bars or targets missing required columns")
    grid = bars.select(keys).unique().sort(keys)
    if grid.height != bars.height:
        raise ValueError("duplicate bars for event_time/security_id")
    if targets.select(keys).unique().height != targets.height:
        raise ValueError("duplicate targets for event_time/security_id")
    if targets.height and targets.join(grid, on=keys, how="anti").height:
        raise ValueError("target outside bar panel")
    if targets.height and targets.filter(~pl.col("target_weight").is_finite()).height:
        raise ValueError("non-finite target weight")
    return grid.join(targets.select(keys + ["target_weight"]), on=keys, how="left").with_columns(
        pl.col("target_weight").fill_null(0.0)
    )


def trailing_nav_allocations(
    equities: Mapping[str, pl.DataFrame],
    *,
    window: int = 60,
    min_obs: int = 20,
    temperature: float = 1.0,
    equal_anchor: float = 0.2,
) -> pl.DataFrame:
    """Use only completed-close net returns through date t for decision t.

    Scores are clipped trailing mean / standard error. An equal-weight anchor
    prevents one short, noisy window from taking the entire allocation.
    """
    if not equities or window < 2 or min_obs < 2 or min_obs > window:
        raise ValueError("need sleeves and 2 <= min_obs <= window")
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be positive and finite")
    if not np.isfinite(equal_anchor) or not 0 <= equal_anchor <= 1:
        raise ValueError("equal_anchor must be in [0, 1]")
    names = sorted(equities)
    first = equities[names[0]].sort("event_time")
    dates = first["event_time"].to_list()
    if len(dates) < 2 or len(set(dates)) != len(dates):
        raise ValueError("equity calendar must have at least two unique dates")
    returns: dict[str, np.ndarray] = {}
    for name in names:
        frame = equities[name].sort("event_time")
        if frame["event_time"].to_list() != dates:
            raise ValueError(f"equity calendar mismatch: {name}")
        nav = frame["nav"].to_numpy().astype(float)
        if not np.all(np.isfinite(nav)) or np.any(nav <= 0):
            raise ValueError(f"invalid NAV path: {name}")
        ret = np.zeros(len(nav), dtype=float)
        ret[1:] = nav[1:] / nav[:-1] - 1.0
        returns[name] = ret
    allocations: dict[str, list[float]] = {name: [] for name in names}
    for i in range(len(dates)):
        scores = np.zeros(len(names), dtype=float)
        start = max(1, i - window + 1)
        if i - start + 1 >= min_obs:
            for j, name in enumerate(names):
                history = returns[name][start : i + 1]
                volatility = float(np.std(history, ddof=1))
                if volatility > 0:
                    scores[j] = np.clip(
                        np.mean(history) * np.sqrt(len(history)) / volatility, -3.0, 3.0
                    )
        exp_scores = np.exp(scores / temperature - np.max(scores / temperature))
        softmax = exp_scores / np.sum(exp_scores)
        alpha = equal_anchor / len(names) + (1.0 - equal_anchor) * softmax
        for j, name in enumerate(names):
            allocations[name].append(float(alpha[j]))
    return pl.DataFrame({"event_time": dates, **allocations})


def mix_targets(
    targets: Mapping[str, pl.DataFrame], allocations: pl.DataFrame
) -> pl.DataFrame:
    """Combine full target snapshots; explicit zeros flow through to exits."""
    names = sorted(targets)
    if not names or not set(names).issubset(allocations.columns):
        raise ValueError("allocation columns must match target sleeves")
    if allocations["event_time"].n_unique() != allocations.height:
        raise ValueError("duplicate allocation dates")
    weights = allocations.select(names).to_numpy().astype(float)
    if not np.all(np.isfinite(weights)) or np.any(weights < 0):
        raise ValueError("invalid allocations")
    if not np.allclose(np.sum(weights, axis=1), 1.0, atol=1e-10):
        raise ValueError("allocations must sum to one")
    keys = ["event_time", "security_id"]
    grid = targets[names[0]].select(keys).sort(keys)
    combined = grid
    for name in names:
        frame = targets[name].sort(keys)
        if not frame.select(keys).equals(grid):
            raise ValueError(f"target calendar mismatch: {name}")
        combined = combined.with_columns(frame["target_weight"].alias(name))
    combined = combined.join(allocations.select(["event_time", *names]), on="event_time")
    if combined.height != grid.height:
        raise ValueError("allocations missing target dates")
    expression = sum(pl.col(f"{name}_right") * pl.col(name) for name in names)
    return combined.select(*keys, expression.alias("target_weight"))


def banded_targets(targets: pl.DataFrame, band: float) -> pl.DataFrame:
    """Suppress small target revisions while always sending explicit exits.

    The reference is the last *requested* weight, since realized portfolio
    weights are known only inside the execution engine. This is a target
    hysteresis rule, not an assertion that live exposure stays within band.
    """
    if not np.isfinite(band) or band < 0:
        raise ValueError("band must be finite and nonnegative")
    keys = ["event_time", "security_id"]
    if not set(keys + ["target_weight"]).issubset(targets.columns):
        raise ValueError("targets missing required columns")
    if targets.select(keys).unique().height != targets.height:
        raise ValueError("duplicate targets for event_time/security_id")
    if targets.filter(~pl.col("target_weight").is_finite()).height:
        raise ValueError("non-finite target weight")
    if band == 0:
        return targets.sort(keys)
    previous: dict[str, float] = {}
    rows: list[dict] = []
    for row in targets.sort(keys).iter_rows(named=True):
        sid = str(row["security_id"])
        requested = float(row["target_weight"])
        old = previous.get(sid, 0.0)
        if (requested == 0.0 and old != 0.0) or (
            requested != 0.0 and abs(requested - old) > band
        ):
            rows.append(row)
            previous[sid] = requested
    if not rows:
        return targets.head(0)
    return pl.DataFrame(rows).select(targets.columns)
