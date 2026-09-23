"""Lagged characteristic long-only books and Faber timing. Research only.

Scores at date t-1 weight the close-to-close return at t. Top or bottom
fraction, equal weight, 10 bp on turnover. No sign flip of a losing book.
``blend_weight`` stays 0.
"""

from __future__ import annotations

import numpy as np
import polars as pl
from numpy.typing import NDArray

Array = NDArray[np.float64]


def _date_key(value: object) -> str:
    if hasattr(value, "isoformat"):
        return str(value.isoformat())[:10]
    text = str(value)
    return text[:10]


def lagged_characteristic_book(
    frame: pl.DataFrame,
    score_col: str,
    *,
    prefer_high: bool,
    k_frac: float = 0.2,
    one_way_cost: float = 0.001,
    ret_col: str = "ret",
) -> tuple[list[str], Array]:
    """Equal-weight the extreme lagged cross-section. First date is cash."""
    if not 0.0 < float(k_frac) <= 1.0:
        raise ValueError("k_frac must be in (0, 1]")
    needed = {"event_time", "security_id", score_col, ret_col}
    missing = needed - set(frame.columns)
    if missing:
        raise ValueError(f"frame missing {sorted(missing)}")
    data = frame.select(["event_time", "security_id", score_col, ret_col]).drop_nulls(
        subset=["event_time", "security_id", ret_col]
    )
    dates = [_date_key(d) for d in data["event_time"].unique().sort().to_list()]
    by_date: dict[str, pl.DataFrame] = {}
    keyed = data.with_columns(pl.col("event_time").map_elements(_date_key, return_dtype=pl.Utf8).alias("_d"))
    for key, grp in keyed.group_by("_d", maintain_order=True):
        stamp = key[0] if isinstance(key, tuple) else key
        by_date[str(stamp)] = grp
    pnl = np.zeros(len(dates), dtype=float)
    prev: dict[str, float] = {}
    for i, stamp in enumerate(dates):
        if i == 0 or stamp not in by_date:
            continue
        prior = dates[i - 1]
        scores = by_date.get(prior)
        today = by_date[stamp]
        if scores is None or scores.height < 5:
            prev = {}
            continue
        ranked = scores.drop_nulls(subset=[score_col]).sort(score_col, descending=prefer_high)
        k = max(1, int(np.floor(ranked.height * float(k_frac))))
        chosen = ranked.head(k)
        names = [str(s) for s in chosen["security_id"].to_list()]
        weight = 1.0 / float(len(names))
        current = dict.fromkeys(names, weight)
        ret_map = {
            str(s): float(r)
            for s, r in zip(today["security_id"].to_list(), today[ret_col].to_list(), strict=True)
            if r is not None and np.isfinite(float(r))
        }
        gross = 0.0
        for name, w in current.items():
            gross += w * ret_map.get(name, 0.0)
        keys = set(prev) | set(current)
        turn = float(sum(abs(current.get(name, 0.0) - prev.get(name, 0.0)) for name in keys))
        pnl[i] = gross - float(one_way_cost) * turn
        prev = current
    return dates, pnl


def price_above_sma_book(
    prices: Array,
    *,
    window: int = 200,
    one_way_cost: float = 0.001,
) -> Array:
    """Equal-weight assets whose lagged price is above their own SMA. Else cash."""
    px = np.asarray(prices, dtype=float)
    if px.ndim != 2:
        raise ValueError("prices must be (T, N)")
    rets = np.zeros_like(px)
    rets[1:] = px[1:] / np.maximum(px[:-1], 1e-12) - 1.0
    rets[~np.isfinite(rets)] = 0.0
    t_len, n_names = px.shape
    pnl = np.zeros(t_len, dtype=float)
    prev = np.zeros(n_names, dtype=float)
    for t in range(window, t_len):
        hist = px[t - window : t]
        sma = np.nanmean(hist, axis=0)
        last = px[t - 1]
        alive = np.isfinite(last) & np.isfinite(sma) & (last > sma)
        raw = np.zeros(n_names, dtype=float)
        if int(alive.sum()) > 0:
            raw[alive] = 1.0 / float(alive.sum())
        turn = float(np.sum(np.abs(raw - prev)))
        pnl[t] = float(np.sum(raw * rets[t])) - float(one_way_cost) * turn
        prev = raw
    return pnl


def binary_timing(signal: Array, asset: Array, *, one_way_cost: float = 0.001) -> Array:
    """Hold ``asset`` at t when ``signal[t-1]`` is true. Flat otherwise."""
    sig = np.asarray(signal, dtype=bool).reshape(-1)
    r = np.asarray(asset, dtype=float).reshape(-1)
    if sig.shape != r.shape:
        raise ValueError("signal and asset must align")
    pnl = np.zeros(r.size, dtype=float)
    prev = False
    for t in range(1, r.size):
        on = bool(sig[t - 1])
        turn = 0.0 if on == prev else 1.0
        prev = on
        pnl[t] = (float(r[t]) if on else 0.0) - float(one_way_cost) * turn
    return pnl
