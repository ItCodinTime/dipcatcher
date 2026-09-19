"""Northset identity checks: OHLC candles and L2 books.

Research-only plumbing. A failing identity is a data-contract bug, not a
trading signal. SYNTHETIC books/candles are labeled and not live evidence.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, cast

import numpy as np
import polars as pl

_EPS = 1e-9


def ohlc_identity_frame(bars: pl.DataFrame) -> pl.DataFrame:
    """Boolean identity columns for classic candle inequalities."""
    required = ("open", "high", "low", "close")
    missing = [c for c in required if c not in bars.columns]
    if missing:
        raise ValueError(f"bars missing OHLC columns: {missing}")
    hi = pl.col("high")
    lo = pl.col("low")
    opn = pl.col("open")
    cl = pl.col("close")
    return bars.with_columns(
        (
            hi.is_finite()
            & lo.is_finite()
            & opn.is_finite()
            & cl.is_finite()
            & (opn > 0.0)
            & (cl > 0.0)
            & (hi + _EPS >= pl.max_horizontal(opn, cl, lo))
            & (lo - _EPS <= pl.min_horizontal(opn, cl, hi))
            & (hi + _EPS >= lo)
        ).alias("ohlc_ok")
    )


def ohlc_identity_rate(bars: pl.DataFrame) -> float:
    """Fraction of rows that satisfy OHLC identities. Empty → NaN."""
    if bars.height == 0:
        return float("nan")
    flagged = ohlc_identity_frame(bars)
    return float(cast(float, flagged["ohlc_ok"].mean()))


def book_identity_frame(book: pl.DataFrame) -> pl.DataFrame:
    """Boolean identity columns for a top-of-book metrics frame."""
    required = ("best_bid", "best_ask")
    missing = [c for c in required if c not in book.columns]
    if missing:
        raise ValueError(f"book missing columns: {missing}")
    bid = pl.col("best_bid")
    ask = pl.col("best_ask")
    uncrossed = bid.is_finite() & ask.is_finite() & (bid < ask)
    if "spread" in book.columns:
        uncrossed = uncrossed & pl.col("spread").is_finite() & (pl.col("spread") > 0.0)
    return book.with_columns(uncrossed.alias("book_uncrossed"))


def book_uncrossed_rate(book: pl.DataFrame) -> float:
    """Fraction of snapshots with best bid strictly below best ask. Empty → NaN."""
    if book.height == 0:
        return float("nan")
    flagged = book_identity_frame(book)
    return float(cast(float, flagged["book_uncrossed"].mean()))


def _session_ohlc(
    open_: float,
    high: float,
    low: float,
    close: float,
    n: int,
    high_first: bool,
) -> list[tuple[float, float, float, float]]:
    """n session candles whose envelope reconstructs the daily OHLC bar."""
    if n < 2:
        raise ValueError("n_session_candles must be >= 2")
    waypoints = [open_, high, low, close] if high_first else [open_, low, high, close]
    xs = np.linspace(0.0, 3.0, n + 1)
    knots = np.interp(xs, [0.0, 1.0, 2.0, 3.0], waypoints)
    knots[0] = open_
    knots[-1] = close
    candles: list[list[float]] = []
    for i in range(n):
        o = float(knots[i])
        c = float(knots[i + 1])
        candles.append([o, max(o, c), min(o, c), c])
    hi_i = int(np.argmax([row[1] for row in candles]))
    lo_i = int(np.argmin([row[2] for row in candles]))
    candles[hi_i][1] = max(candles[hi_i][1], high)
    candles[lo_i][2] = min(candles[lo_i][2], low)
    out: list[tuple[float, float, float, float]] = []
    for o, hi, lo, c in candles:
        hi = max(hi, o, c)
        lo = min(lo, o, c)
        out.append((float(o), float(hi), float(lo), float(c)))
    return out


def session_candles_from_daily(
    bars: pl.DataFrame,
    *,
    n_candles: int = 8,
    seed: int = 7,
) -> pl.DataFrame:
    """Build PIT session candles that reconstruct each daily OHLC envelope.

    First open equals the daily open; last close equals the daily close;
    the session high/low envelope matches the daily high/low. SYNTHETIC path
    only — not a vendor tape.
    """
    required = ("security_id", "event_time", "open", "high", "low", "close")
    missing = [c for c in required if c not in bars.columns]
    if missing:
        raise ValueError(f"bars missing required columns: {missing}")
    if bars.height == 0:
        return pl.DataFrame(
            schema={
                "security_id": pl.String,
                "event_time": pl.Datetime(time_zone="UTC"),
                "available_time": pl.Datetime(time_zone="UTC"),
                "parent_event_time": pl.Datetime(time_zone="UTC"),
                "session_index": pl.Int64,
                "open": pl.Float64,
                "high": pl.Float64,
                "low": pl.Float64,
                "close": pl.Float64,
                "volume": pl.Float64,
                "source": pl.String,
                "revision_id": pl.String,
            }
        )
    if n_candles < 2:
        raise ValueError("n_candles must be >= 2")
    rng = np.random.default_rng(int(seed))
    has_symbol = "symbol" in bars.columns
    has_available = "available_time" in bars.columns
    has_volume = "volume" in bars.columns
    rows: list[dict[str, Any]] = []
    frame = bars.sort(["security_id", "event_time"])
    for row in frame.iter_rows(named=True):
        opn = float(row["open"])
        high = float(row["high"])
        low = float(row["low"])
        close = float(row["close"])
        if not all(np.isfinite(x) for x in (opn, high, low, close)):
            continue
        if close <= 0.0:
            continue
        high_first = bool(rng.random() < 0.5)
        candles = _session_ohlc(opn, high, low, close, n_candles, high_first)
        parent = row["event_time"]
        if not isinstance(parent, datetime):
            raise TypeError("event_time must be datetime")
        parent_available = row["available_time"] if has_available else parent
        if not isinstance(parent_available, datetime):
            raise TypeError("available_time must be datetime")
        # Pseudo-session timestamps end at the parent bar timestamp. Every
        # reconstructed row remains unavailable until the completed parent bar
        # is available; these timestamps never imply an observable intraday tape.
        start = parent - timedelta(minutes=int(6.5 * 60.0))
        vol = float(row["volume"]) if has_volume else 0.0
        slice_vol = vol / float(n_candles) if np.isfinite(vol) else 0.0
        symbol = str(row["symbol"]) if has_symbol else ""
        for i, (o, hi, lo, c) in enumerate(candles):
            ts = start + timedelta(minutes=int((i + 1) * (6.5 * 60.0) / n_candles))
            rows.append(
                {
                    "security_id": str(row["security_id"]),
                    "symbol": symbol,
                    "event_time": ts,
                    "available_time": parent_available,
                    "parent_event_time": parent,
                    "session_index": int(i),
                    "open": o,
                    "high": hi,
                    "low": lo,
                    "close": c,
                    "volume": slice_vol,
                    "source": "synthetic_reconstruction",
                    "revision_id": "NORTHSET_SESSION_v2_PARENT_CLOSE_AVAILABLE",
                }
            )
    return pl.DataFrame(rows).sort(["security_id", "event_time"])


def session_reconstructs_daily_rate(
    daily: pl.DataFrame,
    session: pl.DataFrame,
    *,
    atol: float = 1e-8,
) -> float:
    """Fraction of daily bars whose session envelope matches OHLC."""
    if daily.height == 0 or session.height == 0:
        return float("nan")
    env = session.group_by(["security_id", "parent_event_time"]).agg(
        pl.col("open").sort_by("session_index").first().alias("sess_open"),
        pl.col("close").sort_by("session_index").last().alias("sess_close"),
        pl.col("high").max().alias("sess_high"),
        pl.col("low").min().alias("sess_low"),
    )
    joined = daily.select(
        "security_id",
        pl.col("event_time").alias("parent_event_time"),
        pl.col("open").alias("day_open"),
        pl.col("high").alias("day_high"),
        pl.col("low").alias("day_low"),
        pl.col("close").alias("day_close"),
    ).join(env, on=["security_id", "parent_event_time"], how="inner")
    if joined.height == 0:
        return float("nan")
    ok = (
        ((joined["sess_open"] - joined["day_open"]).abs() <= atol)
        & ((joined["sess_close"] - joined["day_close"]).abs() <= atol)
        & ((joined["sess_high"] - joined["day_high"]).abs() <= atol)
        & ((joined["sess_low"] - joined["day_low"]).abs() <= atol)
    )
    return float(cast(float, ok.mean()))


def session_volume_conservation_rate(
    daily: pl.DataFrame,
    session: pl.DataFrame,
    *,
    rtol: float = 1e-8,
) -> float:
    """Fraction of days whose session volumes sum to the daily volume."""
    if daily.height == 0 or session.height == 0 or "volume" not in daily.columns:
        return float("nan")
    if "volume" not in session.columns:
        return float("nan")
    summed = session.group_by(["security_id", "parent_event_time"]).agg(
        pl.col("volume").sum().alias("sess_volume")
    )
    joined = daily.select(
        "security_id",
        pl.col("event_time").alias("parent_event_time"),
        pl.col("volume").alias("day_volume"),
    ).join(summed, on=["security_id", "parent_event_time"], how="inner")
    if joined.height == 0:
        return float("nan")
    denom = joined["day_volume"].abs().clip(lower_bound=1e-12)
    ok = (joined["sess_volume"] - joined["day_volume"]).abs() <= rtol * denom
    return float(cast(float, ok.mean()))


def session_chain_rate(
    session: pl.DataFrame,
    *,
    atol: float = 1e-8,
) -> float:
    """Fraction of consecutive session candles whose close equals the next open."""
    required = ("security_id", "parent_event_time", "session_index", "open", "close")
    missing = [c for c in required if c not in session.columns]
    if missing or session.height == 0:
        return float("nan")
    frame = session.sort(["security_id", "parent_event_time", "session_index"]).with_columns(
        pl.col("close").shift(1).over(["security_id", "parent_event_time"]).alias("prev_close")
    )
    pairs = frame.filter(pl.col("session_index") > 0)
    if pairs.height == 0:
        return float("nan")
    ok = (pairs["open"] - pairs["prev_close"]).abs() <= atol
    return float(cast(float, ok.mean()))


def gap_finite_rate(bars: pl.DataFrame) -> float:
    """Fraction of bars with a finite overnight gap (open vs previous close)."""
    if bars.height == 0 or "open" not in bars.columns or "close" not in bars.columns:
        return float("nan")
    frame = bars.sort(["security_id", "event_time"]).with_columns(
        (pl.col("open") / pl.col("close").shift(1).over("security_id") - 1.0).alias("gap")
    )
    eligible = frame.filter(pl.col("gap").is_not_null())
    if eligible.height == 0:
        return float("nan")
    return float(cast(float, eligible["gap"].is_finite().mean()))


def validate_session_book_counts(
    session_book: pl.DataFrame,
    *,
    n_session_candles: int,
) -> pl.DataFrame:
    """Fail-closed: every (security_id, parent_event_time) has exactly n snaps.

    Returns the input frame if valid. Raises ``ValueError`` on empty frames,
    missing keys, or any parent day with the wrong session snap count.
    Research-integrity helper for multi-snapshot L2 (ADR-021).
    """
    if n_session_candles < 2:
        raise ValueError("n_session_candles must be >= 2")
    required = ("security_id", "parent_event_time", "session_index")
    missing = [c for c in required if c not in session_book.columns]
    if missing:
        raise ValueError(f"session book missing columns: {missing}")
    if session_book.height == 0:
        raise ValueError("session book is empty")
    counts = (
        session_book.group_by(["security_id", "parent_event_time"])
        .agg(pl.len().alias("n_snaps"), pl.col("session_index").n_unique().alias("n_idx"))
        .sort(["security_id", "parent_event_time"])
    )
    bad = counts.filter(
        (pl.col("n_snaps") != int(n_session_candles)) | (pl.col("n_idx") != int(n_session_candles))
    )
    if bad.height:
        sample = bad.head(3).to_dicts()
        raise ValueError(
            f"session book count mismatch (want {n_session_candles} snaps/indices per parent): "
            f"{sample}"
        )
    return session_book
