"""Synthesize L2 snapshots from OHLCV bars for candle+LOB research.

This is a **labeled SYNTHETIC** microstructure generator: spreads and imbalance
are derived from bar OHLC/volume so the research path can fuse candles and books
without a vendor feed. Not evidence of live edge; ``source=synthetic``.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import polars as pl

from quant_fund.schemas.order_book import BookLevel, OrderBookSnapshot


def _tick_size(price: float) -> float:
    if price >= 100.0:
        return 0.05
    if price >= 10.0:
        return 0.01
    return 0.001


def ensure_side_notional_columns(panel: pl.DataFrame) -> pl.DataFrame:
    """Fail-closed: panel must carry SIDE_NOTIONAL fields + prices/depths.

    ``side_notional_finite_rate`` needs best_bid/ask and bid/ask_depth for
    eligibility. Missing columns are a plumbing bug on SYNTHETIC panels.
    """
    from quant_fund.microstructure.book_metrics import SIDE_NOTIONAL_FIELDS

    required = ("best_bid", "best_ask", "bid_depth", "ask_depth", *SIDE_NOTIONAL_FIELDS)
    missing = [c for c in required if c not in panel.columns]
    if missing:
        raise ValueError(f"L2 panel missing side-notional columns: {missing}")
    return panel


def ensure_queue_structure_columns(panel: pl.DataFrame) -> pl.DataFrame:
    """Fail-closed: panel must carry QUEUE_STRUCTURE fields + side depths.

    Queue-priority proxies need ``bid_depth`` / ``ask_depth`` for eligibility
    in ``queue_priority_finite_rate``. Missing columns are a plumbing bug.
    """
    from quant_fund.microstructure.book_metrics import QUEUE_STRUCTURE_FIELDS

    required = ("bid_depth", "ask_depth", *QUEUE_STRUCTURE_FIELDS)
    missing = [c for c in required if c not in panel.columns]
    if missing:
        raise ValueError(f"L2 panel missing queue-structure columns: {missing}")
    return panel


def ensure_side_structure_columns(panel: pl.DataFrame) -> pl.DataFrame:
    """Fail-closed: panel must carry SIDE_STRUCTURE fields + side depths.

    Concentration tops need ``bid_depth`` / ``ask_depth`` for eligibility
    in ``concentration_top_finite_rate``. Missing columns are a plumbing bug.
    """
    from quant_fund.microstructure.book_metrics import SIDE_STRUCTURE_FIELDS

    required = ("bid_depth", "ask_depth", *SIDE_STRUCTURE_FIELDS)
    missing = [c for c in required if c not in panel.columns]
    if missing:
        raise ValueError(f"L2 panel missing side-structure columns: {missing}")
    return panel


def ensure_depth_shape_columns(panel: pl.DataFrame) -> pl.DataFrame:
    """Fail-closed: panel must carry DEPTH_SHAPE fields + n_*_levels.

    Multi-level SYNTHETIC L2 (depth ≥ 2) must expose the same shape columns
    ``book_metrics_from_snapshot`` emits. Missing columns are a plumbing bug.
    """
    from quant_fund.microstructure.book_metrics import DEPTH_SHAPE_FIELDS

    required = ("n_bid_levels", "n_ask_levels", *DEPTH_SHAPE_FIELDS)
    missing = [c for c in required if c not in panel.columns]
    if missing:
        raise ValueError(f"L2 panel missing depth-shape columns: {missing}")
    return panel


def ensure_book_panel_shape_columns(panel: pl.DataFrame) -> pl.DataFrame:
    """Fail-closed: DEPTH_SHAPE + SIDE/QUEUE/NOTIONAL structure columns present.

    Convenience wrapper used by daily and session SYNTHETIC L2 builders.
    """
    return ensure_side_notional_columns(
        ensure_queue_structure_columns(
            ensure_side_structure_columns(ensure_depth_shape_columns(panel))
        )
    )


def synthesize_l2_from_bars(
    bars: pl.DataFrame,
    *,
    depth: int = 5,
    seed: int = 7,
    base_spread_bps: float = 4.0,
) -> pl.DataFrame:
    """Build one L2 snapshot per bar, aligned on ``security_id`` + ``event_time``.

    Returns a long frame of book metrics (not nested levels) plus a parallel list
    of ``OrderBookSnapshot`` via ``snapshots`` when needed by callers that use
    ``synthesize_snapshots_from_bars``.
    """
    snaps = synthesize_snapshots_from_bars(
        bars, depth=depth, seed=seed, base_spread_bps=base_spread_bps
    )
    from quant_fund.microstructure.book_metrics import book_metrics_from_snapshot

    rows: list[dict[str, object]] = []
    for snap in snaps:
        metrics = book_metrics_from_snapshot(snap)
        rows.append(
            {
                "security_id": snap.security_id,
                "event_time": snap.event_time,
                "available_time": snap.available_time,
                "source": snap.source,
                "revision_id": snap.revision_id,
                **metrics,
            }
        )
    return ensure_book_panel_shape_columns(pl.DataFrame(rows).sort(["security_id", "event_time"]))


def synthesize_snapshots_from_bars(
    bars: pl.DataFrame,
    *,
    depth: int = 5,
    seed: int = 7,
    base_spread_bps: float = 4.0,
) -> list[OrderBookSnapshot]:
    """Return validated ``OrderBookSnapshot`` objects aligned to each bar."""
    required = ("security_id", "event_time", "open", "high", "low", "close", "volume")
    missing = [c for c in required if c not in bars.columns]
    if missing:
        raise ValueError(f"bars missing required columns: {missing}")
    if bars.height == 0:
        return []
    if depth < 1:
        raise ValueError("depth must be >= 1")

    rng = np.random.default_rng(int(seed))
    frame = bars.sort(["security_id", "event_time"])
    # available_time defaults to event_time when absent
    has_avail = "available_time" in frame.columns
    has_symbol = "symbol" in frame.columns
    snaps: list[OrderBookSnapshot] = []

    for row in frame.iter_rows(named=True):
        close = float(row["close"])
        high = float(row["high"])
        low = float(row["low"])
        opn = float(row["open"])
        vol = max(float(row["volume"]), 1.0)
        if not all(math_isfinite(x) for x in (close, high, low, opn, vol)):
            continue
        if close <= 0.0:
            continue
        # Wider spread when bar range is large vs close (candle stress → book stress)
        range_frac = max(high - low, 0.0) / close
        body_frac = abs(close - opn) / close
        spread_bps = float(base_spread_bps) * (1.0 + 8.0 * range_frac + 4.0 * body_frac)
        spread_bps = float(np.clip(spread_bps, 1.0, 80.0))
        half = close * (spread_bps / 1e4) / 2.0
        tick = _tick_size(close)
        half = max(half, tick)
        best_bid = close - half
        best_ask = close + half
        # Imbalance tilted by candle direction (up bar → bid-heavy)
        direction = np.sign(close - opn)
        noise = float(rng.normal(0.0, 0.15))
        imb = float(np.clip(0.55 * direction + noise, -0.85, 0.85))
        # Allocate top size from volume
        top_total = max(vol * 0.02, 1.0)
        bid0 = top_total * (0.5 + 0.5 * imb)
        ask0 = top_total * (0.5 - 0.5 * imb)
        bids: list[BookLevel] = []
        asks: list[BookLevel] = []
        for level in range(depth):
            decay = 0.65**level
            bids.append(
                BookLevel(
                    price=round(best_bid - level * tick, 6),
                    size=round(max(bid0 * decay * (1.0 + 0.05 * rng.random()), 1e-6), 6),
                )
            )
            asks.append(
                BookLevel(
                    price=round(best_ask + level * tick, 6),
                    size=round(max(ask0 * decay * (1.0 + 0.05 * rng.random()), 1e-6), 6),
                )
            )
        event_time = row["event_time"]
        available = row["available_time"] if has_avail else event_time
        if not isinstance(event_time, datetime):
            raise TypeError("event_time must be datetime")
        if not isinstance(available, datetime):
            raise TypeError("available_time must be datetime")
        snaps.append(
            OrderBookSnapshot(
                security_id=str(row["security_id"]),
                symbol=str(row["symbol"]) if has_symbol else "",
                event_time=event_time,
                available_time=available,
                ingested_time=available,
                source="synthetic",
                revision_id="SYNTHETIC_LOB_v1",
                bids=bids,
                asks=asks,
                depth=depth,
            )
        )
    return snaps


def math_isfinite(value: float) -> bool:
    return value == value and value not in (float("inf"), float("-inf"))


def synthesize_session_l2(
    session: pl.DataFrame,
    *,
    depth: int = 5,
    seed: int = 7,
    base_spread_bps: float = 4.0,
) -> pl.DataFrame:
    """One L2 snapshot per session candle (multi-snapshot day path).

    ``session`` must carry ``parent_event_time`` + ``session_index`` + OHLCV.
    Returns a vendor-shaped book panel plus session keys. SYNTHETIC only.
    """
    required = (
        "security_id",
        "event_time",
        "parent_event_time",
        "session_index",
        "open",
        "high",
        "low",
        "close",
        "volume",
    )
    missing = [c for c in required if c not in session.columns]
    if missing:
        raise ValueError(f"session missing columns: {missing}")
    if session.height == 0:
        return pl.DataFrame()
    # Reuse bar synthesizer on session rows (event_time = session stamp).
    panel = synthesize_l2_from_bars(
        session,
        depth=depth,
        seed=seed,
        base_spread_bps=base_spread_bps,
    )
    keys = session.select(
        "security_id",
        "event_time",
        "parent_event_time",
        "session_index",
    )
    out = panel.join(keys, on=["security_id", "event_time"], how="inner").sort(
        ["security_id", "parent_event_time", "session_index"]
    )
    return ensure_book_panel_shape_columns(out)


def aggregate_session_book_to_daily(session_book: pl.DataFrame) -> pl.DataFrame:
    """Collapse multi-snapshot session L2 to one daily research row per parent bar.

    Uses last session snapshot for top-of-book state and path stats for OFI /
    imbalance / spread across the session. Join key is ``event_time`` (= parent).
    """
    required = (
        "security_id",
        "parent_event_time",
        "session_index",
        "imbalance_top",
        "spread_bps",
        "microprice_minus_mid_bps",
        "best_bid",
        "best_ask",
        "mid",
        "top_bid_size",
        "top_ask_size",
        "bid_depth",
        "ask_depth",
    )
    missing = [c for c in required if c not in session_book.columns]
    if missing:
        raise ValueError(f"session_book missing columns: {missing}")
    if session_book.height == 0:
        return pl.DataFrame()

    # OFI across session snapshots (needs consecutive tops).
    from quant_fund.northset.estimators import order_flow_imbalance

    # order_flow_imbalance groups by security_id + event_time order; for session
    # path we want per (security_id, parent) chronology via session_index.
    ordered = session_book.sort(["security_id", "parent_event_time", "session_index"])
    # Temporarily use a monotonic event_time proxy for OFI within each parent day
    # by sorting on session_index already; OFI shifts over security_id only would
    # leak across days — so compute OFI within parent groups manually.
    frame = ordered.with_columns(
        pl.col("best_bid").shift(1).over(["security_id", "parent_event_time"]).alias("_pb"),
        pl.col("best_ask").shift(1).over(["security_id", "parent_event_time"]).alias("_pa"),
        pl.col("top_bid_size").shift(1).over(["security_id", "parent_event_time"]).alias("_ptb"),
        pl.col("top_ask_size").shift(1).over(["security_id", "parent_event_time"]).alias("_pta"),
    )
    ofi = (
        pl.when(pl.col("best_bid") >= pl.col("_pb")).then(pl.col("top_bid_size")).otherwise(0.0)
        - pl.when(pl.col("best_bid") <= pl.col("_pb")).then(pl.col("_ptb")).otherwise(0.0)
        - pl.when(pl.col("best_ask") <= pl.col("_pa")).then(pl.col("top_ask_size")).otherwise(0.0)
        + pl.when(pl.col("best_ask") >= pl.col("_pa")).then(pl.col("_pta")).otherwise(0.0)
    )
    frame = frame.with_columns(ofi.alias("session_step_ofi"))
    del order_flow_imbalance  # imported only to document kinship

    last = (
        frame.sort(["security_id", "parent_event_time", "session_index"])
        .group_by(["security_id", "parent_event_time"], maintain_order=True)
        .agg(
            pl.col("session_index").max().alias("_max_idx"),
        )
    )
    # last snapshot metrics
    last_snap = (
        frame.join(last, on=["security_id", "parent_event_time"], how="inner")
        .filter(pl.col("session_index") == pl.col("_max_idx"))
        .select(
            "security_id",
            pl.col("parent_event_time").alias("event_time"),
            pl.col("mid").alias("session_close_mid"),
            pl.col("spread_bps").alias("session_close_spread_bps"),
            pl.col("imbalance_top").alias("session_close_imbalance"),
            pl.col("microprice_minus_mid_bps").alias("session_close_micro_bps"),
            pl.col("bid_depth").alias("session_close_bid_depth"),
            pl.col("ask_depth").alias("session_close_ask_depth"),
            pl.col("source").alias("session_book_source")
            if "source" in frame.columns
            else pl.lit("synthetic").alias("session_book_source"),
        )
    )
    path = (
        frame.group_by(["security_id", "parent_event_time"])
        .agg(
            pl.col("session_step_ofi").sum().alias("session_ofi_sum"),
            pl.col("session_step_ofi").abs().sum().alias("session_ofi_abs_sum"),
            pl.col("imbalance_top").mean().alias("session_imbalance_mean"),
            pl.col("imbalance_top").std().alias("session_imbalance_std"),
            pl.col("spread_bps").mean().alias("session_spread_bps_mean"),
            pl.col("session_index").count().alias("n_session_book_snaps"),
        )
        .rename({"parent_event_time": "event_time"})
    )
    # session book VPIN proxy: |ofi_sum| / ofi_abs_sum
    path = path.with_columns(
        pl.when(pl.col("session_ofi_abs_sum") > 0.0)
        .then(pl.col("session_ofi_sum").abs() / pl.col("session_ofi_abs_sum"))
        .otherwise(None)
        .alias("session_book_vpin")
    )
    return last_snap.join(path, on=["security_id", "event_time"], how="inner").sort(
        ["security_id", "event_time"]
    )
