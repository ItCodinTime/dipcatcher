"""Canonical equity panel for the paper-replication lab.

The canonical frame is long-format with one row per (security, bar) and these
columns:

- ``security_id``, ``symbol`` — identifiers (strings).
- ``event_time`` — bar timestamp (UTC); sorted per security.
- ``open``, ``close`` — split-adjusted prices.
- ``ret_cc`` — close-to-close total return (dividend-inclusive when the source
  provides a total-return series).
- ``ret_on`` — overnight return ``open_t / close_{t-1} - 1``.
- ``ret_id`` — intraday return ``close_t / open_t - 1``.

``ret_cc`` on day *t* is information available only after the day-*t* close, so
a signal built from data ``<= t`` is evaluated against returns starting at
``t + 1`` — the no-lookahead convention used throughout the package.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

PANEL_SCHEMA_VERSION = "replication_panel.v1"

PANEL_COLUMNS: tuple[str, ...] = (
    "security_id",
    "symbol",
    "event_time",
    "open",
    "close",
    "ret_cc",
    "ret_on",
    "ret_id",
)

# Symbols that carry a pre-aggregated market index rather than a tradable asset
# (the lab's synthetic panel ships ``MKT``). They are excluded from
# cross-sectional strategies; the market leg is rebuilt by ``market_returns``.
MARKET_SYMBOLS = frozenset({"MKT"})


def _require_columns(frame: pl.DataFrame, names: tuple[str, ...], where: str) -> None:
    missing = [name for name in names if name not in frame.columns]
    if missing:
        raise ValueError(f"{where}: missing columns {missing}")


def make_panel(ohlc: pl.DataFrame) -> pl.DataFrame:
    """Build the canonical panel from an OHLC(+total-return) long frame.

    Required input columns: ``security_id``, ``symbol``, ``event_time``,
    ``open``, ``close``; optional ``close_total_return`` (falls back to
    ``close``). Rows are sorted per security before differencing.
    """
    _require_columns(
        ohlc,
        ("security_id", "symbol", "event_time", "open", "close"),
        "make_panel",
    )
    frame = ohlc.clone()
    if "close_total_return" not in frame.columns:
        frame = frame.with_columns(pl.col("close").alias("close_total_return"))
    frame = frame.sort(["security_id", "event_time"])
    frame = frame.with_columns(
        (pl.col("close_total_return") / pl.col("close_total_return").shift(1) - 1.0)
        .over("security_id")
        .alias("ret_cc"),
        (pl.col("open") / pl.col("close").shift(1) - 1.0).over("security_id").alias("ret_on"),
        (pl.col("close") / pl.col("open") - 1.0).alias("ret_id"),
    ).select(PANEL_COLUMNS)
    return validate_panel(frame)


def validate_panel(frame: pl.DataFrame) -> pl.DataFrame:
    """Fail-closed structural checks on the canonical panel."""
    _require_columns(frame, PANEL_COLUMNS, "panel")
    if frame.height == 0:
        raise ValueError("panel is empty")
    for col in ("security_id", "symbol"):
        if frame[col].dtype != pl.String:
            frame = frame.with_columns(pl.col(col).cast(pl.String))
    dup = frame.select("security_id", "event_time").is_duplicated()
    if bool(dup.any()):
        raise ValueError("panel has duplicate (security_id, event_time) rows")
    for col in ("open", "close", "ret_cc", "ret_on", "ret_id"):
        if frame[col].dtype != pl.Float64:
            frame = frame.with_columns(pl.col(col).cast(pl.Float64))
    if bool((frame.select(pl.col("open", "close") <= 0.0).to_numpy()).any()):
        raise ValueError("panel has non-positive prices")
    return frame.sort(["security_id", "event_time"])


def load_silver_bars(path: Path | str, *, drop_market: bool = True) -> pl.DataFrame:
    """Load the lab silver bars parquet into the canonical panel.

    The silver layer carries ``*_split_adjusted`` prices plus a
    ``close_total_return`` column; raw ``open``/``close`` are ignored so
    dividends land in ``ret_cc`` exactly once.
    """
    path = Path(path)
    frame = pl.read_parquet(path)
    _require_columns(
        frame,
        (
            "security_id",
            "symbol",
            "event_time",
            "open_split_adjusted",
            "close_split_adjusted",
        ),
        f"silver bars at {path}",
    )
    if drop_market:
        frame = frame.filter(~pl.col("symbol").is_in(sorted(MARKET_SYMBOLS)))
    frame = frame.select(
        pl.col("security_id"),
        pl.col("symbol"),
        pl.col("event_time"),
        pl.col("open_split_adjusted").alias("open"),
        pl.col("close_split_adjusted").alias("close"),
        pl.col("close_total_return").alias("close_total_return")
        if "close_total_return" in frame.columns
        else pl.col("close_split_adjusted").alias("close_total_return"),
    )
    return make_panel(frame)


def market_returns(panel: pl.DataFrame) -> pl.DataFrame:
    """Equal-weight universe return per date — the replication 'market'.

    McConnell & Xu (2008) and Frazzini & Pedersen (2014) use a value-weighted
    index; with no market caps in the panel the equal-weight mean of the
    universe is the documented stand-in (see docs/PAPER_REPLICATION.md).
    """
    _require_columns(panel, ("event_time", "ret_cc", "ret_on", "ret_id"), "market_returns")
    return (
        panel.group_by("event_time")
        .agg(
            pl.col("ret_cc").mean().alias("mkt_ret_cc"),
            pl.col("ret_on").mean().alias("mkt_ret_on"),
            pl.col("ret_id").mean().alias("mkt_ret_id"),
            pl.len().alias("n_assets"),
        )
        .sort("event_time")
    )
