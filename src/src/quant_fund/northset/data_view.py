"""Canonical corporate-action-safe market view for Northset."""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from quant_fund.northset.identities import ohlc_identity_rate

_ADJUSTED = (
    "open_split_adjusted",
    "high_split_adjusted",
    "low_split_adjusted",
    "close_split_adjusted",
)


@dataclass(frozen=True)
class NorthsetMarketView:
    frame: pl.DataFrame
    price_basis: str
    return_basis: str


def canonical_northset_bars(
    bars: pl.DataFrame,
    *,
    require_adjusted: bool = True,
) -> NorthsetMarketView:
    """Return adjusted OHLCV while retaining PIT and identity columns.

    Split-adjusted prices drive candle geometry and sweep detection. Total-return
    close drives forward exits when available. Raw input is accepted only via an
    explicit opt-out, intended for narrow fixtures without corporate actions.
    """
    required = ("security_id", "event_time", "open", "high", "low", "close", "volume")
    missing = [c for c in required if c not in bars.columns]
    if missing:
        raise ValueError(f"bars missing required columns: {missing}")
    present = [c in bars.columns for c in _ADJUSTED]
    if any(present) and not all(present):
        missing_adjusted = [c for c in _ADJUSTED if c not in bars.columns]
        raise ValueError(f"partial split-adjusted OHLC columns: {missing_adjusted}")
    if all(present):
        volume = (
            pl.col("volume_split_adjusted")
            if "volume_split_adjusted" in bars.columns
            else (
                pl.col("volume") * pl.col("split_factor")
                if "split_factor" in bars.columns
                else pl.col("volume")
            )
        )
        return_close = (
            pl.col("close_total_return")
            if "close_total_return" in bars.columns
            else pl.col("close_split_adjusted")
        )
        return_open = (
            pl.col("open_split_adjusted")
            * pl.col("close_total_return")
            / pl.col("close_split_adjusted")
            if "close_total_return" in bars.columns
            else pl.col("open_split_adjusted")
        )
        frame = bars.with_columns(
            pl.col("open_split_adjusted").alias("open"),
            pl.col("high_split_adjusted").alias("high"),
            pl.col("low_split_adjusted").alias("low"),
            pl.col("close_split_adjusted").alias("close"),
            volume.alias("volume"),
            return_open.alias("return_open"),
            return_close.alias("return_close"),
        )
        price_basis = "split_adjusted"
        return_basis = "total_return" if "close_total_return" in bars.columns else "split_adjusted"
    else:
        if require_adjusted:
            raise ValueError(
                "Northset requires split-adjusted OHLC columns; "
                "set northset.require_adjusted_ohlc=false only for controlled fixtures"
            )
        frame = bars.with_columns(
            pl.col("open").alias("return_open"),
            pl.col("close").alias("return_close"),
        )
        price_basis = "raw_fixture_opt_out"
        return_basis = "raw_fixture_opt_out"
    if ohlc_identity_rate(frame) < 1.0:
        raise ValueError("canonical Northset OHLC identities fail")
    return NorthsetMarketView(
        frame=frame,
        price_basis=price_basis,
        return_basis=return_basis,
    )
