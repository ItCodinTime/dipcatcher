"""Labeled SYNTHETIC OHLCV panel generator for parity fixtures and the shootout.

Deterministic per ``seed``: a shared market factor plus per-name AR(1)
idiosyncratic returns, one calendar that every name shares (no missing bars),
and an overnight gap so ``open[t] != close[t-1]`` — the property that makes
signal-time vs execution-time sizing conventions visible.

All rows carry ``source == "synthetic"``; the panel is not market evidence.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl

PANEL_SOURCE = "synthetic"


def make_synthetic_panel(
    *,
    n_symbols: int = 50,
    n_days: int = 1260,
    seed: int = 7,
    start: datetime = datetime(2020, 1, 1, tzinfo=UTC),
    bar_days: int = 1,
    market_vol: float = 0.012,
    idio_vol: float = 0.018,
    beta_spread: float = 0.4,
    ar1: float = 0.10,
    overnight_gap_sigma: float = 0.003,
    base_volume: float = 2_000_000.0,
    start_price: float = 100.0,
) -> pl.DataFrame:
    """Return a ``quant_fund`` bars panel with an injected trend regime.

    Columns: security_id, event_time, open, high, low, close,
    close_total_return, volume, adv, vol_20, source.

    ``close_total_return == close`` (no corporate actions — a stated
    simplification so every engine sees identical marks).
    """
    if n_symbols < 1 or n_days < 2:
        raise ValueError("n_symbols >= 1 and n_days >= 2 required")
    for name, value in (
        ("market_vol", market_vol),
        ("idio_vol", idio_vol),
        ("overnight_gap_sigma", overnight_gap_sigma),
    ):
        if not np.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and non-negative")
    rng = np.random.default_rng(seed)
    market = rng.normal(0.0002, market_vol, n_days)
    # Slow trend regimes give momentum/crossover strategies real signals.
    trend = np.zeros(n_days)
    trend[n_days // 3 : 2 * n_days // 3] = 0.0009
    trend[2 * n_days // 3 :] = -0.0005
    betas = 1.0 + rng.uniform(-beta_spread, beta_spread, n_symbols)
    idio_ar = np.zeros(n_symbols)
    rows: list[dict] = []
    for k in range(n_symbols):
        eps = rng.normal(0.0, idio_vol, n_days)
        px = start_price * (1.0 + 0.05 * (k % 7))
        prev_close = px
        for i in range(n_days):
            idio_ar[k] = ar1 * idio_ar[k] + eps[i]
            gap = rng.normal(0.0, overnight_gap_sigma)
            open_ = prev_close * (1.0 + gap)
            close = open_ * (1.0 + betas[k] * market[i] + trend[i] + idio_ar[k])
            close = max(close, 0.5)
            high = max(open_, close) * (1.0 + abs(rng.normal(0.0, 0.002)))
            low = min(open_, close) * (1.0 - abs(rng.normal(0.0, 0.002)))
            vol = base_volume * (1.0 + 0.1 * (k % 5)) * (1.0 + abs(rng.normal(0.0, 0.2)))
            rows.append(
                {
                    "security_id": f"S{k:03d}",
                    "event_time": start + timedelta(days=i * bar_days),
                    "open": float(open_),
                    "high": float(high),
                    "low": float(low),
                    "close": float(close),
                    "close_total_return": float(close),
                    "volume": float(vol),
                    "adv": float(close * vol),
                    "vol_20": float(idio_vol * np.sqrt(1.0 + betas[k] ** 2)),
                    "source": PANEL_SOURCE,
                }
            )
            prev_close = close
    return pl.DataFrame(rows).with_columns(
        pl.col("event_time").cast(pl.Datetime("us", "UTC"))
    )


def wide_ohlc(panel: pl.DataFrame) -> dict[str, "object"]:
    """Pivot the panel to wide numpy frames for engine adapters.

    Returns dict with ``dates`` (np.datetime64[us] array), ``symbols``,
    and ``open``/``high``/``low``/``close`` as (n_days, n_symbols) float arrays.
    """
    import pandas as pd

    piv = {
        col: panel.select("event_time", "security_id", col)
        .to_pandas()
        .pivot(index="event_time", columns="security_id", values=col)
        .sort_index()
        for col in ("open", "high", "low", "close")
    }
    first = piv["close"]
    return {
        "dates": first.index.to_numpy(),
        "index": pd.DatetimeIndex(first.index),
        "symbols": list(first.columns),
        "open": piv["open"],
        "high": piv["high"],
        "low": piv["low"],
        "close": piv["close"],
    }
