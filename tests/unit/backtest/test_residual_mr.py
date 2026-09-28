"""Kakushadze residual mean-reversion sleeve: directionality + causality."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl
import pytest

from quant_fund.backtest.sleeves import residual_mr_weights

N_BARS = 260
NAMES = ["AAA", "BBB", "CCC", "DDD", "EEE"]


def _bars(seed: int = 0, shock_name: str | None = None, shock_bar: int = 200, shock: float = 0.0):
    """Factor + idiosyncratic book; optional one-bar shock to ``shock_name``."""
    rng = np.random.default_rng(seed)
    start = datetime(2025, 1, 1, tzinfo=UTC)
    rows = []
    factor = rng.normal(0.0, 0.004, size=N_BARS)
    for i, sid in enumerate(NAMES):
        beta = 0.5 + 0.2 * i
        idio = rng.normal(0.0, 0.002, size=N_BARS)
        r = beta * factor + idio
        if shock_name is not None and sid == shock_name:
            r[shock_bar] += shock
        px = 100.0 * np.exp(np.cumsum(r))
        for t in range(N_BARS):
            dt = start + timedelta(hours=t)
            c = px[t]
            o = px[t - 1] if t else c / np.exp(r[0])
            rows.append(
                {
                    "event_time": dt,
                    "security_id": sid,
                    "open": o,
                    "high": max(o, c) * 1.001,
                    "low": min(o, c) * 0.999,
                    "close": c,
                    "volume": 1e5,
                }
            )
    return pl.DataFrame(rows)


def test_emits_capped_dollar_neutral_weights() -> None:
    w = residual_mr_weights(_bars())
    assert set(w.columns) == {"event_time", "security_id", "target_weight"}
    assert w["target_weight"].abs().max() <= 0.05 + 1e-12
    gross = w.group_by("event_time").agg(pl.col("target_weight").abs().sum().alias("g"))
    assert gross["g"].max() <= 1.0 + 1e-9
    # Demeaned: median ≈ 0 wherever enough names emitted.
    med = w.group_by("event_time").agg(pl.col("target_weight").median().alias("m"))
    assert med["m"].abs().max() < 0.05


def test_residual_shock_flips_weight_sign() -> None:
    """A one-bar positive residual shock must produce a short weight next bars."""
    bars = _bars(shock_name="CCC", shock_bar=200, shock=+0.05)
    w = residual_mr_weights(bars)
    shock_t = datetime(2025, 1, 1, tzinfo=UTC) + timedelta(hours=200)
    after = w.filter((pl.col("security_id") == "CCC") & (pl.col("event_time") > shock_t)).sort(
        "event_time"
    )
    assert after.height > 0
    assert after["target_weight"][0] < 0.0
    # Symmetric: a negative shock goes long.
    bars_dn = _bars(shock_name="CCC", shock_bar=200, shock=-0.05)
    w_dn = residual_mr_weights(bars_dn)
    after_dn = w_dn.filter(
        (pl.col("security_id") == "CCC") & (pl.col("event_time") > shock_t)
    ).sort("event_time")
    assert after_dn["target_weight"][0] > 0.0


def test_causality_tail_perturbation() -> None:
    """Rewriting the tail of the book leaves every earlier weight identical."""
    bars = _bars()
    cut = datetime(2025, 1, 1, tzinfo=UTC) + timedelta(hours=180)
    mutated = bars.with_columns(
        pl.when(pl.col("event_time") > cut)
        .then(pl.col("close") * 1.5)
        .otherwise(pl.col("close"))
        .alias("close")
    )
    w_base = residual_mr_weights(bars).filter(pl.col("event_time") <= cut)
    w_mut = residual_mr_weights(mutated).filter(pl.col("event_time") <= cut)
    assert w_base.equals(w_mut)


def test_determinism() -> None:
    assert residual_mr_weights(_bars(seed=3)).equals(residual_mr_weights(_bars(seed=3)))


def test_validation() -> None:
    bars = _bars()
    with pytest.raises(ValueError, match="factor_window"):
        residual_mr_weights(bars, factor_window=1)
    with pytest.raises(ValueError, match="z_clip"):
        residual_mr_weights(bars, z_clip=0.0)
    with pytest.raises(ValueError, match="missing columns"):
        residual_mr_weights(bars.drop("close"))
