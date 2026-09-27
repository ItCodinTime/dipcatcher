"""Small market fixtures for adversarial backtest properties."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl

from quant_fund.config.models import AppConfig

T0 = datetime(2024, 1, 2, tzinfo=UTC)


def research_config(**cost_overrides: float | bool) -> AppConfig:
    """Relaxed research book. No broker, no live order path."""
    cfg = AppConfig()
    cfg.data.root = Path("/tmp/dipcatcher-adversarial-properties")
    cfg.costs.frictionless = False
    cfg.costs.commission_bps = 1.0
    cfg.costs.half_spread_bps = 1.0
    cfg.costs.impact_y = 0.0
    cfg.costs.bps_per_turnover = 0.0
    cfg.costs.borrow_bps_per_year = 0.0
    cfg.costs.participation_limit = 1.0
    cfg.risk_gate.max_name = 1.0
    cfg.risk_gate.max_net = 1.0
    cfg.risk_gate.max_gross = 2.0
    cfg.risk_gate.max_order_notional = 1e15
    cfg.risk_gate.max_participation = 1.0
    cfg.risk_gate.max_predicted_vol = 100.0
    cfg.risk_gate.stale_price_bars = 10
    for name, value in cost_overrides.items():
        setattr(cfg.costs, name, value)
    return cfg


def daily_bars(
    prices: list[list[float]],
    *,
    adv: float = 1e12,
    vol: float = 0.02,
    source: str = "file",
) -> pl.DataFrame:
    """``prices[day][name]`` is both the open and the total-return close."""
    rows: list[dict[str, object]] = []
    for day, marks in enumerate(prices):
        when = T0 + timedelta(days=day)
        for i, px in enumerate(marks):
            rows.append(
                {
                    "security_id": f"S{i}",
                    "event_time": when,
                    "open": float(px),
                    "close": float(px),
                    "close_total_return": float(px),
                    "volume": 1_000_000.0,
                    "adv": float(adv),
                    "vol_20": float(vol),
                    "source": source,
                }
            )
    return pl.DataFrame(rows).with_columns(pl.col("event_time").cast(pl.Datetime("us", "UTC")))


def weights_frame(
    n_days: int,
    n_names: int,
    value: float | np.ndarray,
) -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    grid = np.asarray(value, dtype=float)
    if grid.ndim == 0:
        grid = np.full((n_days, n_names), float(grid))
    for day in range(n_days):
        when = T0 + timedelta(days=day)
        for i in range(n_names):
            rows.append(
                {
                    "event_time": when,
                    "security_id": f"S{i}",
                    "target_weight": float(grid[day, i]),
                }
            )
    return pl.DataFrame(rows).with_columns(pl.col("event_time").cast(pl.Datetime("us", "UTC")))


def positive_path(n_days: int, n_names: int, draws: list[list[float]]) -> list[list[float]]:
    """Compound positive prices from per-step simple returns in ``draws``."""
    out: list[list[float]] = []
    px = [100.0 * (1.0 + i) for i in range(n_names)]
    out.append(list(px))
    for day in range(1, n_days):
        nxt = []
        for i in range(n_names):
            px[i] = px[i] * (1.0 + draws[day - 1][i])
            nxt.append(px[i])
        out.append(nxt)
    return out
