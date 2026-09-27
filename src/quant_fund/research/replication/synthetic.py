"""Labeled SYNTHETIC equity panels for replication tests and fixtures.

These generators produce the canonical panel schema (``panel.PANEL_COLUMNS``)
with planted, parameterized anomalies so the replication harness can be checked
against known ground truth. They are correctness fixtures — never market
evidence. Every receipt written on synthetic input is labeled
``data_label="SYNTHETIC"``.

Planted effects (all default 0 → the null world where every strategy should
report "inconclusive", exercising the no-false-positive path):

- ``ar1_rho``: per-asset idiosyncratic daily AR(1) loading on total returns.
  Positive → time-series continuation (MOP2012); negative → short-term
  reversal (Jegadeesh 1990 / Lehmann 1990).
- ``lowvol_premium``: per-asset daily drift ``-lowvol_premium * (sigma_i -
  min_sigma)`` — high-volatility assets earn less (Baker-Bradley-Wurgler 2011
  direction; beta is mechanically tied to vol here so BAB inherits it).
- ``overnight_drift``: for the first half of the universe the whole daily
  drift is realized overnight instead of intraday (Lou-Polk-Skouras 2019
  tug-of-war: persistent overnight demand, mean-reverting intraday).
- ``tom_drift``: extra daily market drift on turn-of-the-month days
  (McConnell & Xu 2008 window: last trading day + first 3 of next month).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl

from quant_fund.research.replication.panel import make_panel, validate_panel
from quant_fund.research.replication.signals import turn_of_month_mask

TRADING_DAYS_PER_YEAR = 252


def _require_positive(name: str, value: int, minimum: int = 2) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def trading_days(start: str, n_days: int) -> np.ndarray:
    """Weekday ('business day') calendar — weekends only, no holiday calendar."""
    days = pd.bdate_range(start=pd.Timestamp(start), periods=n_days)
    return np.asarray(days.to_numpy())


def synthetic_panel(
    n_assets: int = 8,
    n_days: int = 320,
    seed: int = 0,
    *,
    ar1_rho: float = 0.0,
    lowvol_premium: float = 0.0,
    overnight_drift: float = 0.0,
    tom_drift: float = 0.0,
    start: str = "2020-01-02",
    base_vol: float = 0.02,
) -> pl.DataFrame:
    """Seeded synthetic equity panel with documented planted anomalies.

    Returns the canonical panel (``PANEL_COLUMNS``). The DGP: per-asset daily
    log return ``mu_i + ar1_rho * r_{t-1} + sigma_i * eps_{i,t}`` plus a small
    common market factor; the drift is split between the overnight and
    intraday legs per ``overnight_drift``. ``lowvol_premium`` tilts drift down
    for high-vol names; ``tom_drift`` adds to the market leg on ToM days.
    """
    n_assets = _require_positive("n_assets", n_assets, 2)
    n_days = _require_positive("n_days", n_days, 30)
    if isinstance(seed, bool):
        raise ValueError("seed must be an integer")
    rng = np.random.default_rng(int(seed))
    days = trading_days(start, n_days)

    sigmas = base_vol * rng.uniform(0.5, 1.5, n_assets)
    mus = np.full(n_assets, 0.0003)
    if lowvol_premium != 0.0:
        mus = mus - float(lowvol_premium) * (sigmas - sigmas.min())
    betas = rng.uniform(0.8, 1.2, n_assets)
    mkt_sigma = 0.008

    tom_days = turn_of_month_mask(days)
    mkt = rng.normal(0.0, mkt_sigma, n_days) + tom_drift * tom_days

    on_assets = np.zeros(n_assets, dtype=bool)
    on_assets[: n_assets // 2] = True

    frames: list[pl.DataFrame] = []
    for i in range(n_assets):
        eps = rng.normal(0.0, sigmas[i], n_days)
        total = np.empty(n_days)
        total[0] = mus[i] + betas[i] * mkt[0] + eps[0]
        for t in range(1, n_days):
            total[t] = mus[i] + ar1_rho * total[t - 1] + betas[i] * mkt[t] + eps[t]
        # Split each day's log return into overnight / intraday legs.
        if on_assets[i] and overnight_drift != 0.0:
            on = mus[i] * overnight_drift + betas[i] * mkt + eps * 0.5
            intr = total - on
        else:
            on = 0.5 * mus[i] + betas[i] * mkt * 0.5 + eps * 0.5
            intr = total - on
        close = 100.0 * np.exp(np.cumsum(total))
        prev_close = np.concatenate([[close[0] / np.exp(on[0])], close[:-1]])
        open_ = prev_close * np.exp(on)
        # Recompute close from open*intr so the OHLC identities hold exactly.
        close = open_ * np.exp(intr)
        frames.append(
            pl.DataFrame(
                {
                    "security_id": f"S{i:04d}",
                    "symbol": f"S{i:04d}",
                    "event_time": days,
                    "open": open_,
                    "close": close,
                }
            )
        )
    panel = make_panel(pl.concat(frames))
    return validate_panel(panel)
