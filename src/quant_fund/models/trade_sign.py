"""Trade signing / order-flow classification.

- Tick rule (trade signed by last nonzero price change).
- Lee & Ready (1991) quote rule with tick-rule fallback at the midpoint.
- Bulk Volume Classification (Easley, Lopez de Prado & O'Hara 2012):
  assign a fraction V * Phi(dP / sigma_dP) of each bar to buy-initiated.

Fail-closed: mismatched lengths, non-finite inputs, empty arrays.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.stats import norm

Array = NDArray[np.float64]


def tick_rule(prices: Array) -> Array:
    """Sign each trade by the last nonzero price change: +1 buy, -1 sell.

    First observation is 0 (no predecessor). Zero-change ticks inherit
    the last nonzero sign.
    """
    p = np.asarray(prices, dtype=float).ravel()
    if p.size == 0 or not np.isfinite(p).all():
        raise ValueError("prices must be finite and non-empty")
    dp = np.diff(p)
    s = np.sign(dp)
    out = np.zeros(p.size)
    last = 0.0
    for i in range(1, p.size):
        if s[i - 1] != 0.0:
            last = s[i - 1]
        out[i] = last
    return out


def lee_ready(price: Array, bid: Array, ask: Array) -> Array:
    """Lee-Ready (1991) classification.

    price > mid -> buy (+1); price < mid -> sell (-1); price == mid ->
    tick rule on the quote midpoint. Caller is responsible for aligning
    quotes to trades (Lee-Ready use a ~5s lag).
    """
    p = np.asarray(price, dtype=float).ravel()
    b = np.asarray(bid, dtype=float).ravel()
    a = np.asarray(ask, dtype=float).ravel()
    if not (p.shape == b.shape == a.shape):
        raise ValueError("price, bid, ask must share shape")
    if (
        p.size == 0
        or not np.isfinite(p).all()
        or not np.isfinite(b).all()
        or not np.isfinite(a).all()
    ):
        raise ValueError("inputs must be finite")
    if (b > a).any():
        raise ValueError("crossed quotes: bid > ask")
    mid = 0.5 * (b + a)
    signs = np.zeros(p.size)
    signs[p > mid] = 1.0
    signs[p < mid] = -1.0
    at_mid = signs == 0.0
    if at_mid.any():
        tick = tick_rule(mid)
        signs[at_mid] = tick[at_mid]
    return signs


def bulk_volume_classify(
    close: Array,
    volume: Array,
    sigma_dp: float | None = None,
) -> dict[str, Array]:
    """Bulk Volume Classification (Easley-Lopez de Prado-O'Hara).

    For each bar i: V_buy_i = V_i * Phi((P_i - P_{i-1}) / sigma_dP).
    ``sigma_dp`` defaults to the sample std of first differences.
    Returns dict with buy_volume, sell_volume, buy_share arrays.
    """
    p = np.asarray(close, dtype=float).ravel()
    v = np.asarray(volume, dtype=float).ravel()
    if p.shape != v.shape or p.size < 2:
        raise ValueError("close and volume must match, >= 2 bars")
    if not np.isfinite(p).all() or not np.isfinite(v).all() or (v < 0).any():
        raise ValueError("close finite, volume finite >= 0")
    dp = np.diff(p)
    if sigma_dp is None:
        sigma_dp = float(dp.std(ddof=1))
    if not np.isfinite(sigma_dp) or sigma_dp <= 0.0:
        raise ValueError("sigma_dp must be > 0")
    frac = norm.cdf(dp / sigma_dp)
    buy = v[1:] * frac
    sell = v[1:] * (1.0 - frac)
    with np.errstate(invalid="ignore", divide="ignore"):
        share = buy / np.where(v[1:] > 0, v[1:], np.nan)
    return {
        "buy_volume": buy,
        "sell_volume": sell,
        "buy_share": np.asarray(share, dtype=float),
        "sigma_dp": np.asarray(sigma_dp),
    }


def signed_volume(signs: Array, volume: Array) -> dict[str, float]:
    """Aggregate signed trade volume into buy/sell/imbalance totals."""
    s = np.asarray(signs, dtype=float).ravel()
    v = np.asarray(volume, dtype=float).ravel()
    if s.shape != v.shape or s.size == 0:
        raise ValueError("signs and volume must match, non-empty")
    if not np.isfinite(s).all() or not np.isfinite(v).all() or (v < 0).any():
        raise ValueError("finite inputs, volume >= 0")
    buy = float(v[s > 0].sum())
    sell = float(v[s < 0].sum())
    tot = buy + sell
    return {
        "buy_volume": buy,
        "sell_volume": sell,
        "imbalance": buy - sell,
        "buy_share": buy / tot if tot > 0 else float("nan"),
    }


def signing_accuracy(signs_hat: Array, signs_true: Array) -> float:
    """Share of correctly signed trades (accuracy vs known labels)."""
    a = np.asarray(signs_hat, dtype=float).ravel()
    b = np.asarray(signs_true, dtype=float).ravel()
    if a.shape != b.shape or a.size == 0:
        raise ValueError("shapes must match, non-empty")
    mask = (a != 0.0) & (b != 0.0)
    if mask.sum() == 0:
        raise ValueError("no signed trades to compare")
    return float((a[mask] == b[mask]).mean())
