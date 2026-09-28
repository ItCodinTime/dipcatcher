"""Point-in-time pair pipeline.

Every estimator call consumes a trailing window only: the output at index
``t`` is a deterministic function of observations ``<= t``. The no-lookahead
test mutates data after ``t`` and demands bit-identical outputs at ``<= t``.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from quant_fund.research.pairs.hedge import kalman_hedge_ratio, ols_hedge_ratio
from quant_fund.research.pairs.spread import bands_position, zscore_trailing

Array = NDArray[np.float64]

_HEDGE_METHODS = ("ols", "kalman")


def rolling_hedge_ratio(
    price_a: Array,
    price_b: Array,
    window: int,
    *,
    method: str = "ols",
    kalman_q: float = 1e-4,
) -> tuple[Array, Array]:
    """Trailing-window hedge ratio and intercept on log prices.

    ``(alpha[t], beta[t])`` is estimated on the window ``[t-w+1, t]`` of
    *log prices*; both are ``NaN`` for ``t < window - 1``. ``method`` is
    ``"ols"`` (static window OLS) or ``"kalman"`` (scalar Kalman filter on
    the window, terminal value used — Elliott et al. 2005).
    """
    if method not in _HEDGE_METHODS:
        raise ValueError(f"method must be one of {_HEDGE_METHODS}")
    a = np.asarray(price_a, dtype=float).reshape(-1)
    b = np.asarray(price_b, dtype=float).reshape(-1)
    if a.size != b.size or a.size < 20:
        raise ValueError("price series must match and have >= 20 obs")
    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        raise ValueError("prices must be finite")
    if np.any(a <= 0) or np.any(b <= 0):
        raise ValueError("prices must be positive")
    w = int(window)
    if w < 10 or w > a.size:
        raise ValueError("window must be in [10, len]")
    la, lb = np.log(a), np.log(b)
    alpha = np.full(a.size, np.nan)
    beta = np.full(a.size, np.nan)
    for t in range(w - 1, a.size):
        ya = la[t - w + 1 : t + 1]
        xb = lb[t - w + 1 : t + 1]
        if method == "ols":
            alpha[t], beta[t] = ols_hedge_ratio(ya, xb)
        else:
            path = kalman_hedge_ratio(ya, xb, q=kalman_q)
            alpha_full, _ = ols_hedge_ratio(ya, xb)
            alpha[t], beta[t] = alpha_full, float(path[-1])
    return alpha, beta


def pit_pair_signals(
    price_a: Array,
    price_b: Array,
    *,
    window: int = 120,
    z_window: int = 60,
    entry: float = 2.0,
    exit: float = 0.5,
    method: str = "ols",
    kalman_q: float = 1e-4,
) -> dict[str, Array]:
    """Full trailing pipeline for one pair.

    Returns per-date arrays: ``alpha``, ``beta`` (window hedge estimates),
    ``spread`` (``logA - alpha - beta * logB``), ``z`` (trailing z-score of
    the spread over ``z_window``), and ``position`` (the band state machine
    in ``{-1, 0, +1}``). All arrays are ``NaN``/flat until enough trailing
    history exists — nothing reads past index ``t``.
    """
    a = np.asarray(price_a, dtype=float).reshape(-1)
    b = np.asarray(price_b, dtype=float).reshape(-1)
    alpha, beta = rolling_hedge_ratio(a, b, window, method=method, kalman_q=kalman_q)
    la, lb = np.log(a), np.log(b)
    spread = la - alpha - beta * lb  # NaN during hedge warmup
    z = np.full(a.size, np.nan)
    valid = np.isfinite(spread)
    first = int(np.argmax(valid)) if valid.any() else a.size
    if a.size - first >= int(z_window):
        tail = zscore_trailing(spread[first:], int(z_window))
        z[first:] = tail
    position = bands_position(z, entry=entry, exit=exit)
    return {
        "alpha": alpha,
        "beta": beta,
        "spread": spread,
        "z": z,
        "position": position,
    }
