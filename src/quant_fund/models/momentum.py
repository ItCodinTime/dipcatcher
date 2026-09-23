"""Cross-sectional and time-series momentum signals.

References:
- Jegadeesh & Titman (1993): formation-period return momentum.
- George & Hwang (2004): 52-week high signal.
- Blitz, Huij & Martens (2011): residual momentum.
- Moskowitz, Ooi & Pedersen (2012): time-series momentum (sign of own
  past return scaled by volatility).
- Novy-Marx (2012): intermediate-horizon momentum (t-12 to t-7).
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray
from scipy import stats

Array = NDArray[np.float64]


def _px(prices: Array, min_t: int = 60) -> Array:
    p = np.asarray(prices, dtype=float)
    if p.ndim != 2 or p.shape[0] < min_t or not np.all(np.isfinite(p)):
        raise ValueError(f"prices must be finite (T >= {min_t}, N)")
    if np.any(p <= 0):
        raise ValueError("prices must be positive")
    return p


def jt_momentum(prices: Array, formation: int = 252, skip: int = 21) -> Array:
    """Jegadeesh–Titman (1993) momentum: cumulative log return over
    [t-skip-formation, t-skip], skipping the most recent month to avoid
    short-term reversal. Returns (T, N) NaN-warmup signals; the last
    finite row holds the current cross-sectional scores."""
    p = _px(prices)
    T, N = p.shape
    if formation + skip >= T:
        raise ValueError("formation + skip must be < T")
    lp = np.log(p)
    out = np.full((T, N), np.nan)
    for t in range(formation + skip, T):
        out[t] = lp[t - skip] - lp[t - skip - formation]
    return out


def high_52w(prices: Array, window: int = 252) -> Array:
    """George–Hwang (2004): price / rolling 52-week high in (0, 1]."""
    p = _px(prices)
    T, N = p.shape
    if window >= T:
        raise ValueError("window must be < T")
    out = np.full((T, N), np.nan)
    for t in range(window, T):
        hi = p[t - window : t].max(axis=0)
        out[t] = p[t] / np.maximum(hi, 1e-12)
    return out


def residual_momentum(
    prices: Array,
    factor_returns: Array,
    formation: int = 252,
    skip: int = 21,
    est_window: int = 252,
) -> dict[str, Array]:
    """Blitz–Huij–Martens (2011) residual momentum.

    For each asset, regress returns on the factor over ``est_window``
    days ending at t-skip, then accumulate residual log-returns over the
    formation window. Returns per-asset signals (T, N) and betas."""
    p = _px(prices)
    f = np.asarray(factor_returns, dtype=float).reshape(-1)
    T, N = p.shape
    if f.size != T or not np.all(np.isfinite(f)):
        raise ValueError("factor_returns must be finite (T,)")
    need = formation + skip + est_window
    if need >= T:
        raise ValueError("formation + skip + est_window must be < T")
    r = np.diff(np.log(p), axis=0)  # (T-1, N); r[i] ~ t = i+1
    fr = f  # treat factor as already a return series aligned with p
    sig = np.full((T, N), np.nan)
    betas = np.full((T, N), np.nan)
    for t in range(need, T):
        # Estimation window: returns indexed t-est-skip .. t-skip-1.
        e0 = t - skip - est_window - 1
        e1 = t - skip - 1
        fe = fr[e0:e1]
        if fe.size < est_window // 2 or fe.std() == 0:
            continue
        Xd = np.column_stack([np.ones(fe.size), fe])
        # Formation window: returns t-skip-formation .. t-skip-1.
        f0 = t - skip - formation
        f1 = t - skip
        for i in range(N):
            ri = r[e0:e1, i]
            beta, *_ = np.linalg.lstsq(Xd, ri, rcond=None)
            betas[t, i] = beta[1]
            resid = r[f0:f1, i] - (beta[0] + beta[1] * fr[f0:f1])
            sig[t, i] = resid.sum()
    return {"signal": sig, "beta": betas}


def tsmom_signal(returns: Array, lookback: int = 252, vol_window: int = 60) -> dict[str, Array]:
    """Moskowitz–Ooi–Pedersen (2012) time-series momentum.

    ``signal_t = sign(r_{t-lookback:t}) / sigma_t`` — or scaled return
    ``r_past / sigma``. We return both the sign and the vol-scaled
    magnitude. Input is a (T,) or (T, N) return series."""
    r = np.asarray(returns, dtype=float)
    if r.ndim == 1:
        r = r[:, None]
        squeeze = True
    else:
        squeeze = False
    if not np.all(np.isfinite(r)) or r.shape[0] < lookback + vol_window + 5:
        raise ValueError("returns too short for lookback + vol_window")
    T, N = r.shape
    sig = np.full((T, N), np.nan)
    sign = np.full((T, N), np.nan)
    for t in range(lookback + vol_window, T):
        past = r[t - lookback : t].sum(axis=0)
        vol = r[t - vol_window : t].std(axis=0, ddof=1)
        vol = np.maximum(vol, 1e-12)
        sig[t] = past / vol
        sign[t] = np.sign(past)
    if squeeze:
        return {"signal": sig[:, 0], "sign": sign[:, 0]}
    return {"signal": sig, "sign": sign}


def tsmom_tstat(returns: Array, lookback: int = 252) -> dict[str, float]:
    """MOP t-statistic on the time-series-momentum strategy return.

    Strategy return = sign(past return) * next-period return; tests
    whether its mean is > 0 via a Newey-West t-stat."""
    r = np.asarray(returns, dtype=float).reshape(-1)
    if r.size < lookback + 50 or not np.all(np.isfinite(r)):
        raise ValueError("returns too short")
    T = r.size
    strat_l = [np.sign(r[t - lookback : t].sum()) * r[t + 1] for t in range(lookback, T - 1)]
    strat = np.asarray(strat_l)
    n = strat.size
    mu = float(strat.mean())
    d = strat - mu
    lag = int(math.floor(4.0 * (n / 100.0) ** 0.25))
    lrv = float(d @ d / n)
    for h in range(1, lag + 1):
        w = 1.0 - h / (lag + 1.0)
        lrv += 2.0 * w * float(np.dot(d[h:], d[:-h]) / n)
    lrv = max(lrv, 1e-20)
    t = float(mu / math.sqrt(lrv / n))
    return {
        "t_stat": t,
        "pvalue": float(1.0 - stats.norm.cdf(t)),
        "mean_ret": mu,
        "n": float(n),
    }
