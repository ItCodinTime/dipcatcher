"""Classical event-study methodology for abnormal returns.

References:
- MacKinlay (1997): event-study econometrics review.
- Brown & Warner (1985): market-model abnormal returns, cross-sectional
  aggregation.
- Patell (1976): standardized abnormal return test.
- Corrado (1989): nonparametric rank test.
- Boehmer, Musumeci & Poulsen (1991): BMP standardized cross-sectional
  test robust to event-induced variance.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray
from scipy import stats

Array = NDArray[np.float64]


def _v(x: Array, n: int = 20) -> Array:
    v = np.asarray(x, dtype=float).reshape(-1)
    if v.size < n or not np.all(np.isfinite(v)):
        raise ValueError(f"series must be finite with length >= {n}")
    return v


def market_model_fit(est_returns: Array, est_market: Array) -> dict[str, float]:
    """OLS market model on an estimation window: r_i = a + b*r_m + e.

    Returns alpha, beta, residual std, and estimation length for use in
    ``abnormal_returns``."""
    ri = _v(est_returns)
    rm = _v(est_market)
    if ri.size != rm.size:
        raise ValueError("estimation windows must align")
    if rm.std() == 0:
        raise ValueError("market leg degenerate")
    X = np.column_stack([np.ones(ri.size), rm])
    beta, *_ = np.linalg.lstsq(X, ri, rcond=None)
    u = ri - X @ beta
    s = float(np.sqrt(u @ u / max(ri.size - 2, 1)))
    if s <= 0:
        raise ValueError("zero residual variance")
    return {"alpha": float(beta[0]), "beta": float(beta[1]), "sigma": s, "n_est": float(ri.size)}


def abnormal_returns(event_returns: Array, event_market: Array, fit: dict[str, float]) -> Array:
    """Brown–Warner market-model abnormal returns in the event window."""
    ri = _v(event_returns, n=1)
    rm = _v(event_market, n=1)
    if ri.size != rm.size:
        raise ValueError("event windows must align")
    return ri - (fit["alpha"] + fit["beta"] * rm)


def cumulative_abnormal(ars: Array) -> float:
    """CAR: sum of abnormal returns over the event window."""
    ar = _v(ars, n=1)
    return float(ar.sum())


def patell_test(ar_matrix: Array, est_lengths: Array) -> dict[str, float]:
    """Patell (1976) standardized abnormal-return test.

    ``ar_matrix`` is (n_events, window) abnormal returns standardized by
    each security's estimation-window residual std (callers pass SARs).
    Statistic ~ N(0,1) under no abnormal performance.
    """
    sar = np.asarray(ar_matrix, dtype=float)
    if sar.ndim != 2 or sar.shape[0] < 2 or not np.all(np.isfinite(sar)):
        raise ValueError("need finite (n_events >= 2, window) standardized ARs")
    L = np.asarray(est_lengths, dtype=float).reshape(-1)
    if L.size != sar.shape[0] or not np.all(L > 2) or not np.all(np.isfinite(L)):
        raise ValueError("est_lengths must be finite > 2 per event")
    n, w = sar.shape
    # Patell standardization: SARs already divided by sigma_i * C_t; we
    # aggregate with the classic correction sqrt((L-2)/(L-4)) per event.
    adj = np.sqrt(np.maximum((L - 2.0) / np.maximum(L - 4.0, 1e-9), 0.0))
    z_per_day = sar.mean(axis=0) * np.sqrt(n) * adj.mean()
    z_total = float(z_per_day.sum() / math.sqrt(w))
    return {
        "statistic": z_total,
        "pvalue": float(2.0 * (1.0 - stats.norm.cdf(abs(z_total)))),
        "n_events": float(n),
        "window": float(w),
    }


def corrado_rank_test(ar_matrix: Array) -> dict[str, float]:
    """Corrado (1989) rank test on abnormal returns.

    Ranks each security's ARs over the combined estimation+event sample;
    the event-window mean rank is compared to the null (K+1)/2.
    ``ar_matrix`` should contain ranks computed over each security's full
    sample (caller computes with scipy.stats.rankdata)."""
    R = np.asarray(ar_matrix, dtype=float)
    if R.ndim != 2 or R.shape[0] < 2 or not np.all(np.isfinite(R)):
        raise ValueError("need finite (n_events >= 2, K) rank matrix")
    n, K = R.shape
    mean_rank = R.mean()
    # Variance of mean rank under independence: K(K+1)/... standard form:
    # sd = sqrt( (K+1)^2 * (K-1) / (12 K) * ... ) — Corrado's s^2 =
    # (1/K) sum_k (mean_rank_k - (K+1)/2)^2 across days.
    day_means = R.mean(axis=0)
    s2 = float(np.sum((day_means - (K + 1.0) / 2.0) ** 2) / K)
    if s2 <= 0:
        raise ValueError("degenerate rank dispersion")
    z = float((mean_rank - (K + 1.0) / 2.0) / math.sqrt(s2 / K))
    return {
        "statistic": z,
        "pvalue": float(2.0 * (1.0 - stats.norm.cdf(abs(z)))),
        "mean_rank": float(mean_rank),
        "n_events": float(n),
    }


def bmp_test(ar_matrix: Array, sar_matrix: Array | None = None) -> dict[str, float]:
    """Boehmer–Musumeci–Poulsen (1991) standardized cross-sectional test.

    Robust to event-induced volatility: standardizes CARs by the
    cross-sectional std of standardized ARs rather than estimation
    variance. Input: (n_events, window) SARs."""
    sar = np.asarray(ar_matrix, dtype=float)
    if sar.ndim != 2 or sar.shape[0] < 3 or not np.all(np.isfinite(sar)):
        raise ValueError("need finite (n_events >= 3, window) SARs")
    n, w = sar.shape
    car = sar.sum(axis=1)
    sc = car.std(ddof=1)
    if sc <= 0:
        raise ValueError("zero cross-sectional dispersion")
    z = float(car.mean() * math.sqrt(n) / sc)
    return {
        "statistic": z,
        "pvalue": float(2.0 * (1.0 - stats.norm.cdf(abs(z)))),
        "mean_car": float(car.mean()),
        "n_events": float(n),
    }


def event_study(
    event_returns: Array,
    event_market: Array,
    est_returns: Array,
    est_market: Array,
) -> dict[str, float]:
    """Convenience wrapper: single-event market-model study.

    Fits the market model on the estimation window, returns ARs, CAR,
    and a two-sided p-value under Gaussian ARs (rough single-event test;
    for cross-sections use patell_test/bmp_test)."""
    fit = market_model_fit(est_returns, est_market)
    ars = abnormal_returns(event_returns, event_market, fit)
    car = cumulative_abnormal(ars)
    se = fit["sigma"] * math.sqrt(ars.size)  # ignores forecast-error term
    z = car / max(se, 1e-20)
    return {
        "car": car,
        "statistic": float(z),
        "pvalue": float(2.0 * (1.0 - stats.norm.cdf(abs(z)))),
        "alpha": fit["alpha"],
        "beta": fit["beta"],
    }
