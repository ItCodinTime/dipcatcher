"""pyRisk-style market risk (VaR / ES / EVT / backtests).

In-repo implementation of the lprtk/pyRisk contract (MIT):
https://github.com/lprtk/pyRisk

Dipcatcher already owns historical/Gaussian VaR-ES and Kupiec/Christoffersen.
This module is the named pyRisk API plus EVT (Pickands / Hill) and Monte Carlo,
without a Hub/pip dependency or matplotlib. ``alpha`` here is the *tail
probability* (pyRisk default 0.01 = 99% VaR), not the loss quantile.

Research / paper-book diagnostic only — not a live P&L claim.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy import stats

from quant_fund.metrics.probability import christoffersen_cc, kupiec_pof
from quant_fund.metrics.risk import gaussian_es, gaussian_var, historical_es, historical_var

Array = NDArray[np.float64]


def _finite_1d(array: Array) -> Array:
    x = np.asarray(array, dtype=float).reshape(-1)
    return x[np.isfinite(x)]


def _loss_alpha(tail_p: float) -> float:
    if not np.isfinite(tail_p) or not 0.0 < float(tail_p) < 1.0:
        raise ValueError("alpha (tail probability) must be in (0, 1)")
    return float(1.0 - tail_p)


class Statistics:
    """Sample moments. Fisher kurtosis is optional (pyRisk default is Pearson)."""

    def __init__(self, array: Array, axis: int = 0) -> None:
        self.x = _finite_1d(array)
        self.axis = int(axis)

    def minimum(self) -> float:
        return float(np.min(self.x)) if self.x.size else float("nan")

    def maximum(self) -> float:
        return float(np.max(self.x)) if self.x.size else float("nan")

    def mean(self) -> float:
        return float(np.mean(self.x)) if self.x.size else float("nan")

    def var(self) -> float:
        return float(np.var(self.x, ddof=1)) if self.x.size >= 2 else float("nan")

    def std(self) -> float:
        return float(np.std(self.x, ddof=1)) if self.x.size >= 2 else float("nan")

    def skewness(self) -> float:
        if self.x.size < 3:
            return float("nan")
        return float(stats.skew(self.x, bias=False))

    def kurtosis(self, fisher: bool = False) -> float:
        if self.x.size < 4:
            return float("nan")
        return float(stats.kurtosis(self.x, fisher=bool(fisher), bias=False))


class ValueAtRisk:
    """VaR on a return series. Losses are ``-returns`` internally."""

    def __init__(self, array: Array, alpha: float = 0.01, axis: int = 0) -> None:
        self.returns = _finite_1d(array)
        self.losses = -self.returns
        self.alpha = float(alpha)
        self.loss_alpha = _loss_alpha(self.alpha)
        self.axis = int(axis)

    def empirical_var(self, plot: bool = False, bins: int = 50) -> float:
        del plot, bins
        return historical_var(self.losses, self.loss_alpha)

    def parametrical_var(self, plot: bool = False, bins: int = 50) -> float:
        del plot, bins
        return gaussian_var(self.losses, self.loss_alpha)

    def non_parametrical_var(
        self,
        random_state: int = 42,
        n_iter: int = 20_000,
        plot: bool = False,
        bins: int = 50,
    ) -> float:
        del plot, bins
        if self.returns.size < 8:
            return float("nan")
        rng = np.random.default_rng(int(random_state))
        draws = rng.choice(self.returns, size=int(n_iter), replace=True)
        return historical_var(-draws, self.loss_alpha)

    def extreme_var(self, k: int = 5, plot: bool = False, bins: int = 50) -> float:
        del plot, bins
        return hill_var(self.losses, self.loss_alpha, k=int(k))


class ExpectedShortfall:
    def __init__(self, array: Array, alpha: float = 0.01, axis: int = 0) -> None:
        self.returns = _finite_1d(array)
        self.losses = -self.returns
        self.alpha = float(alpha)
        self.loss_alpha = _loss_alpha(self.alpha)
        self.axis = int(axis)

    def empirical_cvar(self, plot: bool = False, bins: int = 50) -> float:
        del plot, bins
        return historical_es(self.losses, self.loss_alpha)

    def parametrical_cvar(self, plot: bool = False, bins: int = 50) -> float:
        del plot, bins
        return gaussian_es(self.losses, self.loss_alpha)

    def non_parametrical_cvar(
        self,
        random_state: int = 42,
        n_iter: int = 20_000,
        plot: bool = False,
        bins: int = 50,
    ) -> float:
        del plot, bins
        if self.returns.size < 8:
            return float("nan")
        rng = np.random.default_rng(int(random_state))
        draws = rng.choice(self.returns, size=int(n_iter), replace=True)
        return historical_es(-draws, self.loss_alpha)

    def extreme_cvar(self, k: int = 5, plot: bool = False, bins: int = 50) -> float:
        del plot, bins
        return hill_es(self.losses, self.loss_alpha, k=int(k))


class PickandsEstimator:
    def __init__(self, array: Array, k: int = 5, alpha: float = 0.01, axis: int = 0) -> None:
        self.x = np.sort(_finite_1d(array))
        self.k = int(k)
        self.alpha = float(alpha)
        self.axis = int(axis)

    def gev_parameter(self, plot: bool = False, n_iter: int = 100) -> float:
        del plot, n_iter
        return pickands_xi(self.x, self.k)


class Leadbetter:
    """Extremal index via runs de-clustering of threshold exceedances."""

    def __init__(self, array: Array, threshold: float, axis: int = 0) -> None:
        self.x = _finite_1d(array)
        self.threshold = float(threshold)
        self.axis = int(axis)

    def extremal_index(self) -> float:
        if not np.isfinite(self.threshold) or self.x.size < 4:
            return float("nan")
        exceed = self.x > self.threshold
        n_ex = int(np.sum(exceed))
        if n_ex == 0:
            return float("nan")
        padded = np.concatenate([[False], exceed, [False]])
        clusters = int(np.sum((~padded[:-1]) & padded[1:]))
        return float(clusters / n_ex)


class BackTesting:
    """VaR exception tests. ``threshold`` is a *loss* VaR (positive)."""

    def __init__(self, array: Array, axis: int = 0) -> None:
        self.returns = _finite_1d(array)
        self.losses = -self.returns
        self.axis = int(axis)

    def _hits(self, threshold: float) -> Array:
        if not np.isfinite(threshold):
            return np.asarray([], dtype=float)
        return (self.losses > float(threshold)).astype(float)

    def student_test(self, threshold: float, alpha: float = 0.05) -> dict[str, Any]:
        del threshold
        x = self.returns
        if x.size < 8:
            return {"statistic": float("nan"), "p_value": float("nan"), "n": int(x.size)}
        stat, p = stats.ttest_1samp(x, 0.0)
        return {"statistic": float(stat), "p_value": float(p), "n": int(x.size), "alpha": float(alpha)}

    def normal_test(self, threshold: float, alpha: float = 0.01) -> dict[str, Any]:
        del threshold
        x = self.returns
        if x.size < 8:
            return {"statistic": float("nan"), "p_value": float("nan"), "n": int(x.size)}
        stat, p = stats.normaltest(x)
        return {"statistic": float(stat), "p_value": float(p), "n": int(x.size), "alpha": float(alpha)}

    def kupiec_test(self, threshold: float, alpha: float = 0.01) -> dict[str, Any]:
        hits = self._hits(threshold)
        rate, lr, p = kupiec_pof(hits, float(alpha))
        return {"hit_rate": float(rate), "lr": float(lr), "p_value": float(p), "n": int(hits.size)}

    def christoffersen_test(self, threshold: float, alpha: float = 0.01) -> dict[str, Any]:
        hits = self._hits(threshold)
        lr, p, extras = christoffersen_cc(hits, float(alpha))
        return {"lr": float(lr), "p_value": float(p), "n": int(hits.size), **extras}

    def kupiec_christoffersen_test(self, threshold: float, alpha: float = 0.01) -> dict[str, Any]:
        kupiec = self.kupiec_test(threshold, alpha=alpha)
        cc = self.christoffersen_test(threshold, alpha=alpha)
        return {"kupiec": kupiec, "christoffersen": cc}

    def var_diameter(self, var: list[float], alpha: float = 0.01) -> dict[str, Any]:
        del alpha
        v = np.asarray(var, dtype=float)
        v = v[np.isfinite(v)]
        if v.size == 0:
            return {"diameter": float("nan"), "min": float("nan"), "max": float("nan")}
        return {"diameter": float(np.max(v) - np.min(v)), "min": float(np.min(v)), "max": float(np.max(v))}

    def cvar_diameter(self, cvar: list[float], alpha: float = 0.01) -> dict[str, Any]:
        return self.var_diameter(cvar, alpha=alpha)


def pickands_xi(sorted_sample: Array, k: int) -> float:
    """Pickands (1975) tail-index estimator on an ascending sample."""
    x = np.asarray(sorted_sample, dtype=float).reshape(-1)
    x = np.sort(x[np.isfinite(x)])
    n = int(x.size)
    kk = int(k)
    if kk < 1 or 4 * kk >= n:
        return float("nan")
    a = float(x[n - kk - 1])
    b = float(x[n - 2 * kk - 1])
    c = float(x[n - 4 * kk - 1])
    if not (a > b > c):
        return float("nan")
    return float(np.log((a - b) / (b - c)) / np.log(2.0))


def hill_var(losses: Array, loss_alpha: float, k: int = 20) -> float:
    """Hill EVT VaR on losses. Falls back to historical VaR if the tail is short."""
    x = np.sort(_finite_1d(losses))
    n = int(x.size)
    kk = min(int(k), max(n // 5, 1))
    if n < 16 or kk < 2 or kk >= n:
        return historical_var(losses, loss_alpha)
    tail = x[-kk:]
    if np.any(tail <= 0):
        tail = tail - float(np.min(tail)) + 1e-8
    thresh = float(tail[0])
    xi = float(np.mean(np.log(np.maximum(tail, 1e-12)) - np.log(max(thresh, 1e-12))))
    if not np.isfinite(xi) or xi <= -0.5:
        return historical_var(losses, loss_alpha)
    p = 1.0 - float(loss_alpha)
    return float(thresh * ((n / kk) * p) ** (-xi))


def hill_es(losses: Array, loss_alpha: float, k: int = 20) -> float:
    var = hill_var(losses, loss_alpha, k=k)
    x = _finite_1d(losses)
    n = int(x.size)
    kk = min(int(k), max(n // 5, 1))
    if n < 16 or kk < 2:
        return historical_es(losses, loss_alpha)
    tail = np.sort(x)[-kk:]
    if np.any(tail <= 0):
        tail = tail - float(np.min(tail)) + 1e-8
    thresh = float(tail[0])
    xi = float(np.mean(np.log(np.maximum(tail, 1e-12)) - np.log(max(thresh, 1e-12))))
    if not np.isfinite(xi) or xi >= 1.0 or xi <= -0.5:
        return historical_es(losses, loss_alpha)
    return float(var / (1.0 - xi))
