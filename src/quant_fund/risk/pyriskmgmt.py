"""pyriskmgmt-style equity portfolio VaR / ES.

In-repo subset of GianMarcoOddo/pyriskmgmt (MIT):
https://github.com/GianMarcoOddo/pyriskmgmt

Covers the equity engine used to *size* a paper book: historical and EWMA
parametric VaR/ES, portfolio aggregation, component VaR, and scale-to-ES.
Derivative and fixed-income modules are not vendored (hundreds of KB of
option/bond engines Dipcatcher does not trade).

Research / paper-book diagnostic only — not a live P&L claim.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.stats import norm

from quant_fund.metrics.risk import gaussian_es, gaussian_var, historical_es, historical_var

Array = NDArray[np.float64]


def _finite_1d(array: Array) -> Array:
    x = np.asarray(array, dtype=float).reshape(-1)
    return x[np.isfinite(x)]


def ewma_variance(returns: Array, lam: float = 0.94) -> float:
    """RiskMetrics EWMA variance of a return series (last filtered value)."""
    r = _finite_1d(returns)
    if r.size < 2 or not np.isfinite(lam) or not 0.0 < float(lam) < 1.0:
        return float("nan")
    var = float(np.var(r[: min(20, r.size)], ddof=1))
    for value in r:
        var = float(lam) * var + (1.0 - float(lam)) * float(value) * float(value)
    return var


def ewma_var_es(returns: Array, alpha: float = 0.95, lam: float = 0.94) -> tuple[float, float]:
    """Parametric Gaussian VaR/ES using EWMA sigma. ``alpha`` is the loss quantile."""
    r = _finite_1d(returns)
    if r.size < 8:
        return float("nan"), float("nan")
    mu = float(np.mean(r[-min(60, r.size) :]))
    sig = float(np.sqrt(max(ewma_variance(r, lam=lam), 0.0)))
    losses = -r
    if not np.isfinite(sig) or sig <= 0:
        return gaussian_var(losses, alpha), gaussian_es(losses, alpha)
    z = float(norm.ppf(alpha))
    var = -mu + sig * z
    es = -mu + sig * float(norm.pdf(z)) / (1.0 - alpha)
    return float(var), float(es)


def portfolio_returns(asset_returns: Array, weights: Array) -> Array:
    """``asset_returns`` is (T, N); ``weights`` is (N,). Causal: caller lags weights."""
    r = np.asarray(asset_returns, dtype=float)
    w = np.asarray(weights, dtype=float).reshape(-1)
    if r.ndim != 2 or w.size != r.shape[1]:
        raise ValueError("asset_returns must be (T, N) and weights length N")
    r = np.where(np.isfinite(r), r, 0.0)
    w = np.where(np.isfinite(w), w, 0.0)
    return r @ w


def portfolio_var_es(
    asset_returns: Array,
    weights: Array,
    *,
    alpha: float = 0.95,
    method: str = "historical",
    lam: float = 0.94,
) -> dict[str, Any]:
    """Portfolio VaR/ES. ``method`` is historical | gaussian | ewma."""
    port = portfolio_returns(asset_returns, weights)
    losses = -port
    method_n = str(method).strip().lower()
    if method_n == "historical":
        var, es = historical_var(losses, alpha), historical_es(losses, alpha)
    elif method_n == "gaussian":
        var, es = gaussian_var(losses, alpha), gaussian_es(losses, alpha)
    elif method_n == "ewma":
        var, es = ewma_var_es(port, alpha=alpha, lam=lam)
    else:
        raise ValueError("method must be historical, gaussian, or ewma")
    return {
        "var": float(var),
        "es": float(es),
        "method": method_n,
        "alpha": float(alpha),
        "n": int(port.size),
        "catalog": "hedge_lab_analytics",
        "research_only": True,
        "execution_claim": "paper_backtest",
    }


def component_var(asset_returns: Array, weights: Array, *, alpha: float = 0.95) -> Array:
    """Euler component VaR under a covariance approximation."""
    r = np.asarray(asset_returns, dtype=float)
    w = np.asarray(weights, dtype=float).reshape(-1)
    if r.ndim != 2 or r.shape[0] < 8 or w.size != r.shape[1]:
        return np.full(w.size, np.nan)
    r = np.where(np.isfinite(r), r, 0.0)
    w = np.where(np.isfinite(w), w, 0.0)
    cov = np.cov(r, rowvar=False, ddof=1)
    if cov.ndim == 0:
        return np.full(w.size, np.nan)
    sigma_w = cov @ w
    port_var = float(w @ sigma_w)
    if not np.isfinite(port_var) or port_var <= 0:
        return np.full(w.size, np.nan)
    port_sigma = float(np.sqrt(port_var))
    z = float(norm.ppf(alpha))
    mvar = (sigma_w / port_sigma) * z
    return np.asarray(mvar * w, dtype=float)


def scale_weights_to_es(
    asset_returns: Array,
    weights: Array,
    *,
    es_limit: float,
    alpha: float = 0.95,
    method: str = "ewma",
) -> tuple[Array, dict[str, Any]]:
    """Multiply weights by ``es_limit / ES`` when predicted ES exceeds the cap."""
    blob = portfolio_var_es(asset_returns, weights, alpha=alpha, method=method)
    es = float(blob["es"])
    w = np.asarray(weights, dtype=float).reshape(-1)
    if not np.isfinite(es_limit) or es_limit <= 0:
        raise ValueError("es_limit must be finite and positive")
    if not np.isfinite(es) or es <= 0:
        return w, {**blob, "scale": 1.0, "capped": False}
    scale = float(min(1.0, es_limit / es))
    return w * scale, {**blob, "scale": scale, "capped": bool(scale < 1.0 - 1e-12)}
