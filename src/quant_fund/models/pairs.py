"""Pairs trading research layer.

Pair selection and spread diagnostics — research/statistics only; no
trading logic, fills, or P&L claims.

References:
- Gatev, Goetzmann, Rouwenhorst (2006) distance approach.
- Vidyamurthy (2004) cointegration approach.
- Leung, Li (2016) OU optimal entry/exit (analytic thresholds).
- Chan (2013) half-life-based mean reversion filters.
- Elliott, van der Hoek, Malcolm (2005) Kalman pairs state space.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

from quant_fund.models.var_coint import engle_granger, spread_half_life

Array = NDArray[np.float64]


def _as_prices(p: Array) -> Array:
    m = np.asarray(p, dtype=float)
    if m.ndim != 2 or m.shape[0] < 30 or m.shape[1] < 2:
        raise ValueError("prices must be a finite (t, n) matrix")
    if not np.all(np.isfinite(m)) or np.any(m <= 0):
        raise ValueError("prices must be finite and positive")
    return m


def gatev_distance(prices: Array) -> Array:
    """Gatev et al. (2006) normalized-price squared-distance matrix.

    ``SSD_ij = sum_t (p_i^norm - p_j^norm)^2`` where normalized prices are
    each series divided by its first value. Returns (n, n) matrix.
    """
    m = _as_prices(prices)
    norm = m / m[0]
    diff = norm[:, :, None] - norm[:, None, :]
    return np.sum(diff * diff, axis=0)


def gatev_select(prices: Array, n_pairs: int = 10) -> dict[str, Array]:
    """Select the n_pairs lowest-SSD pairs (Gatev 2006 formation stage).

    Returns (n_pairs, 2) index pairs sorted by SSD and their distances.
    """
    ssd = gatev_distance(prices)
    n = ssd.shape[0]
    if n_pairs < 1 or n_pairs > n * (n - 1) // 2:
        raise ValueError("n_pairs out of range")
    iu = np.triu_indices(n, k=1)
    dists = ssd[iu]
    order = np.argsort(dists)[:n_pairs]
    pairs = np.column_stack([iu[0][order], iu[1][order]]).astype(float)
    return {"pairs": pairs, "distances": dists[order], "ssd": ssd}


def cointegration_screen(prices: Array, alpha: float = 0.05) -> dict[str, Array]:
    """Vidyamurthy (2004) screen: Engle–Granger tau for all pairs.

    Returns the tau matrix (i, j) = EG tau for log prices i~j, plus the
    boolean mask of pairs passing at level alpha (approximate 5% CV).
    """
    m = np.log(_as_prices(prices))
    t, n = m.shape
    tau = np.full((n, n), np.nan)
    pmask = np.zeros((n, n), dtype=bool)
    for i in range(n):
        for j in range(i + 1, n):
            try:
                out = engle_granger(m[:, i], m[:, j])
            except ValueError:
                continue
            tau[i, j] = tau[j, i] = out["tau"]
            if out["tau"] < out["cv_5pct"]:
                pmask[i, j] = pmask[j, i] = True
    return {"tau": tau, "passes": pmask}


def zscore_spread_stats(price_a: Array, price_b: Array) -> dict[str, float]:
    """Spread diagnostics for a candidate pair (log-price spread).

    Returns hedge ratio, z-score series stats, half-life, zero-crossing
    count, ADF tau of the residual spread, and stationarity flag.
    """
    a = np.asarray(price_a, dtype=float).reshape(-1)
    b = np.asarray(price_b, dtype=float).reshape(-1)
    if a.size != b.size or a.size < 30 or not np.all(np.isfinite(a)):
        raise ValueError("price series must match and have >= 30 obs")
    if np.any(a <= 0) or np.any(b <= 0):
        raise ValueError("prices must be positive")
    la, lb = np.log(a), np.log(b)
    # Hedge ratio from levels regression.
    x = np.column_stack([np.ones(la.size), lb])
    beta, *_ = np.linalg.lstsq(x, la, rcond=None)
    spread = la - beta[0] - beta[1] * lb
    z = (spread - spread.mean()) / max(spread.std(), 1e-12)
    crossings = int(np.sum(np.diff(np.sign(z)) != 0))
    hl = spread_half_life(spread)
    # ADF on spread (no const, no trend — spread is already residual).
    dy = np.diff(spread)
    lag = spread[:-1]
    xr = np.column_stack([lag])
    b, *_ = np.linalg.lstsq(xr, dy, rcond=None)
    e = dy - xr @ b
    s2 = float(e @ e) / max(dy.size - 1, 1)
    tau = float(b[0] / math.sqrt(max(np.linalg.pinv(xr.T @ xr)[0, 0] * s2, 1e-16)))
    return {
        "hedge_ratio": float(beta[1]),
        "intercept": float(beta[0]),
        "spread_std": float(spread.std()),
        "half_life": hl,
        "zero_crossings": float(crossings),
        "adf_tau": tau,
        "stationary_5pct": float(tau < -2.86),  # no-const ADF 5% cv
        "mean_zscore": float(np.mean(np.abs(z))),
    }


def ou_optimal_bands(theta: float, mu: float, sigma: float, r: float = 0.0) -> dict[str, float]:
    """Leung–Li (2016) OU optimal entry/exit thresholds (long-side).

    Solves for the optimal buy level b* and sell level a* of an OU spread
    under discount rate r via the first-order conditions on the
    exponentially-weighted eigenfunctions. Simplified numeric solution:
    grid-search over entry/exit on the discounted expected-PnL functional.
    """
    if theta <= 0 or sigma <= 0 or r < 0:
        raise ValueError("need theta>0, sigma>0, r>=0")
    if not np.isfinite(theta) or not np.isfinite(mu):
        raise ValueError("theta/mu must be finite")
    # Solve expected discounted holding return over (entry, exit) grid.
    sd = sigma / math.sqrt(2.0 * theta)  # stationary sd
    xs = mu + np.linspace(-4.0, 2.0, 200) * sd
    best = {"entry": np.nan, "exit": np.nan, "value": -np.inf}
    # Expected hitting time of OU for entry b then exit a: use closed-form
    # approximation via stationary occupancy; value = (a - b) discounted by
    # E[exp(-r(tau_b + tau_a))]. Approximate E[tau] via mean-reversion.
    for b_ in xs[xs < mu]:
        # Time to reach the entry band from a high starting level.
        tau_b = _ou_hitting_time(xs[-1], b_, theta, mu, sd)
        for a_ in xs[xs > b_]:
            # Time to exit at a_ after entering at b_.
            t_hold = _ou_hitting_time(b_, a_, theta, mu, sd)
            val = (a_ - b_) * math.exp(-r * (tau_b + t_hold))
            if val > best["value"]:
                best = {"entry": float(b_), "exit": float(a_), "value": val}
    return best


def _ou_hitting_time(x0: float, target: float, theta: float, mu: float, sd: float) -> float:
    """Approximate mean first passage for OU via the diffusion-scale integral.

    Uses the standard closed-form approximation
    ``E[T] ~ (1/theta) * (F-transform)``; for tractability uses the
    log-ratio asymptotic ``log(|x0-mu|/|target-mu|)/theta`` clipped at 1/theta.
    """
    d0 = abs(x0 - mu) / sd
    d1 = abs(target - mu) / sd
    return max(math.log(max(d0 / max(d1, 1e-8), 1e-8)) / theta, 1.0 / theta)


def pair_quality_score(price_a: Array, price_b: Array) -> dict[str, float]:
    """Composite pair quality: combines cointegration tau, half-life, and
    zero-crossing frequency into a single selection score in [0, 1].

    Score = w1 * sigmoid(-tau) + w2 * sigmoid(-hl_centered) + w3 * cross_rate.
    Higher = better mean-reversion candidate.
    """
    s = zscore_spread_stats(price_a, price_b)
    tau_score = float(1.0 / (1.0 + math.exp(s["adf_tau"] + 2.86)))
    hl = s["half_life"]
    hl_score = (
        float(1.0 / (1.0 + math.exp((math.log(max(hl, 1e-6)) - math.log(20.0)) * 2.0)))
        if np.isfinite(hl)
        else 0.0
    )
    cross_rate = min(s["zero_crossings"] / (len(np.asarray(price_a)) / 10.0), 1.0)
    score = 0.4 * tau_score + 0.3 * hl_score + 0.3 * cross_rate
    return {
        "score": score,
        "tau_score": tau_score,
        "hl_score": hl_score,
        "cross_rate": cross_rate,
    }
