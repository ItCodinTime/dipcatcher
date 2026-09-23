"""GARCH-MIDAS (Engle, Ghysels & Sohn 2013): volatility decomposed into
a short-run unit-mean GARCH component and a slowly varying long-run
component driven by MIDAS-weighted realized variances.

  r_t = mu + sqrt(tau_t * g_t) eps_t,
  g_t = (1 - alpha - beta) + alpha (r_{t-1} - mu)^2 / tau_{t-1} + beta g_{t-1},
  log tau_t = m + theta * sum_{k=1..K} w_k(omega) RV_{block(t)-k},

with beta-function MIDAS weights w_k ∝ (k/K)^{w1-1} (1-k/K)^{w2-1}
normalized to sum 1 (w1 fixed at 1 by default => weights decay).

Realized variance is computed on non-overlapping blocks of ``block``
days; tau_t is constant within each block. Gaussian QMLE.

Fail-closed: non-finite returns, insufficient blocks, invalid sizes.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy import optimize

Array = NDArray[np.float64]


def _beta_weights(k: int, w2: float) -> Array:
    """w_j ∝ (1 - j/K)^{w2-1} for j=1..K (w1=1 => monotone decay)."""
    j = np.arange(1, k + 1, dtype=float)
    w = (1.0 - j / k) ** (w2 - 1.0)
    w = np.where(np.isfinite(w), w, 0.0)
    s = w.sum()
    return w / s if s > 0 else np.full(k, 1.0 / k)


def _rv_blocks(r: Array, block: int) -> Array:
    n_b = r.size // block
    return np.array([float(np.sum(r[i * block : (i + 1) * block] ** 2)) for i in range(n_b)])


def garch_midas_fit(
    returns: Array,
    block: int = 21,
    k_lag: int = 12,
    fix_w2: float | None = None,
) -> dict[str, Array | float]:
    """QMLE fit. Returns mu, alpha, beta, m, theta, w2, tau/g paths, ll.

    ``fix_w2`` pins the weight shape (else estimated). theta is the
    long-run sensitivity to past realized variance.
    """
    r = np.asarray(returns, dtype=float).ravel()
    if r.size < 200 or not np.isfinite(r).all():
        raise ValueError("returns must be finite, >= 200 obs")
    if block < 5 or r.size // block < k_lag + 3:
        raise ValueError("need >= k_lag+3 complete blocks")
    rv = _rv_blocks(r, block)
    n_b = rv.size
    # map each day t to its block index (0-based); day in block b uses
    # RV lags of blocks b-1 .. b-K
    day_block = np.repeat(np.arange(n_b), block)
    n_eff = n_b * block

    w2_lo, w2_hi = 1.0, 30.0

    def paths(theta: Array) -> tuple[Array, Array] | None:
        mu, alpha, beta, m, th, w2 = theta
        if alpha < 0 or beta < 0 or alpha + beta >= 0.999 or not (w2_lo <= w2 <= w2_hi):
            return None
        w = _beta_weights(k_lag, w2)
        tau_b = np.full(n_b, np.nan)
        for b in range(n_b):
            lo = max(0, b - k_lag)
            hist = rv[lo:b]
            if hist.size == 0:
                tau_b[b] = np.exp(m)
            else:
                wt = w[k_lag - hist.size :]
                tau_b[b] = np.exp(m + th * float(wt @ hist))
        tau = tau_b[day_block]
        g = np.empty(n_eff)
        g[0] = 1.0
        for t in range(1, n_eff):
            g[t] = (
                (1.0 - alpha - beta) + alpha * (r[t - 1] - mu) ** 2 / tau[t - 1] + beta * g[t - 1]
            )
            if not np.isfinite(g[t]) or g[t] <= 0:
                return None
        return tau, g

    def nll(theta: Array) -> float:
        out = paths(theta)
        if out is None:
            return 1e12
        tau, g = out
        var = tau * g
        if (var <= 0).any() or not np.isfinite(var).all():
            return 1e12
        mu = theta[0]
        e = (r[:n_eff] - mu) ** 2 / var
        if not np.isfinite(e).all():
            return 1e12
        return float(0.5 * np.sum(np.log(2.0 * np.pi) + np.log(var) + e))

    sig2 = float(r.var())
    theta0 = np.array([r.mean(), 0.05, 0.85, np.log(max(sig2, 1e-8)), 0.5, 5.0])
    if fix_w2 is not None:
        w2v = float(fix_w2)

        def nll_fw(th: Array) -> float:
            return nll(np.concatenate([th, [w2v]]))

        res = optimize.minimize(
            nll_fw,
            theta0[:5],
            method="L-BFGS-B",
            bounds=[(None, None), (0.0, 0.5), (0.0, 0.999), (None, None), (None, None)],
        )
        theta_hat = np.concatenate([res.x, [w2v]])
    else:
        res = optimize.minimize(
            nll,
            theta0,
            method="L-BFGS-B",
            bounds=[
                (None, None),
                (0.0, 0.5),
                (0.0, 0.999),
                (None, None),
                (None, None),
                (w2_lo, w2_hi),
            ],
        )
        theta_hat = res.x
    out = paths(theta_hat)
    if out is None or not np.isfinite(res.fun):
        raise ValueError("GARCH-MIDAS fit failed")
    tau, g = out
    return {
        "mu": float(theta_hat[0]),
        "alpha": float(theta_hat[1]),
        "beta": float(theta_hat[2]),
        "m": float(theta_hat[3]),
        "theta": float(theta_hat[4]),
        "w2": float(theta_hat[5]),
        "tau": tau,
        "g": g,
        "total_vol": np.sqrt(tau * g),
        "rv_blocks": rv,
        "loglik": float(-res.fun),
        "n_eff": float(n_eff),
    }
