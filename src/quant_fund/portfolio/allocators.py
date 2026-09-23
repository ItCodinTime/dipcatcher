"""Portfolio allocators beyond mean-variance optimization.

References:
- López de Prado (2016). Building diversified portfolios that outperform
  out of sample (HRP). *Journal of Portfolio Management* 42.
- Maillard, Roncalli, Teiletche (2010). The properties of equally weighted
  risk contribution portfolios (ERC). *Journal of Portfolio Management* 36.
- Choueifaty, Coignard (2008). Toward maximum diversification.
  *Journal of Portfolio Management* 35.
- Black, Litterman (1992). Global portfolio optimization.
  *Financial Analysts Journal* 48.
- Kelly (1956). A new interpretation of information rate.  Thorp (2006) for
  the multi-asset continuous-time form ``w* = Sigma^{-1} mu``.
- Rockafellar, Uryasev (2000). Optimization of conditional value-at-risk.
  *Journal of Risk* 2 — CVaR linear program.
- Hallerbach (2012). A proof of the optimality of volatility weighting.
  Volatility targeting = scale to target sigma.
- Ward (1963). Hierarchical grouping (linkage used by HRP).
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray
from scipy import optimize as opt
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform

Array = NDArray[np.float64]


def _as_cov(cov: Array) -> Array:
    m = np.asarray(cov, dtype=float)
    if m.ndim != 2 or m.shape[0] != m.shape[1] or m.shape[0] < 2 or not np.all(np.isfinite(m)):
        raise ValueError("cov must be a finite square matrix (n >= 2)")
    if np.max(np.abs(m - m.T)) > 1e-6 * max(1.0, float(np.abs(m).max())):
        raise ValueError("cov must be symmetric")
    return m


def _as_returns(r: Array, n: int | None = None) -> Array:
    m = np.asarray(r, dtype=float)
    if m.ndim != 2 or not np.all(np.isfinite(m)):
        raise ValueError("returns must be a finite T x N matrix")
    if m.shape[0] < 5 or m.shape[1] < 2:
        raise ValueError("returns must have >= 5 rows and >= 2 assets")
    if n is not None and m.shape[1] != n:
        raise ValueError("returns column count must match cov")
    return m


def _as_mu(mu: Array, n: int) -> Array:
    v = np.asarray(mu, dtype=float).reshape(-1)
    if v.size != n or not np.all(np.isfinite(v)):
        raise ValueError("mu must be a finite vector matching cov")
    return v


def _cov_to_corr(cov: Array) -> Array:
    sd = np.sqrt(np.diag(cov))
    if np.any(sd <= 0.0):
        raise ValueError("cov must have positive diagonal")
    return cov / np.outer(sd, sd)


def inverse_volatility(cov: Array) -> Array:
    """Inverse-volatility weights ``w_i = (1/sigma_i) / sum_j(1/sigma_j)``."""
    m = _as_cov(cov)
    sd = np.sqrt(np.diag(m))
    if np.any(sd <= 0.0):
        raise ValueError("cov must have positive diagonal")
    iv = 1.0 / sd
    return iv / iv.sum()


def hierarchical_risk_parity(cov: Array) -> Array:
    """López de Prado (2016) HRP via Ward linkage on the correlation metric.

    Distance ``d_ij = sqrt(0.5 * (1 - rho_ij))``, quasi-diagonalization via
    the linkage leaf order, recursive bisection with inverse-variance splits.
    """
    m = _as_cov(cov)
    corr = _cov_to_corr(m)
    dist = np.sqrt(np.clip(0.5 * (1.0 - corr), 0.0, None))
    np.fill_diagonal(dist, 0.0)
    condensed = squareform(dist, checks=False)
    link = linkage(condensed, method="ward")
    order = leaves_list(link)
    w = np.ones(m.shape[0])
    var = np.diag(m)
    if np.any(var <= 0.0):
        raise ValueError("cov must have positive diagonal")

    def _cluster_var(items: NDArray[np.intp]) -> float:
        sub = m[np.ix_(items, items)]
        iv = 1.0 / np.diag(sub)
        w_iv = iv / iv.sum()
        return float(w_iv @ sub @ w_iv)

    def _bisect(items: NDArray[np.intp], weight: float) -> None:
        if items.size <= 1:
            w[items[0]] = weight
            return
        half = items.size // 2
        left, right = items[:half], items[half:]
        lv, rv = _cluster_var(left), _cluster_var(right)
        denom = lv + rv
        alpha = 1.0 - lv / denom if denom > 0.0 else 0.5
        _bisect(left, weight * alpha)
        _bisect(right, weight * (1.0 - alpha))

    _bisect(order.astype(np.intp), 1.0)
    return w / w.sum()


def equal_risk_contribution(cov: Array, x0: Array | None = None) -> Array:
    """Maillard–Roncalli–Teiletche (2010) ERC.

    Solves for weights where each asset contributes equally to portfolio
    volatility: ``w_i * (Sigma w)_i = sigma_p^2 / n``.  SLSQP fallback uses
    the sum-of-squared-deviations objective.
    """
    m = _as_cov(cov)
    n = m.shape[0]

    def _rc(w: Array) -> Array:
        sig_w = m @ w
        return w * sig_w

    def _obj(w: Array) -> float:
        rc = _rc(w)
        total = float(rc.sum())
        if total <= 0.0:
            return 1e6
        target = total / n
        return float(np.sum((rc - target) ** 2))

    start = inverse_volatility(m) if x0 is None else np.asarray(x0, dtype=float).reshape(-1)
    if start.size != n or not np.all(np.isfinite(start)) or np.any(start <= 0.0):
        raise ValueError("x0 must be a positive finite vector matching cov")
    res = opt.minimize(
        _obj,
        start,
        method="SLSQP",
        bounds=[(1e-9, None)] * n,
        constraints=[{"type": "eq", "fun": lambda w: float(w.sum() - 1.0)}],
        options={"maxiter": 500, "ftol": 1e-14},
    )
    w = np.clip(res.x, 0.0, None)
    s = w.sum()
    if s <= 0.0:
        raise ValueError("ERC optimization failed to find feasible weights")
    return w / s


def maximum_diversification(cov: Array) -> Array:
    """Choueifaty–Coignard (2008) most-diversified portfolio.

    Maximizes ``(w' sigma) / sqrt(w' Sigma w)`` subject to ``sum w = 1``,
    long-only.  Equivalent minimizer: ``min w' Sigma w s.t. w' sigma = 1``,
    rescaled — solved via SLSQP on the diversification ratio directly.
    """
    m = _as_cov(cov)
    n = m.shape[0]
    sd = np.sqrt(np.diag(m))

    def _neg_dr(w: Array) -> float:
        num = float(w @ sd)
        den = math.sqrt(max(float(w @ m @ w), 1e-18))
        return -num / den

    start = inverse_volatility(m)
    res = opt.minimize(
        _neg_dr,
        start,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * n,
        constraints=[{"type": "eq", "fun": lambda w: float(w.sum() - 1.0)}],
        options={"maxiter": 500, "ftol": 1e-12},
    )
    w = np.clip(res.x, 0.0, None)
    s = w.sum()
    if s <= 0.0:
        raise ValueError("max-diversification optimization failed")
    return w / s


def black_litterman(
    cov: Array,
    prior_mu: Array | None = None,
    views_P: Array | None = None,
    views_Q: Array | None = None,
    tau: float = 0.05,
    omega: Array | None = None,
    delta: float = 2.5,
) -> tuple[Array, Array]:
    """Black–Litterman (1992) posterior expected returns.

    ``prior_mu`` defaults to equilibrium-implied returns
    ``delta * Sigma w_mkt`` with equal weights as the market proxy.
    ``views_P`` (k x n pick matrix) and ``views_Q`` (k views) specify active
    views; ``omega`` defaults to ``tau * P Sigma P'`` diagonal.

    Returns ``(posterior_mu, posterior_cov)`` where ``posterior_cov`` is the
    *return* covariance ``Sigma + Sigma_post`` used in BL portfolio choice.
    """
    m = _as_cov(cov)
    n = m.shape[0]
    if prior_mu is None:
        w_mkt = np.full(n, 1.0 / n)
        pi = delta * (m @ w_mkt)
    else:
        pi = _as_mu(np.asarray(prior_mu, dtype=float), n)
    if not np.isfinite(tau) or tau <= 0.0:
        raise ValueError("tau must be positive and finite")
    if views_P is None and views_Q is None:
        return pi, m + tau * m
    P = np.asarray(views_P, dtype=float)
    Q = np.asarray(views_Q, dtype=float).reshape(-1)
    if P.ndim != 2 or P.shape[1] != n or not np.all(np.isfinite(P)):
        raise ValueError("views_P must be a finite k x n matrix")
    if Q.size != P.shape[0] or not np.all(np.isfinite(Q)):
        raise ValueError("views_Q must be a finite vector of length k")
    if omega is None:
        om = np.diag(np.diag(tau * P @ m @ P.T))
    else:
        om = np.asarray(omega, dtype=float)
        if om.shape != (P.shape[0], P.shape[0]) or not np.all(np.isfinite(om)):
            raise ValueError("omega must be a finite k x k matrix")
    tm = tau * m
    # Posterior mean: pi + tm P' (P tm P' + Om)^{-1} (Q - P pi)
    middle = np.linalg.pinv(P @ tm @ P.T + om)
    post_mu = pi + tm @ P.T @ middle @ (Q - P @ pi)
    # Posterior covariance of the mean estimate.
    post_var = tm - tm @ P.T @ middle @ P @ tm
    return post_mu, m + post_var


def kelly_weights(
    mu: Array,
    cov: Array,
    fraction: float = 1.0,
    *,
    long_only: bool = False,
) -> Array:
    """Kelly (1956)/Thorp continuous-time weights ``w* = f * Sigma^{-1} mu``.

    ``fraction`` = 1 is full Kelly; 0.5 is half-Kelly.  With
    ``long_only=True`` negatives are clipped and renormalized.
    """
    m = _as_cov(cov)
    mu_v = _as_mu(mu, m.shape[0])
    if not np.isfinite(fraction) or fraction <= 0.0 or fraction > 1.0:
        raise ValueError("fraction must be in (0, 1]")
    w = fraction * np.linalg.pinv(m) @ mu_v
    if long_only:
        w = np.clip(w, 0.0, None)
        s = w.sum()
        if s > 0.0:
            w = w / s
    return w


def cvar_minimization(
    returns: Array,
    alpha: float = 0.95,
    *,
    long_only: bool = True,
    target_return: float | None = None,
) -> tuple[Array, float]:
    """Rockafellar–Uryasev (2000) CVaR-minimal portfolio via LP.

    Minimizes ``VaR_alpha + (1/(1-alpha)) * mean(shortfall)`` over scenarios
    ``-r_t' w``; returns ``(weights, cvar)``.  The LP uses scenario
    auxiliary variables — exact for empirical distributions.
    """
    r = _as_returns(np.asarray(returns, dtype=float))
    t, n = r.shape
    if not np.isfinite(alpha) or not (0.5 < alpha < 1.0):
        raise ValueError("alpha must be in (0.5, 1)")
    # Variables: [w (n), zeta (1), u_t (t)]
    c = np.concatenate([np.zeros(n), [1.0], np.full(t, 1.0 / (t * (1.0 - alpha)))])
    # u_t >= -r_t'w - zeta  ->  -r_t'w - zeta - u_t <= 0
    a_ub = np.zeros((t, n + 1 + t))
    a_ub[:, :n] = -r
    a_ub[:, n] = -1.0
    a_ub[:, n + 1 :] = -np.eye(t)
    b_ub = np.zeros(t)
    bounds: list[tuple[float | None, float | None]] = [
        (0.0, 1.0) if long_only else (None, None)
    ] * n
    bounds += [(None, None)] + [(0.0, None)] * t
    a_eq = np.zeros((1, n + 1 + t))
    a_eq[0, :n] = 1.0
    b_eq = [1.0]
    if target_return is not None:
        mu = r.mean(axis=0)
        row = np.zeros(n + 1 + t)
        row[:n] = mu
        a_ub = np.vstack([a_ub, -row[np.newaxis, :]])
        b_ub = np.concatenate([b_ub, [-float(target_return)]])
    res = opt.linprog(
        c,
        A_ub=a_ub,
        b_ub=b_ub,
        A_eq=a_eq,
        b_eq=np.asarray(b_eq),
        bounds=bounds,
        method="highs",
    )
    if not res.success:
        raise ValueError(f"CVaR LP failed: {res.message}")
    w = np.clip(res.x[:n], 0.0, None) if long_only else res.x[:n]
    s = np.abs(w).sum()
    if s <= 0.0:
        raise ValueError("CVaR LP returned zero portfolio")
    w = w / w.sum()
    losses = -(r @ w)
    var_alpha = float(np.quantile(losses, alpha))
    tail = losses[losses >= var_alpha]
    cvar = float(tail.mean()) if tail.size else var_alpha
    return w, cvar


def volatility_target(
    weights: Array, cov: Array, target_vol: float, max_leverage: float = 5.0
) -> tuple[Array, float]:
    """Scale ``weights`` so portfolio vol equals ``target_vol``.

    Returns ``(scaled_weights, leverage)`` where leverage multiplies the
    input book; clipped at ``max_leverage`` (fail-closed on zero variance).
    """
    w = np.asarray(weights, dtype=float).reshape(-1)
    m = _as_cov(cov)
    if w.size != m.shape[0] or not np.all(np.isfinite(w)):
        raise ValueError("weights must be a finite vector matching cov")
    if not np.isfinite(target_vol) or target_vol <= 0.0:
        raise ValueError("target_vol must be positive and finite")
    if not np.isfinite(max_leverage) or max_leverage <= 0.0:
        raise ValueError("max_leverage must be positive and finite")
    var = float(w @ m @ w)
    if var <= 0.0:
        raise ValueError("portfolio variance must be positive")
    lev = min(target_vol / math.sqrt(var), max_leverage)
    return w * lev, lev
