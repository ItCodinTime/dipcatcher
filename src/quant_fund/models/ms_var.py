"""Markov-switching VAR(1) — Hamilton (1989) regime dynamics with EM.

``y_t = mu_{s_t} + A_{s_t} y_{t-1} + e_t``, ``e_t ~ N(0, Sigma)`` (common
covariance), ``s_t`` first-order Markov with transition matrix ``P``.
Estimation is EM: the E-step runs the Hamilton filter and Kim (1994)
smoother; the M-step is regime-probability-weighted OLS per regime plus
row-normalized smoothed transition counts for ``P``.

References: J. D. Hamilton (1989), Econometrica 57(2); C.-J. Kim (1994),
Journal of Econometrics 60(1-2).  Fail-closed on non-finite input,
T < 30, or K < 2.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


def _as_panel(y: Array) -> Array:
    m = np.asarray(y, dtype=float)
    if m.ndim == 1:
        m = m.reshape(-1, 1)
    if m.ndim != 2 or m.shape[0] < 30:
        raise ValueError("y must be a finite (t, n) matrix, t >= 30")
    if not np.all(np.isfinite(m)):
        raise ValueError("y must be finite")
    return m


def _gauss_loglik(y: Array, mu_k: Array, a_k: Array, sigma: Array) -> Array:
    """(T-1, K) Gaussian log-densities of y_t under each regime."""
    t1, n = y.shape
    k = mu_k.shape[0]
    ll = np.empty((t1 - 1, k))
    inv_sig = np.linalg.pinv(sigma)
    logdet = float(np.linalg.slogdet(sigma)[1])
    for j in range(k):
        fitted = mu_k[j] + (y[:-1] @ a_k[j].T)
        r = y[1:] - fitted
        m = np.einsum("ti,ij,tj->t", r, inv_sig, r)
        ll[:, j] = -0.5 * (n * np.log(2.0 * np.pi) + logdet + m)
    return ll


def _hamilton_filter(ll: Array, p_trans: Array) -> tuple[Array, Array, float]:
    """Forward filter. Returns (filtered probs, one-step predictions, loglik)."""
    t, k = ll.shape
    n_states = p_trans.shape[0]
    filt = np.empty((t + 1, n_states))
    pred = np.empty((t + 1, n_states))
    filt[0] = np.full(n_states, 1.0 / n_states)
    loglik = 0.0
    for i in range(t):
        pred[i + 1] = p_trans.T @ filt[i]
        like = np.exp(ll[i] - ll[i].max())
        num = pred[i + 1] * like
        tot = float(num.sum())
        if tot <= 0 or not np.isfinite(tot):
            filt[i + 1] = np.full(n_states, 1.0 / n_states)
            continue
        filt[i + 1] = num / tot
        loglik += np.log(tot) + ll[i].max()
    return filt, pred, loglik


def _kim_smooth(filt: Array, pred: Array, p_trans: Array) -> tuple[Array, Array]:
    """Kim smoother: smoothed probs and pairwise transition expectations."""
    t = filt.shape[0] - 1
    k = filt.shape[1]
    smooth = filt.copy()
    xi = np.zeros((k, k))
    for i in range(t, 0, -1):
        gain = np.divide(smooth[i], np.maximum(pred[i], 1e-300), out=np.zeros(k), where=pred[i] > 0)
        smooth[i - 1] = filt[i - 1] * (p_trans @ gain)
        s = smooth[i - 1].sum()
        if s > 0:
            smooth[i - 1] /= s
        xi += np.outer(filt[i - 1], gain) * p_trans
    return smooth, xi


def _weighted_ols(y: Array, ylag: Array, w: Array) -> tuple[Array, Array]:
    """Weighted VAR(1): returns (mu, A) with y_t = mu + A y_{t-1}."""
    n = y.shape[1]
    sw = np.sqrt(np.maximum(w, 0.0))
    xd = np.hstack([np.ones((y.shape[0], 1)), ylag]) * sw[:, None]
    b = np.linalg.lstsq(xd, y * sw[:, None], rcond=None)[0]
    return b[0], b[1:].T.reshape(n, n)


def msvar_fit(
    y: Array,
    k: int = 2,
    *,
    max_iter: int = 100,
    tol: float = 1e-6,
    seed: int = 0,
) -> dict[str, Array]:
    """EM fit of the K-regime Markov-switching VAR(1).

    Initialization: VAR(1) residuals split by squared-norm quantiles into
    K ordered regimes (calm -> turbulent).  Returns per-regime intercepts
    and coefficient matrices, the pooled covariance, the estimated
    transition matrix, filtered/smoothed probabilities, and loglik.
    """
    m = _as_panel(y)
    t, n = m.shape
    if not 2 <= k <= 4:
        raise ValueError("k must be in [2, 4]")
    if max_iter < 1 or tol <= 0:
        raise ValueError("max_iter/tol must be positive")
    # init: VAR(1) residual norm quantiles
    xd = np.hstack([np.ones((t - 1, 1)), m[:-1]])
    b0 = np.linalg.lstsq(xd, m[1:], rcond=None)[0]
    r0 = m[1:] - xd @ b0
    norms = np.sum(r0**2, axis=1)
    qs = np.quantile(norms, np.linspace(0, 1, k + 1))
    assign = np.clip(np.digitize(norms, qs[1:-1]), 0, k - 1)
    mu = np.empty((k, n))
    a = np.empty((k, n, n))
    for j in range(k):
        idx = np.where(assign == j)[0]
        if idx.size < n + 1:
            idx = np.where(assign == min(assign.max(), j))[0]
        mu[j], a[j] = _weighted_ols(
            m[1:][idx] if idx.size else m[1:],
            m[:-1][idx],
            np.ones(idx.size) if idx.size else np.ones(t - 1),
        )
    sigma = r0.T @ r0 / (t - 1) + np.eye(n) * 1e-8
    p_trans = np.full((k, k), 0.1 / (k - 1))
    np.fill_diagonal(p_trans, 0.9)
    prev_ll = -np.inf
    filt = np.full((t, k), 1.0 / k)
    smooth = filt.copy()
    for _ in range(max_iter):
        ll = _gauss_loglik(m, mu, a, sigma)
        filt, pred, loglik = _hamilton_filter(ll, p_trans)
        smooth, xi = _kim_smooth(filt, pred, p_trans)
        w = smooth[1:]  # (T-1, k)
        for j in range(k):
            mu[j], a[j] = _weighted_ols(m[1:], m[:-1], w[:, j])
        sigma = np.zeros((n, n))
        for j in range(k):
            r = m[1:] - (mu[j] + m[:-1] @ a[j].T)
            sigma += (r * w[:, j : j + 1]).T @ r
        sigma /= max(t - 1, 1)
        sigma += np.eye(n) * 1e-8
        if xi.sum() > 0:
            p_new = xi / np.maximum(xi.sum(axis=1, keepdims=True), 1e-12)
            p_trans = np.clip(p_new, 1e-6, 1.0)
            p_trans /= p_trans.sum(axis=1, keepdims=True)
        if abs(loglik - prev_ll) < tol * max(1.0, abs(loglik)):
            break
        prev_ll = loglik
    return {
        "mu": mu,
        "A": a,
        "sigma": sigma,
        "P": p_trans,
        "filtered": filt,
        "smoothed": smooth,
        "loglik": np.array([float(prev_ll)]),
        "k": np.array([k]),
    }


def msvar_forecast(fit: dict[str, Array], y_last: Array, steps: int = 8) -> dict[str, Array]:
    """Probability-weighted h-step forecast from the filtered state probs.

    Propagates the regime belief with P^h and applies each regime's VAR
    recursion to the last observation.
    """
    if steps < 1:
        raise ValueError("steps must be >= 1")
    mu = np.asarray(fit["mu"], dtype=float)
    a = np.asarray(fit["A"], dtype=float)
    p_trans = np.asarray(fit["P"], dtype=float)
    probs = np.asarray(fit["filtered"], dtype=float)[-1]
    yv = np.asarray(y_last, dtype=float).ravel()
    k, n = mu.shape
    if yv.size != n:
        raise ValueError("y_last must match the panel width")
    # regime-conditional paths share the same start; beliefs evolve via P
    out = np.empty((steps, n))
    yreg = np.tile(yv, (k, 1))
    bel = probs.copy()
    for i in range(steps):
        bel = p_trans.T @ bel
        for j in range(k):
            yreg[j] = mu[j] + a[j] @ yreg[j]
        out[i] = bel @ yreg
    return {"y_forecast": out, "regime_probs_final": bel}
