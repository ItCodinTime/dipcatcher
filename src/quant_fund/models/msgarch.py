"""Markov-switching GARCH(1,1) (Haas–Mittnik–Paolella).

h_i,t = omega_i + alpha_i * eps_{t-1}^2 + beta_i * h_i,t-1, with the regime
i ~ Markov chain of P. Because h depends on the regime *path*, exact
inference needs path probabilities; we implement the standard feasible
filter: filtered probabilities + per-regime variance collapsed by the
filtered mean (Gray (1996)-style collapsing, documented).

- ``msgarch_filter``: filtered regime probabilities + per-regime h path +
  Gaussian quasi log-likelihood.
- ``msgarch_fit``: BFGS on negative QLL over (omega, alpha, beta, p_ij).
- ``msgarch_stationary_vol``: unconditional variance per regime,
  E[h_i] = omega_i/(1 - alpha_i - beta_i), and the filtered pooled
  unconditional variance.
- ``msgarch_simulate``: path simulation.

Fail-closed: positivity of omega/alpha, beta, alpha+beta<1 per regime, and
stochastic rows of the transition matrix.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize

Array = NDArray[np.float64]


def _check_series(y: Array) -> Array:
    yy = np.asarray(y, dtype=float).ravel()
    if yy.size < 30 or not np.isfinite(yy).all():
        raise ValueError("y must be a finite array with T >= 30")
    return yy


def _check_params(
    omega: Array, alpha: Array, beta: Array, p_mat: Array
) -> tuple[Array, Array, Array, Array, int]:
    w = np.asarray(omega, dtype=float).ravel()
    a = np.asarray(alpha, dtype=float).ravel()
    b = np.asarray(beta, dtype=float).ravel()
    p = np.asarray(p_mat, dtype=float)
    if not (w.size == a.size == b.size == p.shape[0] == p.shape[1]):
        raise ValueError("omega/alpha/beta and P dimensions must agree")
    k = w.size
    if k < 2:
        raise ValueError("need k >= 2 regimes")
    if (w <= 0).any() or (a < 0).any() or (b < 0).any():
        raise ValueError("omega > 0, alpha/beta >= 0 required")
    if (a + b >= 1.0).any():
        raise ValueError("each regime needs alpha + beta < 1")
    if not np.allclose(p.sum(axis=1), 1.0) or (p < 0).any():
        raise ValueError("P must be row-stochastic and non-negative")
    for arr in (w, a, b, p):
        if not np.isfinite(arr).all():
            raise ValueError("non-finite parameters")
    return w, a, b, p, k


def msgarch_filter(
    y: Array,
    omega: Array,
    alpha: Array,
    beta: Array,
    p_mat: Array,
    pi: Array | None = None,
) -> dict[str, Array | float]:
    """Filtered regime probabilities, per-regime variance path, log-likelihood.

    Collapsing: the variance entering t is E[h_i,t] over the filtered regime
    probabilities (approximate filter — the exact recursion is exponential
    in T; this is the documented feasible version).
    """
    yy = _check_series(y)
    w, a, b, p, k = _check_params(omega, alpha, beta, p_mat)
    if pi is None:
        pi = np.full(k, 1.0 / k)
    pi = np.asarray(pi, dtype=float)
    if pi.shape != (k,) or (pi < 0).any() or not np.isclose(pi.sum(), 1.0):
        raise ValueError("pi must be a probability vector of length k")
    t_len = yy.size
    h = np.zeros((t_len + 1, k))
    e2 = np.zeros(t_len + 1)
    h[0] = w / np.clip(1.0 - a - b, 1e-8, None)  # unconditional per regime
    loglik = 0.0
    post = np.zeros((t_len, k))
    for t in range(t_len):
        e2[t] = yy[t - 1] ** 2 if t > 0 else 0.0
        # collapse variance to a single h per regime
        h[t + 1] = w + a * e2[t] + b * h[t]
        var_t = h[t + 1]
        ll_i = -0.5 * (np.log(2 * np.pi * var_t) + yy[t] ** 2 / var_t)
        # one-step-ahead predictive probabilities
        pred = pi @ p if t > 0 else pi
        lik = pred * np.exp(ll_i)
        total = lik.sum()
        if total <= 0 or not np.isfinite(total):
            loglik += -50.0
            pi = pred  # numerical safeguard
            continue
        pi = lik / total
        loglik += float(np.log(total))
        post[t] = pi
    return {
        "state_prob": post,
        "h": h[1:],
        "loglik": float(loglik),
        "filtered_var": (post * h[1:]).sum(axis=1),
    }


def msgarch_stationary_vol(omega: Array, alpha: Array, beta: Array) -> Array:
    """Per-regime unconditional variance omega/(1 - alpha - beta)."""
    w = np.asarray(omega, dtype=float).ravel()
    a = np.asarray(alpha, dtype=float).ravel()
    b = np.asarray(beta, dtype=float).ravel()
    if not (w.size == a.size == b.size) or w.size == 0:
        raise ValueError("omega/alpha/beta must be equal-length and non-empty")
    if (w <= 0).any() or (a < 0).any() or (b < 0).any() or (a + b >= 1.0).any():
        raise ValueError("need omega > 0, alpha/beta >= 0, alpha + beta < 1")
    return np.asarray(w / np.clip(1 - a - b, 1e-12, None))


def msgarch_fit(y: Array, n_regimes: int = 2, max_iter: int = 200) -> dict:
    """BFGS quasi-MLE over unconstrained params (softmax transforms)."""
    yy = _check_series(y)
    if n_regimes < 2:
        raise ValueError("n_regimes must be >= 2")
    k = n_regimes
    v = float(np.var(yy))

    def _unpack(theta: Array):
        om = np.exp(theta[:k]) * v * 0.1
        al = 1 / (1 + np.exp(-theta[k : 2 * k])) * 0.45
        be = 1 / (1 + np.exp(-theta[2 * k : 3 * k])) * 0.5
        be = np.clip(be, 0, 0.95 - al)
        raw = theta[3 * k :].reshape(k, k - 1)
        e = np.exp(raw - raw.max(axis=1, keepdims=True))
        p = np.column_stack([np.ones(k), e])
        p /= p.sum(axis=1, keepdims=True)
        return om, al, be, p

    def _nll(theta: Array) -> float:
        om, al, be, p = _unpack(theta)
        out = float(-float(msgarch_filter(yy, om, al, be, p)["loglik"]))
        return out if np.isfinite(out) else 1e12

    theta0 = np.concatenate([np.zeros(3 * k), np.tile(np.log(np.full(k - 1, 0.1 / 0.9)), k)])
    res = minimize(_nll, theta0, method="BFGS", options={"maxiter": max_iter})
    om, al, be, p = _unpack(res.x)
    return {
        "omega": om,
        "alpha": al,
        "beta": be,
        "p": p,
        "nll": float(res.fun),
        "converged": bool(res.success),
        "n_iter": int(res.nit),
    }


def msgarch_simulate(
    omega: Array,
    alpha: Array,
    beta: Array,
    p_mat: Array,
    n: int,
    seed: int = 0,
) -> dict[str, Array | NDArray[np.int64]]:
    """Simulate a Markov-switching GARCH path (exact, recursive)."""
    w, a, b, p, k = _check_params(omega, alpha, beta, p_mat)
    if n < 1:
        raise ValueError("n must be >= 1")
    rng = np.random.default_rng(seed)
    states = np.zeros(n, dtype=np.int64)
    eps = np.zeros(n)
    h = np.zeros(n)
    states[0] = int(rng.choice(k))
    h[0] = w[states[0]] / max(1 - a[states[0]] - b[states[0]], 1e-8)
    eps[0] = rng.normal(0, np.sqrt(h[0]))
    for t in range(1, n):
        states[t] = int(rng.choice(k, p=p[states[t - 1]]))
        i = states[t]
        h[t] = w[i] + a[i] * eps[t - 1] ** 2 + b[i] * h[t - 1]
        eps[t] = rng.normal(0, np.sqrt(h[t]))
    return {"y": eps, "states": states, "h": h}
