"""Input-output HMM: covariate-driven regime transitions.

Bengio & Frasconi (1995): the transition matrix is a per-step function of
exogenous covariates, P_t[i, j] = softmax_j(W_i . x_t), while emissions are
Gaussian N(mu_i, sigma_i^2). Fitting is exact-MLE by BFGS on the forward-
recursion negative log-likelihood (gradients via finite difference through
``scipy.optimize.minimize``).

- ``io_hmm_transitions``: build the (T-1, K, K) tensor of P_t.
- ``io_hmm_nll``: forward-filtered negative log-likelihood given parameters.
- ``io_hmm_filter``: filtered state posteriors for fitted parameters.
- ``io_hmm_fit``: quantile-init + BFGS maximum likelihood.

Fail-closed: shape/finite checks, K >= 2, positive sigma.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize
from scipy.special import logsumexp

Array = NDArray[np.float64]


def _check_xy(y: Array, x: Array) -> tuple[Array, Array]:
    yy = np.asarray(y, dtype=float).ravel()
    xx = np.asarray(x, dtype=float)
    if xx.ndim == 1:
        xx = xx[:, None]
    if yy.size < 8 or not np.isfinite(yy).all():
        raise ValueError("y must be a finite array with T >= 8")
    if xx.ndim != 2 or xx.shape[0] != yy.size or not np.isfinite(xx).all():
        raise ValueError("x must be a finite (T, p) array matching y")
    return yy, xx


def io_hmm_transitions(x: Array, w: Array) -> Array:
    """P_t[i, j] = softmax_j(x_t . w_i); returns (T, K, K)."""
    xx = np.asarray(x, dtype=float)
    if xx.ndim == 1:
        xx = xx[:, None]
    w = np.asarray(w, dtype=float)
    if w.ndim != 3 or w.shape[0] != w.shape[1] or w.shape[2] != xx.shape[1]:
        raise ValueError("w must be (K, K, p)")
    if not np.isfinite(w).all():
        raise ValueError("non-finite w")
    # W[i, j] is the logit row for exiting state i into j: logits[t, i, j]
    logits = np.einsum("ijp,tp->tij", w, xx)
    logits -= logits.max(axis=2, keepdims=True)
    e = np.exp(logits)
    return np.asarray(e / e.sum(axis=2, keepdims=True))


def _forward_ll(log_emit: Array, p_t: Array, pi: Array) -> tuple[float, Array]:
    """Standard HMM forward pass with per-step transition matrices."""
    t_len, k = log_emit.shape
    alpha = np.full((t_len, k), -np.inf)
    alpha[0] = np.log(np.clip(pi, 1e-300, None)) + log_emit[0]
    for t in range(1, t_len):
        alpha[t] = log_emit[t] + logsumexp(
            alpha[t - 1][:, None] + np.log(np.clip(p_t[t - 1], 1e-300, None)), axis=0
        )
    return float(logsumexp(alpha[-1])), alpha


def io_hmm_nll(
    y: Array,
    x: Array,
    w: Array,
    mu: Array,
    sigma: Array,
    pi: Array | None = None,
) -> float:
    """Negative log-likelihood of (y | x) under the IO-HMM."""
    yy, xx = _check_xy(y, x)
    mu = np.asarray(mu, dtype=float).ravel()
    sigma = np.asarray(sigma, dtype=float).ravel()
    w = np.asarray(w, dtype=float)
    k = mu.size
    if sigma.size != k or (sigma <= 0).any() or not np.isfinite(sigma).all():
        raise ValueError("sigma must be positive, length K")
    if w.ndim != 3 or w.shape[1] != k or w.shape[2] != xx.shape[1]:
        raise ValueError("w must be (K, K, p)")
    if pi is None:
        pi = np.full(k, 1.0 / k)
    log_emit = -0.5 * ((yy[:, None] - mu[None, :]) / sigma[None, :]) ** 2 - np.log(sigma)[None, :]
    p_t = io_hmm_transitions(xx, w)
    ll, _ = _forward_ll(log_emit, p_t, np.asarray(pi, dtype=float))
    return -ll


def io_hmm_filter(
    y: Array,
    x: Array,
    w: Array,
    mu: Array,
    sigma: Array,
    pi: Array | None = None,
) -> dict[str, Array | float]:
    """Filtered state posteriors under fitted parameters."""
    yy, xx = _check_xy(y, x)
    mu = np.asarray(mu, dtype=float).ravel()
    sigma = np.asarray(sigma, dtype=float).ravel()
    k = mu.size
    if pi is None:
        pi = np.full(k, 1.0 / k)
    log_emit = -0.5 * ((yy[:, None] - mu[None, :]) / sigma[None, :]) ** 2 - np.log(sigma)[None, :]
    p_t = io_hmm_transitions(xx, np.asarray(w, dtype=float))
    ll, alpha = _forward_ll(log_emit, p_t, np.asarray(pi, dtype=float))
    post = np.exp(alpha - logsumexp(alpha, axis=1, keepdims=True))
    return {"loglik": ll, "state_prob": post}


def io_hmm_fit(
    y: Array,
    x: Array,
    n_states: int = 2,
    max_iter: int = 300,
) -> dict[str, Array | float]:
    """Quantile-init BFGS MLE for the IO-HMM."""
    yy, xx = _check_xy(y, x)
    if n_states < 2:
        raise ValueError("n_states must be >= 2")
    k, p = n_states, xx.shape[1]
    qs = np.quantile(yy, np.linspace(0.2, 0.8, k))
    mu0 = qs
    log_sigma0 = np.log(np.full(k, float(yy.std()) + 1e-6))
    w0 = np.zeros((k, k, p))
    for i in range(k):
        w0[i, i, 0] = 2.0  # sticky self-transition bias
    pi0 = np.full(k, 1.0 / k)

    def _nll(theta: Array) -> float:
        w = theta[: k * k * p].reshape(k, k, p)
        mu = theta[k * k * p : k * k * p + k]
        sigma = np.exp(theta[k * k * p + k :])
        val = io_hmm_nll(yy, xx, w, mu, sigma, pi0)
        return val if np.isfinite(val) else 1e12

    theta0 = np.concatenate([w0.ravel(), mu0, log_sigma0])
    res = minimize(_nll, theta0, method="BFGS", options={"maxiter": max_iter})
    th = res.x
    w = th[: k * k * p].reshape(k, k, p)
    mu = th[k * k * p : k * k * p + k]
    sigma = np.exp(th[k * k * p + k :])
    return {
        "w": w,
        "mu": mu,
        "sigma": sigma,
        "pi": pi0,
        "nll": float(res.fun),
        "converged": bool(res.success),
        "n_iter": int(res.nit),
    }
