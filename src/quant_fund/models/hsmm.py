"""Explicit-duration hidden semi-Markov model (Gaussian emissions).

Unlike an HMM (geometric durations by construction), an HSMM puts an
arbitrary pmf on run lengths — appropriate for regime-like processes whose
dwell times are far from geometric (Ferguson 1980; Yu & Kobayashi 2003).

The model is handled by expansion to the extended state space (state,
remaining duration) with transitions

    (i, 1) -> (j, d)   w.p.  A[i, j] * dur_j(d)
    (i, d) -> (i, d-1) w.p.  1            for d > 1

so standard forward filtering / Viterbi apply on K * D states.

- ``hsmm_nll``/``hsmm_filter``: forward recursion, returns log-likelihood
  and filtered posteriors collapsed to the K state marginals.
- ``hsmm_viterbi``: MAP (state, duration) path on the extended space.
- ``hsmm_fit``: EM-lite — initialized by y quantiles; each iteration runs
  the forward pass and re-fits emissions on filtered posteriors; durations
  and the transition matrix stay at their (uniform / sticky) init. Honest
  scope: the E-step is filtered (not smoothed), documented as such.

Fail-closed: shape/non-finite checks, K >= 1, 1 <= d_max, positive sigma.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.special import logsumexp

Array = NDArray[np.float64]


def _check_hsmm(
    y: Array,
    mu: Array,
    sigma: Array,
    trans: Array,
    dur_probs: Array,
) -> tuple[Array, Array, Array, Array, Array, int, int]:
    yy = np.asarray(y, dtype=float).ravel()
    mu = np.asarray(mu, dtype=float).ravel()
    sigma = np.asarray(sigma, dtype=float).ravel()
    trans = np.asarray(trans, dtype=float)
    dur_probs = np.asarray(dur_probs, dtype=float)
    k = mu.size
    if yy.size < 4 or not np.isfinite(yy).all():
        raise ValueError("y must be a finite array with T >= 4")
    if sigma.size != k or (sigma <= 0).any() or not np.isfinite(sigma).all():
        raise ValueError("sigma must be positive, length K")
    if trans.shape != (k, k) or (trans < 0).any():
        raise ValueError("trans must be (K, K) non-negative")
    if np.abs(trans.sum(axis=1) - 1.0).max() > 1e-6:
        raise ValueError("trans rows must sum to 1")
    if dur_probs.ndim != 2 or dur_probs.shape[0] != k or (dur_probs < 0).any():
        raise ValueError("dur_probs must be (K, D) non-negative")
    if np.abs(dur_probs.sum(axis=1) - 1.0).max() > 1e-6:
        raise ValueError("dur_probs rows must sum to 1")
    return yy, mu, sigma, trans, dur_probs, k, dur_probs.shape[1]


def _emit(y: Array, mu: Array, sigma: Array) -> Array:
    """Log N(y | mu_k, sigma_k^2) matrix (T, K)."""
    z = (y[:, None] - mu[None, :]) / sigma[None, :]
    return np.asarray(-0.5 * z * z - np.log(sigma)[None, :] - 0.5 * np.log(2.0 * np.pi))


def _forward(log_emit: Array, trans: Array, dur_probs: Array, pi: Array) -> tuple[float, Array]:
    """Forward pass on the extended (state, duration) space.

    ``alpha[t, i*D + d]`` = log P(y_1..t, in state i with d remaining obs
    *including* t). Returns (loglik, alpha).
    """
    t_len, k = log_emit.shape
    d_max = dur_probs.shape[1]
    alpha = np.full((t_len, k * d_max), -np.inf)
    alpha[0] = np.log(np.clip(pi[:, None] * dur_probs, 1e-300, None)).ravel() + np.repeat(
        log_emit[0], d_max
    )
    # exit mass of each state at t: sum over d of alpha weighted... entries
    # at (i, 1) leave; entry to (j, d) gets A[i,j]*dur_j(d)*exit_i
    for t in range(1, t_len):
        exit_log = np.array([alpha[t - 1, i * d_max] for i in range(k)])  # (i, d=0 idx)
        enter_j_d = logsumexp(
            exit_log[:, None, None] + np.log(np.clip(trans, 1e-300, None))[:, :, None], axis=0
        ) + np.log(np.clip(dur_probs, 1e-300, None))  # (j, d)
        prev = alpha[t - 1].reshape(k, d_max)
        cont = np.full((k, d_max), -np.inf)
        cont[:, : d_max - 1] = prev[:, 1:]  # (i, d-1) <- (i, d)
        alpha[t] = np.logaddexp(enter_j_d, cont).ravel() + np.repeat(log_emit[t], d_max)
    return float(logsumexp(alpha[-1])), alpha


def hsmm_filter(
    y: Array,
    mu: Array,
    sigma: Array,
    trans: Array,
    dur_probs: Array,
    pi: Array | None = None,
) -> dict[str, Array | float]:
    """Filtered state posteriors + log predictive likelihood."""
    yy, mu, sigma, trans, dur_probs, k, d_max = _check_hsmm(y, mu, sigma, trans, dur_probs)
    if pi is None:
        pi = np.full(k, 1.0 / k)
    log_emit = _emit(yy, mu, sigma)
    loglik, alpha = _forward(log_emit, trans, dur_probs, np.asarray(pi, dtype=float))
    post = np.exp(alpha.reshape(yy.size, k, d_max) - logsumexp(alpha, axis=1)[:, None, None])
    return {
        "loglik": loglik,
        "state_prob": post.sum(axis=2),
        "state_dur_prob": post,
        "alpha": alpha,
    }


def hsmm_viterbi(
    y: Array,
    mu: Array,
    sigma: Array,
    trans: Array,
    dur_probs: Array,
    pi: Array | None = None,
) -> dict[str, NDArray[np.int64]]:
    """MAP path over the extended (state, remaining-duration) space."""
    yy, mu, sigma, trans, dur_probs, k, d_max = _check_hsmm(y, mu, sigma, trans, dur_probs)
    if pi is None:
        pi = np.full(k, 1.0 / k)
    log_emit = _emit(yy, mu, sigma)
    t_len = yy.size
    delta = np.full((t_len, k * d_max), -np.inf)
    psi = np.zeros((t_len, k * d_max), dtype=np.int64)
    delta[0] = np.log(
        np.clip(np.asarray(pi)[:, None] * dur_probs, 1e-300, None)
    ).ravel() + np.repeat(log_emit[0], d_max)
    for t in range(1, t_len):
        prev = delta[t - 1].reshape(k, d_max)
        exit_log = prev[:, 0]
        enter = (
            exit_log[:, None, None]
            + np.log(np.clip(trans, 1e-300, None))[:, :, None]
            + np.log(np.clip(dur_probs, 1e-300, None))[None, :, :]
        ).max(axis=0)  # (j, d): best predecessor exiting
        cont = np.full((k, d_max), -np.inf)
        cont[:, : d_max - 1] = prev[:, 1:]
        cand = np.stack([enter, cont], axis=0)
        choose = cand.argmax(axis=0)
        best = cand[choose, np.arange(k)[:, None], np.arange(d_max)[None, :]]
        delta[t] = best.ravel() + np.repeat(log_emit[t], d_max)
        # psi: 0 = entered (predecessor unknown-collapsed), 1 = continued
        psi[t] = choose.ravel()
    states = np.zeros(t_len, dtype=np.int64)
    idx = int(np.argmax(delta[-1]))
    for t in range(t_len - 1, -1, -1):
        states[t] = idx // d_max
        d = idx % d_max
        if t > 0:
            if psi[t, idx] == 1 and d + 1 < d_max:
                idx = idx + 1  # continued from (same state, d+1)
            else:
                # re-entered: argmax over exiting predecessors is d-free, so
                # this backtrack reproduces the max-product predecessor exactly
                prev = delta[t - 1].reshape(k, d_max)
                j_score = int(
                    (prev[:, 0] + np.log(np.clip(trans, 1e-300, None))[:, states[t]]).argmax()
                )
                idx = j_score * d_max  # predecessor at (j_score, d=0)
    return {"states": states}


def hsmm_fit(
    y: Array,
    n_states: int,
    d_max: int = 20,
    n_iter: int = 25,
) -> dict[str, Array | float]:
    """EM-lite HSMM fit: quantile init, filtered-posterior emission updates."""
    yy = np.asarray(y, dtype=float).ravel()
    if yy.size < 8 or not np.isfinite(yy).all():
        raise ValueError("y must be a finite array with T >= 8")
    if n_states < 1 or d_max < 2:
        raise ValueError("need n_states >= 1 and d_max >= 2")
    k, d = n_states, d_max
    qs = np.quantile(yy, np.linspace(0.1, 0.9, k + 2)[1:-1] if k > 1 else [0.5])
    mu = np.atleast_1d(qs)
    sigma = np.full(k, float(yy.std()) + 1e-6)
    trans = np.full((k, k), 0.05 / max(k - 1, 1))
    np.fill_diagonal(trans, 0.95)
    trans /= trans.sum(axis=1, keepdims=True)
    dur_probs = np.full((k, d), 1.0 / d)
    pi = np.full(k, 1.0 / k)
    loglik = -np.inf
    for _ in range(n_iter):
        out = hsmm_filter(yy, mu, sigma, trans, dur_probs, pi)
        loglik = float(out["loglik"])
        post = np.asarray(out["state_prob"], dtype=float)
        mu = (post * yy[:, None]).sum(axis=0) / post.sum(axis=0).clip(1e-300)
        var = (post * (yy[:, None] - mu[None, :]) ** 2).sum(axis=0) / post.sum(axis=0).clip(1e-300)
        sigma = np.sqrt(np.clip(var, 1e-8, None))
    return {
        "mu": mu,
        "sigma": sigma,
        "trans": trans,
        "dur_probs": dur_probs,
        "pi": pi,
        "loglik": loglik,
    }
