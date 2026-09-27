"""Markov-switching regressions.

Hamilton's (1989) two/more-state Markov-switching model: EM via the
Hamilton filter, Kim (1994) smoothing, expected regime durations, and the
AR(0)-intercept and dynamic-regression forms.

References:
- Hamilton (1989) Markov-switching model / Hamilton filter.
- Hamilton (1994) textbook EM algorithm (ch. 22).
- Kim (1994) backward smoothing (Kim smoother).
- Krolzig (1997) MS-VAR estimation.
- Ang, Bekaert (2002) regime-switching asset allocation context.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


def _v(x: Array, n: int = 20) -> Array:
    v = np.asarray(x, dtype=float).reshape(-1)
    if v.size < n or not np.all(np.isfinite(v)):
        raise ValueError(f"series must be finite with length >= {n}")
    return v


def _norm_pdf(x: Array, mu: Array, sigma: Array) -> Array:
    s = np.maximum(np.asarray(sigma, dtype=float), 1e-12)
    z = (x - mu) / s
    return np.asarray(np.exp(-0.5 * z * z) / (math.sqrt(2 * math.pi) * s), dtype=float)


def fit_markov_switching_mean(
    y: Array,
    n_states: int = 2,
    max_iter: int = 200,
    tol: float = 1e-6,
    seed: int = 0,
) -> dict[str, Array]:
    """Hamilton (1989) MS model with state-dependent mean and variance.

    ``y_t ~ N(mu_{s_t}, sigma_{s_t}^2)``, ``P`` constant transition matrix.
    EM via the Hamilton filter + M-step on smoothed probabilities.
    """
    v = _v(y)
    t = v.size
    if n_states < 2 or n_states > 6:
        raise ValueError("n_states must be in [2, 6]")
    rng = np.random.default_rng(seed)
    # Init: k-quantile means.
    qs = np.quantile(v, np.linspace(0.05, 0.95, n_states))
    mus = qs + rng.normal(scale=0.05, size=n_states)
    sigs = np.full(n_states, float(v.std()))
    P = np.full((n_states, n_states), 0.05 / (n_states - 1))
    np.fill_diagonal(P, 0.95)
    xi = np.full(n_states, 1.0 / n_states)
    prev_ll = -np.inf
    filt_hist = np.zeros((t, n_states))
    for _ in range(max_iter):
        # Hamilton filter.
        loglik = 0.0
        for s in range(t):
            dens = _norm_pdf(np.full(n_states, v[s]), mus, sigs)
            pred = P.T @ xi
            lik = float(np.sum(dens * pred))
            if lik <= 0:
                lik = 1e-300
            xi = dens * pred / lik
            filt_hist[s] = xi
            loglik += math.log(lik)
        # Kim smoother.
        smooth = np.zeros_like(filt_hist)
        smooth[-1] = filt_hist[-1]
        for s in range(t - 2, -1, -1):
            pred = P.T @ filt_hist[s]
            ratio = np.divide(smooth[s + 1], np.maximum(pred, 1e-300))
            smooth[s] = filt_hist[s] * (P @ ratio)
            smooth[s] /= max(smooth[s].sum(), 1e-300)
        # M-step.
        for j in range(n_states):
            w = smooth[:, j]
            wsum = float(w.sum())
            if wsum <= 1e-8:
                continue
            mus[j] = float(w @ v) / wsum
            sigs[j] = math.sqrt(max(float(w @ (v - mus[j]) ** 2) / wsum, 1e-10))
        # Transition update via filtered-implied joint probabilities.
        for i in range(n_states):
            for j in range(n_states):
                joint = 0.0
                for s in range(t - 1):
                    joint += (
                        filt_hist[s, i]
                        * P[i, j]
                        * _norm_pdf_scalar(v[s + 1], mus[j], sigs[j])
                        / max(
                            float(
                                np.sum(
                                    _norm_pdf(np.full(n_states, v[s + 1]), mus, sigs)
                                    * (P.T @ filt_hist[s])
                                )
                            ),
                            1e-300,
                        )
                    )
                P[i, j] = joint / max(float(filt_hist[:-1, i].sum()), 1e-12)
            P[i] /= max(P[i].sum(), 1e-12)
        if loglik - prev_ll < tol * (1.0 + abs(prev_ll)):
            break
        prev_ll = loglik
    durations = 1.0 / np.maximum(1.0 - np.diag(P), 1e-8)
    return {
        "means": mus,
        "sigmas": sigs,
        "P": P,
        "filtered": filt_hist,
        "smoothed": smooth,
        "loglik": np.array([loglik]),
        "durations": durations,
        "n_iter": np.array([float(_ + 1)]),
    }


def _norm_pdf_scalar(x: float, mu: float, sigma: float) -> float:
    z = (x - mu) / max(sigma, 1e-12)
    return math.exp(-0.5 * z * z) / (math.sqrt(2 * math.pi) * max(sigma, 1e-12))


def fit_markov_switching_regression(
    y: Array,
    x: Array,
    n_states: int = 2,
    max_iter: int = 200,
    tol: float = 1e-6,
    seed: int = 0,
) -> dict[str, Array]:
    """Markov-switching regression: ``y_t = x_t' beta_{s_t} + sigma_{s_t} e_t``.

    Regressors include the constant if desired (pass a ones column).
    EM: E-step = Hamilton filter + Kim smoother; M-step = weighted least
    squares per state.
    """
    v = _v(y)
    m = np.asarray(x, dtype=float)
    if m.ndim == 1:
        m = m.reshape(-1, 1)
    if m.shape[0] != v.size or not np.all(np.isfinite(m)):
        raise ValueError("x must be a finite n x k matrix matching y")
    t, k = m.shape
    if n_states < 2 or n_states > 4:
        raise ValueError("n_states must be in [2, 4]")
    qs = np.quantile(v, np.linspace(0.1, 0.9, n_states))
    betas = np.zeros((n_states, k))
    betas[:, 0] = qs
    sigs = np.full(n_states, float(v.std()))
    P = np.full((n_states, n_states), 0.05 / (n_states - 1))
    np.fill_diagonal(P, 0.95)
    xi = np.full(n_states, 1.0 / n_states)
    prev_ll = -np.inf
    filt_hist = np.zeros((t, n_states))
    smooth = np.zeros_like(filt_hist)
    for _ in range(max_iter):
        loglik = 0.0
        for s in range(t):
            dens = np.array(
                [_norm_pdf_scalar(v[s], float(m[s] @ betas[j]), sigs[j]) for j in range(n_states)]
            )
            pred = P.T @ xi
            lik = float(np.sum(dens * pred))
            if lik <= 0:
                lik = 1e-300
            xi = dens * pred / lik
            filt_hist[s] = xi
            loglik += math.log(lik)
        smooth[-1] = filt_hist[-1]
        for s in range(t - 2, -1, -1):
            pred = P.T @ filt_hist[s]
            ratio = np.divide(smooth[s + 1], np.maximum(pred, 1e-300))
            smooth[s] = filt_hist[s] * (P @ ratio)
            smooth[s] /= max(smooth[s].sum(), 1e-300)
        # M-step: WLS per state.
        for j in range(n_states):
            w = np.sqrt(smooth[:, j] + 1e-12)
            xw = m * w[:, None]
            yw = v * w
            if float((smooth[:, j]).sum()) <= k + 1:
                continue
            betas[j], *_ = np.linalg.lstsq(xw, yw, rcond=None)
            res = v - m @ betas[j]
            sigs[j] = math.sqrt(
                max(float(smooth[:, j] @ res**2) / float(smooth[:, j].sum()), 1e-10)
            )
        # Transition update (numerically stable filtered form).
        for i in range(n_states):
            for j in range(n_states):
                num = 0.0
                for s in range(t - 1):
                    dens_next = np.array(
                        [
                            _norm_pdf_scalar(v[s + 1], float(m[s + 1] @ betas[j2]), sigs[j2])
                            for j2 in range(n_states)
                        ]
                    )
                    lik_next = float(np.sum(dens_next * (P.T @ filt_hist[s])))
                    num += filt_hist[s, i] * P[i, j] * dens_next[j] / max(lik_next, 1e-300)
                P[i, j] = num / max(float(filt_hist[:-1, i].sum()), 1e-12)
            P[i] /= max(P[i].sum(), 1e-12)
        if loglik - prev_ll < tol * (1.0 + abs(prev_ll)):
            break
        prev_ll = loglik
    durations = 1.0 / np.maximum(1.0 - np.diag(P), 1e-8)
    return {
        "betas": betas,
        "sigmas": sigs,
        "P": P,
        "filtered": filt_hist,
        "smoothed": smooth,
        "loglik": np.array([loglik]),
        "durations": durations,
    }


def regime_expected_duration(P: Array) -> Array:
    """Expected regime durations ``1 / (1 - p_ii)`` for each state."""
    m = np.asarray(P, dtype=float)
    if m.ndim != 2 or m.shape[0] != m.shape[1]:
        raise ValueError("P must be a square matrix")
    if np.any(m < -1e-8) or np.any(np.abs(m.sum(axis=1) - 1.0) > 1e-6):
        raise ValueError("P must be a stochastic matrix")
    return 1.0 / np.maximum(1.0 - np.diag(m), 1e-8)


def regime_classification_uncertainty(smoothed: Array) -> float:
    """Mean entropy of the smoothed state distribution — assignment
    ambiguity measure (higher = less certain regime calls).
    """
    s = np.asarray(smoothed, dtype=float)
    if s.ndim != 2 or not np.all(np.isfinite(s)) or np.any(s < -1e-6):
        raise ValueError("smoothed must be a finite (t, k) prob matrix")
    s = np.clip(s, 1e-12, 1.0)
    return float(-np.mean(np.sum(s * np.log(s), axis=1)))


def ergodic_probabilities(P: Array) -> Array:
    """Stationary (ergodic) distribution of the Markov chain."""
    m = np.asarray(P, dtype=float)
    if m.ndim != 2 or m.shape[0] != m.shape[1]:
        raise ValueError("P must be square")
    vals, vecs = np.linalg.eig(m.T)
    idx = int(np.argmin(np.abs(vals - 1.0)))
    pi = np.real(vecs[:, idx])
    pi = np.abs(pi) / np.sum(np.abs(pi))
    return np.asarray(pi, dtype=float)
