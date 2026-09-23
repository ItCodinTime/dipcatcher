"""Conjugate Bayesian linear regression and Bayesian model averaging.

Normal-Inverse-Gamma model: y = X b + eps, eps ~ N(0, s^2 I);
b | s^2 ~ N(b0, s^2 V0); s^2 ~ IG(a0, b0_scale).

Posterior is NIG with
  V_n = (V0^{-1} + X'X)^{-1},  b_n = V_n (V0^{-1} b0 + X'y),
  a_n = a0 + n/2,
  b_n = b0 + 0.5 (y'y + b0' V0^{-1} b0 - b_n' V_n^{-1} b_n).
Posterior predictive at x: Student-t with df = 2 a_n, mean x' b_n,
scale^2 = (b_n/a_n) (1 + x' V_n x).

Default prior is Zellner's g-prior: b0 = 0, V0 = g (X'X)^{-1}.

BMA enumerates subsets of the p regressors (feasible for p <= ~10),
weights by marginal likelihood under the same g-prior.

Fail-closed on non-finite input, singular designs, invalid hyperparams.
"""

from __future__ import annotations

from itertools import combinations
from math import lgamma

import numpy as np
from numpy.typing import NDArray
from scipy.stats import t as student_t

Array = NDArray[np.float64]


def _design(x: Array) -> tuple[Array, Array]:
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    if x.ndim != 2 or not np.isfinite(x).all():
        raise ValueError("X must be a finite 2-D array")
    return x, np.column_stack([np.ones(x.shape[0]), x])


def bayes_ols(
    y: Array,
    x: Array,
    g: float | None = None,
    b0: Array | None = None,
    a0: float = 0.01,
    b0_scale: float = 0.01,
) -> dict[str, Array | float]:
    """Conjugate NIG Bayesian OLS. Intercept added automatically.

    ``g`` selects the Zellner g-prior V0 = g (X'X)^{-1} on all columns
    including the intercept (default g = max(n, p^2) ~ BIC-ish).
    Returns posterior dict with b_n, V_n, a_n, b_n(scale), log_ml.
    """
    y = np.asarray(y, dtype=float).ravel()
    x, z = _design(x)
    n, k = z.shape
    if y.shape[0] != n:
        raise ValueError("y and X must have matching rows")
    if not np.isfinite(y).all() or n <= k + 2:
        raise ValueError("need n > k + 2 finite observations")
    if not np.isfinite([a0, b0_scale]).all() or a0 <= 0.0 or b0_scale <= 0.0:
        raise ValueError("a0, b0_scale must be > 0")
    if g is None:
        g = float(max(n, k * k))
    if not np.isfinite(g) or g <= 0.0:
        raise ValueError("g must be > 0")
    xtx = z.T @ z
    try:
        xtx_inv = np.linalg.inv(xtx)
    except np.linalg.LinAlgError as e:
        raise ValueError("design matrix singular") from e
    v0 = g * xtx_inv
    v0_inv = xtx / g
    b0v = np.zeros(k) if b0 is None else np.asarray(b0, dtype=float).ravel()
    if b0v.shape != (k,) or not np.isfinite(b0v).all():
        raise ValueError("b0 must be finite, length k")

    v_n = np.linalg.inv(v0_inv + xtx)
    beta_n = v_n @ (v0_inv @ b0v + z.T @ y)
    a_n = a0 + n / 2.0
    b_scale = b0_scale + 0.5 * float(
        y @ y + b0v @ v0_inv @ b0v - beta_n @ np.linalg.inv(v_n) @ beta_n
    )
    sign_v0, logdet_v0 = np.linalg.slogdet(v0)
    sign_vn, logdet_vn = np.linalg.slogdet(v_n)
    log_ml = (
        a0 * np.log(b0_scale)
        - a_n * np.log(b_scale)
        + 0.5 * (logdet_vn - logdet_v0)
        + lgamma(a_n)
        - lgamma(a0)
        - (n / 2.0) * np.log(np.pi)
    )
    return {
        "b_n": np.asarray(beta_n, dtype=float),
        "V_n": v_n,
        "a_n": float(a_n),
        "b_n_scale": float(b_scale),
        "log_ml": float(log_ml),
        "g": float(g),
        "k": float(k),
        "n": float(n),
    }


def bayes_predictive(
    post: dict[str, Array | float],
    x_new: Array,
) -> dict[str, Array | float]:
    """Posterior predictive Student-t at new design rows (intercept added)."""
    x = np.asarray(x_new, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    if x.ndim != 2 or not np.isfinite(x).all():
        raise ValueError("x_new must be finite 2-D")
    z = np.column_stack([np.ones(x.shape[0]), x])
    b_n = np.asarray(post["b_n"], dtype=float)
    v_n = np.asarray(post["V_n"], dtype=float)
    a_n = float(post["a_n"])
    b_scale = float(post["b_n_scale"])
    if z.shape[1] != b_n.shape[0]:
        raise ValueError("x_new columns inconsistent with posterior")
    mean = z @ b_n
    df = 2.0 * a_n
    scale2 = (b_scale / a_n) * (1.0 + np.einsum("ij,jk,ik->i", z, v_n, z))
    return {
        "mean": np.asarray(mean, dtype=float),
        "df": df,
        "scale": np.sqrt(np.maximum(scale2, 0.0)),
        "lo_95": mean + student_t.ppf(0.025, df) * np.sqrt(np.maximum(scale2, 0.0)),
        "hi_95": mean + student_t.ppf(0.975, df) * np.sqrt(np.maximum(scale2, 0.0)),
    }


def bayes_log_ml(y: Array, x: Array, g: float | None = None) -> float:
    """Marginal likelihood (evidence) of a single regression spec."""
    return float(bayes_ols(y, x, g=g)["log_ml"])


def bayesian_model_averaging(
    y: Array,
    x: Array,
    g: float | None = None,
    prior_prob: float | None = None,
) -> dict[str, Array | float]:
    """Enumerate all non-empty regressor subsets; return BMA posterior.

    Returns: subset list (tuples of column indices into x), model_probs,
    posterior inclusion probability per column, BMA coefficient means
    (intercept excluded from averaging — always included), log-evidence.
    """
    y = np.asarray(y, dtype=float).ravel()
    xx = np.asarray(x, dtype=float)
    if xx.ndim == 1:
        xx = xx[:, None]
    if xx.ndim != 2 or not np.isfinite(xx).all():
        raise ValueError("x must be finite 2-D")
    p = xx.shape[1]
    if p < 1 or p > 10:
        raise ValueError("BMA enumeration needs 1 <= p <= 10 regressors")
    if y.shape[0] != xx.shape[0]:
        raise ValueError("y and x must have matching rows")

    subsets: list[tuple[int, ...]] = []
    for r in range(0, p + 1):
        subsets.extend(combinations(range(p), r))
    n_models = len(subsets)
    subset_arr = np.full((n_models, p), -1, dtype=int)
    for i, s in enumerate(subsets):
        subset_arr[i, : len(s)] = s
    if prior_prob is None:
        prior_prob = 1.0 / len(subsets)
    if not np.isfinite(prior_prob) or prior_prob <= 0.0:
        raise ValueError("prior_prob must be > 0")

    log_mls = np.empty(len(subsets))
    posts: list[dict[str, Array | float]] = []
    for i, sub in enumerate(subsets):
        cols = xx[:, list(sub)] if sub else np.empty((y.size, 0))
        post = bayes_ols(y, cols, g=g)
        posts.append(post)
        log_mls[i] = float(post["log_ml"])
    lse = log_mls.max()
    w = np.exp(log_mls - lse)
    w /= w.sum()
    probs = w * prior_prob  # prior uniform; keep explicit for clarity
    probs /= probs.sum()

    pip = np.zeros(p)
    for i, sub in enumerate(subsets):
        for c in sub:
            pip[c] += probs[i]
    bma_beta = np.zeros(p)
    for i, sub in enumerate(subsets):
        b_n = np.asarray(posts[i]["b_n"], dtype=float)
        for j, c in enumerate(sub):
            bma_beta[c] += probs[i] * b_n[j + 1]
    return {
        "subsets": subset_arr,
        "model_probs": probs,
        "pip": pip,
        "bma_beta": bma_beta,
        "log_marginal_likelihoods": log_mls,
        "n_models": float(n_models),
    }
