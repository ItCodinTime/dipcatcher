"""Mixture models via EM.

Gaussian and Student-t finite mixtures with EM fitting, BIC/ICL model
selection, filtered/smoothed component probabilities, and mixing-density
diagnostics. Multivariate (diagonal or full covariance).

References:
- Dempster, Laird, Rubin (1977) EM algorithm.
- McLachlan, Peel (2000) finite mixture models.
- Peel, McLachlan (2000) robust t-mixtures (ECM for nu).
- Schwarz (1978) BIC; Biernacki, Celeux, Govaert (2000) ICL.
- Hamilton (1990) mixture-regime context; Kontolemis (2007) regime mixing.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray
from scipy import optimize as opt
from scipy.special import digamma, gammaln, logsumexp

Array = NDArray[np.float64]


def _as_data(x: Array) -> Array:
    m = np.asarray(x, dtype=float)
    if m.ndim == 1:
        m = m.reshape(-1, 1)
    if m.ndim != 2 or m.shape[0] < 10 or not np.all(np.isfinite(m)):
        raise ValueError("x must be a finite (n, d) matrix, n >= 10")
    return m


def _log_norm_pdf(x: Array, mu: Array, cov: Array) -> Array:
    d = x.shape[1]
    sign, logdet = np.linalg.slogdet(cov)
    if sign <= 0:
        raise ValueError("covariance not positive definite")
    diff = x - mu
    quad = np.einsum("ij,jk,ik->i", diff, np.linalg.pinv(cov), diff)
    return np.asarray(-0.5 * (d * math.log(2 * math.pi) + logdet + quad), dtype=float)


def _log_t_pdf(x: Array, mu: Array, cov: Array, nu: float) -> Array:
    d = x.shape[1]
    sign, logdet = np.linalg.slogdet(cov)
    if sign <= 0:
        raise ValueError("covariance not positive definite")
    diff = x - mu
    quad = np.einsum("ij,jk,ik->i", diff, np.linalg.pinv(cov), diff)
    return np.asarray(
        gammaln((nu + d) / 2.0)
        - gammaln(nu / 2.0)
        - 0.5 * (d * math.log(nu * math.pi) + logdet)
        - 0.5 * (nu + d) * np.log1p(quad / nu),
        dtype=float,
    )


def fit_gaussian_mixture(
    x: Array,
    k: int,
    seed: int = 0,
    max_iter: int = 300,
    tol: float = 1e-6,
    diag: bool = False,
) -> dict[str, Array]:
    """EM for a k-component Gaussian mixture.

    Returns weights, means (k, d), covariances (k, d, d), component
    responsibilities (n, k), and the observed-data log-likelihood path.
    """
    m = _as_data(x)
    n, d = m.shape
    if k < 1 or k >= n:
        raise ValueError("k must be in [1, n)")
    rng = np.random.default_rng(seed)
    # k-means++-ish init: random distinct rows.
    idx = rng.choice(n, size=k, replace=False)
    mus = m[idx].copy()
    global_cov = np.cov(m.T) + 1e-6 * np.eye(d)
    covs = np.broadcast_to(global_cov, (k, d, d)).copy()
    if diag:
        covs = np.array([np.diag(np.diag(c)) for c in covs])
    w = np.full(k, 1.0 / k)
    ll_path = []
    prev_ll = -np.inf
    resp = np.full((n, k), 1.0 / k)
    for _ in range(max_iter):
        # E-step.
        log_p = np.column_stack(
            [_log_norm_pdf(m, mus[j], covs[j]) + math.log(w[j]) for j in range(k)]
        )
        ll = float(logsumexp(log_p, axis=1).sum())
        ll_path.append(ll)
        resp = np.exp(log_p - logsumexp(log_p, axis=1, keepdims=True))
        # M-step.
        n_k = resp.sum(axis=0) + 1e-12
        w = n_k / n
        mus = (resp.T @ m) / n_k[:, None]
        for j in range(k):
            diff = m - mus[j]
            c = (resp[:, j] * diff.T) @ diff / n_k[j]
            c += 1e-8 * np.eye(d)
            covs[j] = np.diag(np.diag(c)) if diag else c
        if ll - prev_ll < tol * (1.0 + abs(prev_ll)):
            break
        prev_ll = ll
    return {
        "weights": w,
        "means": mus,
        "covs": covs,
        "responsibilities": resp,
        "loglik": np.array(ll_path),
        "n_iter": np.array([float(len(ll_path))]),
    }


def fit_t_mixture(
    x: Array,
    k: int,
    seed: int = 0,
    max_iter: int = 200,
    tol: float = 1e-5,
) -> dict[str, Array]:
    """ECM for a k-component Student-t mixture (Peel–McLachlan 2000).

    Adds the latent scale u_i ~ Gamma(nu/2, nu/2) E-step and a per-
    component nu line search (damped to [3, 300]).
    """
    m = _as_data(x)
    n, d = m.shape
    if k < 1 or k >= n:
        raise ValueError("k must be in [1, n)")
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=k, replace=False)
    mus = m[idx].copy()
    global_cov = np.cov(m.T) + 1e-6 * np.eye(d)
    covs = np.broadcast_to(global_cov, (k, d, d)).copy()
    w = np.full(k, 1.0 / k)
    nus = np.full(k, 10.0)
    ll_path = []
    prev_ll = -np.inf
    resp = np.full((n, k), 1.0 / k)
    for _ in range(max_iter):
        log_p = np.column_stack(
            [_log_t_pdf(m, mus[j], covs[j], nus[j]) + math.log(w[j]) for j in range(k)]
        )
        ll = float(logsumexp(log_p, axis=1).sum())
        ll_path.append(ll)
        resp = np.exp(log_p - logsumexp(log_p, axis=1, keepdims=True))
        # Latent u | x,j ~ Gamma((nu+d)/2, (nu+delta)/2):
        # E[u] = (nu+d)/(nu+delta); E[ln u] = psi((nu+d)/2) - ln((nu+delta)/2).
        diff_norm = np.zeros((n, k))
        diff_logn = np.zeros((n, k))
        for j in range(k):
            diff = m - mus[j]
            delta = np.einsum("ij,jk,ik->i", diff, np.linalg.pinv(covs[j]), diff)
            diff_norm[:, j] = (nus[j] + d) / (nus[j] + delta)
            diff_logn[:, j] = digamma((nus[j] + d) / 2.0) - np.log((nus[j] + delta) / 2.0)
        n_k = resp.sum(axis=0) + 1e-12
        w = n_k / n
        mus = ((resp * diff_norm).T @ m) / (resp * diff_norm).sum(axis=0)[:, None]
        for j in range(k):
            diff = m - mus[j]
            wr = resp[:, j] * diff_norm[:, j]
            covs[j] = (wr * diff.T) @ diff / n_k[j] + 1e-8 * np.eye(d)
        # CM-step for nu per component (Peel-McLachlan).
        for j in range(k):
            u_bar = float((resp[:, j] * diff_norm[:, j]).sum() / n_k[j])
            log_u_bar = float((resp[:, j] * diff_logn[:, j]).sum() / n_k[j])

            nk, lu, ub = float(n_k[j]), log_u_bar, u_bar

            def nu_obj(nu: float, nk: float = nk, lu: float = lu, ub: float = ub) -> float:
                return float(
                    -(
                        -nk * gammaln(nu / 2.0)
                        + nk * (nu / 2.0) * math.log(nu / 2.0)
                        + (nu / 2.0) * nk * (lu - ub)
                    )
                )

            res = opt.minimize_scalar(nu_obj, bounds=(3.0, 300.0), method="bounded")
            if res.success and np.isfinite(res.x):
                nus[j] = float(res.x)
        if ll - prev_ll < tol * (1.0 + abs(prev_ll)):
            break
        prev_ll = ll
    return {
        "weights": w,
        "means": mus,
        "covs": covs,
        "nus": nus,
        "responsibilities": resp,
        "loglik": np.array(ll_path),
        "n_iter": np.array([float(len(ll_path))]),
    }


def mixture_bic(fit: dict[str, Array], n: int, diag: bool = False) -> float:
    """BIC for a fitted mixture: ``-2 LL + p log n``.

    Parameters counted: k-1 weights + k*d means + k*d(d+1)/2 covs (or
    k*d if diag) + k nu's for t-mixtures (detected via 'nus' key).
    """
    m = fit["means"]
    k, d = m.shape
    p = (k - 1) + k * d + (k * d if diag else k * d * (d + 1) // 2)
    if "nus" in fit:
        p += k
    ll = float(fit["loglik"][-1])
    return float(-2.0 * ll + p * math.log(n))


def select_mixture_k(
    x: Array,
    k_range: range | None = None,
    kind: str = "gaussian",
    seed: int = 0,
) -> dict[str, Array]:
    """Select the number of mixture components by BIC."""
    m = _as_data(x)
    n = m.shape[0]
    ks = list(k_range) if k_range is not None else [1, 2, 3, 4]
    if any(k < 1 or k >= n for k in ks):
        raise ValueError("k_range must satisfy 1 <= k < n")
    bics = np.zeros(len(ks))
    fits = []
    for i, k in enumerate(ks):
        fit = (
            fit_gaussian_mixture(m, k, seed=seed)
            if kind == "gaussian"
            else fit_t_mixture(m, k, seed=seed)
        )
        fits.append(fit)
        bics[i] = mixture_bic(fit, n)
    best = int(np.argmin(bics))
    bf = fits[best]
    return {
        "best_k": np.array([float(ks[best])]),
        "bics": bics,
        "best_weights": bf["weights"],
        "best_means": bf["means"],
        "best_loglik": np.array([float(bf["loglik"][-1])]),
    }


def mixing_density_stats(fit: dict[str, Array]) -> dict[str, float]:
    """Summary of a fitted mixture: effective components, mean gap, and
    the entropy of the responsibility matrix (assignment ambiguity).
    """
    w = fit["weights"]
    mus = fit["means"]
    resp = fit["responsibilities"]
    k = w.size
    eff = 1.0 / float(np.sum(w * w))
    gaps = [float(np.linalg.norm(mus[i] - mus[j])) for i in range(k) for j in range(i + 1, k)]
    ent = float(-np.mean(np.sum(resp * np.log(resp + 1e-16), axis=1)))
    return {
        "eff_components": eff,
        "mean_gap": float(np.mean(gaps)) if gaps else 0.0,
        "resp_entropy": ent,
    }
