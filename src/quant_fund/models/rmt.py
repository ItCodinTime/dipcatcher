"""Random-matrix-theory diagnostics and covariance denoising.

References:
- Marchenko, Pastur (1967). Distribution of eigenvalues for some sets of
  random matrices. *Mathematics of the USSR-Sbornik* 1.
- Laloux et al. (1999). Noise dressing of financial correlation matrices.
  *Physical Review Letters* 83.
- Plerou et al. (2002). Random matrix approach to cross correlations.
  *Physical Review E* 65.
- López de Prado (2018). *Advances in Financial Machine Learning*, ch. 2 —
  constant-residual eigenvalue denoising and detoning.
- Kritzman, Li, Page, Rigobon (2011). Principal components as a measure of
  systemic risk (absorption ratio). *Journal of Portfolio Management* 37.
- Meucci (2009). Managing diversification (effective rank / entropy of
  eigenvalue distribution). *Risk* 22.
- Participation ratio (IPR) — standard localization diagnostic.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


def _as_matrix(x: Array, name: str = "X", *, min_cols: int = 2) -> Array:
    m = np.asarray(x, dtype=float)
    if m.ndim != 2 or not np.all(np.isfinite(m)):
        raise ValueError(f"{name} must be a finite 2-D array")
    if m.shape[0] < 10 or m.shape[1] < min_cols:
        raise ValueError(f"{name} must have >= 10 rows and >= {min_cols} cols")
    return m


def _as_cov(c: Array, name: str = "cov") -> Array:
    m = np.asarray(c, dtype=float)
    if m.ndim != 2 or m.shape[0] != m.shape[1] or not np.all(np.isfinite(m)):
        raise ValueError(f"{name} must be a finite square matrix")
    if m.shape[0] < 2:
        raise ValueError(f"{name} must be at least 2x2")
    if np.max(np.abs(m - m.T)) > 1e-6 * max(1.0, float(np.abs(m).max())):
        raise ValueError(f"{name} must be symmetric")
    return m


def marchenko_pastur_bounds(q: float, sigma2: float = 1.0) -> tuple[float, float]:
    """Marchenko–Pastur (1967) support edges for noise eigenvalues.

    ``q`` = N/T (assets per observation), ``sigma2`` = noise variance.
    Returns ``(lambda_min, lambda_max) = sigma2 * (1 ± sqrt(q))^2``.
    """
    if not np.isfinite(q) or q <= 0.0:
        raise ValueError("q must be positive and finite")
    if not np.isfinite(sigma2) or sigma2 <= 0.0:
        raise ValueError("sigma2 must be positive and finite")
    lo = sigma2 * (1.0 - math.sqrt(q)) ** 2
    hi = sigma2 * (1.0 + math.sqrt(q)) ** 2
    return lo, hi


def marchenko_pastur_pdf(lam: Array, q: float, sigma2: float = 1.0) -> Array:
    """MP density evaluated at ``lam`` (0 outside the support)."""
    lo, hi = marchenko_pastur_bounds(q, sigma2)
    lv = np.asarray(lam, dtype=float)
    out = np.zeros_like(lv)
    ok = (lv > lo) & (lv < hi)
    out[ok] = np.sqrt((hi - lv[ok]) * (lv[ok] - lo)) / (2.0 * math.pi * q * sigma2 * lv[ok])
    return out


def correlation_eigenvalues(x: Array) -> tuple[Array, Array]:
    """Eigenpairs of the sample correlation matrix of T×N returns ``x``.

    Columns are standardized to unit variance before correlating (correlation
    matrix, not covariance — the MP null applies to standardized data).
    Returns eigenvalues in descending order and matching eigenvectors.
    """
    m = _as_matrix(x)
    t, n = m.shape
    z = m - m.mean(axis=0)
    sd = z.std(axis=0, ddof=1)
    if np.any(sd <= 0.0):
        raise ValueError("columns of x must have positive variance")
    z /= sd
    corr = (z.T @ z) / (t - 1.0)
    vals, vecs = np.linalg.eigh(corr)
    order = np.argsort(vals)[::-1]
    return vals[order], vecs[:, order]


def noise_fraction(eigenvalues: Array, q: float, sigma2: float = 1.0) -> float:
    """Fraction of eigenvalues inside the MP noise band."""
    v = np.asarray(eigenvalues, dtype=float).reshape(-1)
    if v.size < 2 or not np.all(np.isfinite(v)):
        raise ValueError("eigenvalues must be a finite vector of length >= 2")
    lo, hi = marchenko_pastur_bounds(q, sigma2)
    return float(np.mean((v >= lo) & (v <= hi)))


def eigenvalue_clip(cov: Array, q: float, *, constant_residual: bool = True) -> Array:
    """Laloux-style eigenvalue clipping (AFML ch. 2, ``denoiseCov``).

    Eigenvalues at or below the MP upper edge are treated as noise: each is
    replaced by its own value (no change) or, with ``constant_residual=True``,
    by the mean noise eigenvalue — preserving trace while killing noise
    variance.  Returns the reconstructed covariance.
    """
    m = _as_cov(cov)
    lo, hi = marchenko_pastur_bounds(q)
    vals, vecs = np.linalg.eigh(m)
    noise = vals <= hi
    if np.any(noise) and constant_residual:
        vals[noise] = vals[noise].mean()
    out = (vecs * vals) @ vecs.T
    return (out + out.T) / 2.0


def detone_cov(cov: Array, n_market: int = 1) -> Array:
    """Remove the top ``n_market`` eigencomponents (the 'market mode').

    AFML ``detone``: zero out the largest eigenvalues and rebuild.  Useful to
    isolate residual correlation structure for clustering/HRP.
    """
    m = _as_cov(cov)
    if isinstance(n_market, bool) or not isinstance(n_market, int):
        raise ValueError("n_market must be a positive integer")
    if n_market < 1 or n_market >= m.shape[0]:
        raise ValueError("n_market must be in [1, n_assets)")
    vals, vecs = np.linalg.eigh(m)
    vals[-n_market:] = 0.0
    out = (vecs * vals) @ vecs.T
    return (out + out.T) / 2.0


def absorption_ratio(eigenvalues: Array, top_k: int | None = None) -> float:
    """Kritzman et al. (2011) absorption ratio.

    Share of total variance absorbed by the top ``top_k`` eigenvectors
    (default ``ceil(n/5)``, the original 1/5 convention).
    """
    v = np.asarray(eigenvalues, dtype=float).reshape(-1)
    if v.size < 2 or not np.all(np.isfinite(v)) or np.any(v < 0.0):
        raise ValueError("eigenvalues must be a finite non-negative vector")
    k = max(1, int(math.ceil(v.size / 5.0))) if top_k is None else int(top_k)
    if k < 1 or k > v.size:
        raise ValueError("top_k must be in [1, len(eigenvalues)]")
    total = float(v.sum())
    if total <= 0.0:
        raise ValueError("eigenvalues must have positive total variance")
    s = np.sort(v)[::-1]
    return float(s[:k].sum() / total)


def effective_rank(eigenvalues: Array) -> float:
    """Meucci-style effective number of uncorrelated bets.

    ``exp(entropy(w))`` where ``w_i = lambda_i / sum(lambda)`` — equals n for
    spherical risk, 1 for a single dominant factor.
    """
    v = np.asarray(eigenvalues, dtype=float).reshape(-1)
    if v.size < 2 or not np.all(np.isfinite(v)) or np.any(v < 0.0):
        raise ValueError("eigenvalues must be a finite non-negative vector")
    total = float(v.sum())
    if total <= 0.0:
        raise ValueError("eigenvalues must have positive total variance")
    w = v[v > 0.0] / total
    return float(np.exp(-np.sum(w * np.log(w))))


def inverse_participation_ratio(eigenvector: Array) -> float:
    """IPR of a single eigenvector: ``sum(u^4)``.

    In [1/n, 1]; ~1/n for a delocalized (market-like) mode, ~1 for a vector
    concentrated on a single asset.
    """
    u = np.asarray(eigenvector, dtype=float).reshape(-1)
    if u.size < 2 or not np.all(np.isfinite(u)):
        raise ValueError("eigenvector must be a finite vector of length >= 2")
    nrm = float(np.linalg.norm(u))
    if nrm <= 0.0:
        raise ValueError("eigenvector must be non-zero")
    u = u / nrm
    return float(np.sum(u**4))


def risk_in_eigenmodes(weights: Array, cov: Array) -> Array:
    """Share of portfolio variance carried by each eigenmode (Plerou 2002).

    ``w' = V^T w`` projected onto eigenvectors; returns the normalized
    contribution ``(w'_i^2 * lambda_i) / w'Σw``.
    """
    w = np.asarray(weights, dtype=float).reshape(-1)
    m = _as_cov(cov)
    if w.size != m.shape[0] or not np.all(np.isfinite(w)):
        raise ValueError("weights must be a finite vector matching cov")
    vals, vecs = np.linalg.eigh(m)
    proj = vecs.T @ w
    contrib = vals * proj**2
    total = float(contrib.sum())
    if total <= 0.0:
        raise ValueError("portfolio variance is zero")
    return np.asarray(contrib / total, dtype=float)
