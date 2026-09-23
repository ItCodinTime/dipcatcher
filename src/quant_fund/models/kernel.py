"""Kernel methods for nonlinear regression and dimensionality.

References:
- Rasmussen & Williams (2006): Gaussian process regression (exact
  Cholesky inference, marginal likelihood, predictive variance).
- Shawe-Taylor & Cristianini (2004): kernel ridge regression.
- Scholkopf, Smola & Muller (1998): kernel PCA.
- Williams & Seeger (2001): Nystrom-style subsample GP (subset cap).
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


def _xmat(X: Array, min_n: int = 10) -> Array:
    A = np.asarray(X, dtype=float)
    if A.ndim == 1:
        A = A[:, None]
    if A.ndim != 2 or A.shape[0] < min_n or not np.all(np.isfinite(A)):
        raise ValueError(f"X must be finite (n >= {min_n}, d)")
    if A.shape[0] >= 5 and np.all(A.std(axis=0) == 0):
        raise ValueError("X is degenerate (all columns constant)")
    return A


def _rbf(A: Array, B: Array, length: float, signal: float) -> Array:
    d2 = ((A[:, None, :] - B[None, :, :]) ** 2).sum(axis=2)
    return signal * signal * np.exp(-d2 / (2.0 * length * length))


def _median_length(A: Array) -> float:
    d2 = ((A[:, None, :] - A[None, :, :]) ** 2).sum(axis=2)
    med = float(np.median(np.sqrt(d2[d2 > 0]))) if np.any(d2 > 0) else 1.0
    return max(med, 1e-6)


def fit_gp_regression(
    X: Array,
    y: Array,
    length: float | None = None,
    signal: float | None = None,
    noise: float | None = None,
    optimize: bool = True,
) -> dict[str, Array | float]:
    """Exact GP regression with RBF kernel (Rasmussen–Williams ch. 2).

    Fits hyperparameters by marginal likelihood when ``optimize``;
    returns the Cholesky factor, alpha = K^{-1} y, and hyperparameters
    for later ``predict_gp`` calls. Fail-closed on non-PSD systems.
    """
    A = _xmat(X)
    yv = np.asarray(y, dtype=float).reshape(-1)
    if yv.size != A.shape[0] or not np.all(np.isfinite(yv)):
        raise ValueError("y must be finite and match n")
    n, d = A.shape
    l0 = _median_length(A)
    s0 = float(np.std(yv)) if float(np.std(yv)) > 1e-9 else 1.0
    n0 = 0.05 * s0

    if optimize and n >= 15:
        from scipy import optimize as opt

        def nll(theta: Array) -> float:
            ell, s, nz = np.exp(theta)  # optimize in log space
            K = _rbf(A, A, ell, s) + nz * nz * np.eye(n)
            try:
                L = np.linalg.cholesky(K)
            except np.linalg.LinAlgError:
                return 1e12
            alpha = _chol_solve(L, yv)
            ll = -0.5 * float(yv @ alpha) - float(np.sum(np.log(np.diag(L)))) - 0.5 * n * math.log(2 * math.pi)
            return -ll if np.isfinite(ll) else 1e12

        res = opt.minimize(
            nll,
            np.log([l0, s0, n0]),
            method="Nelder-Mead",
            options={"maxiter": 80, "xatol": 1e-3},
        )
        if np.isfinite(res.fun):
            ell, s, nz = np.exp(res.x)
        else:
            ell, s, nz = l0, s0, n0
    else:
        ell = length if length is not None else l0
        s = signal if signal is not None else s0
        nz = noise if noise is not None else n0
    K = _rbf(A, A, float(ell), float(s)) + float(nz) ** 2 * np.eye(n)
    try:
        L = np.linalg.cholesky(K)
    except np.linalg.LinAlgError as exc:
        raise ValueError("GP kernel matrix not positive definite") from exc
    alpha = _chol_solve(L, yv)
    return {
        "X": A,
        "L": L,
        "alpha": alpha,
        "length": float(ell),
        "signal": float(s),
        "noise": float(nz),
    }


def _chol_solve(L: Array, b: Array) -> Array:
    z = np.linalg.solve(L, b)
    return np.linalg.solve(L.T, z)


def predict_gp(fit: dict[str, Array | float], X_new: Array) -> dict[str, Array]:
    """GP posterior mean and predictive variance at ``X_new``."""
    A = _xmat(X_new, min_n=1)
    Xtr = np.asarray(fit["X"], dtype=float)
    ell = float(fit["length"])
    s = float(fit["signal"])
    nz = float(fit["noise"])
    Ks = _rbf(Xtr, A, ell, s)
    Kss = _rbf(A, A, ell, s) + nz * nz * np.eye(A.shape[0])
    L = np.asarray(fit["L"], dtype=float)
    alpha = np.asarray(fit["alpha"], dtype=float)
    mean = Ks.T @ alpha
    V = _chol_solve(L, Ks)
    var = np.diag(Kss) - np.sum(V * Ks, axis=0)
    var = np.maximum(var, 0.0)
    return {"mean": np.asarray(mean), "std": np.sqrt(var)}


def fit_kernel_ridge(
    X: Array, y: Array, lam: float = 1e-3, length: float | None = None
) -> dict[str, Array | float]:
    """Kernel ridge regression (Shawe–Taylor & Cristianini 2004):
    ``alpha = (K + n*lam*I)^{-1} y``."""
    A = _xmat(X)
    yv = np.asarray(y, dtype=float).reshape(-1)
    if yv.size != A.shape[0] or not np.all(np.isfinite(yv)):
        raise ValueError("y must be finite and match n")
    if lam <= 0:
        raise ValueError("lam must be positive")
    n = A.shape[0]
    ell = length if length is not None else _median_length(A)
    K = _rbf(A, A, ell, 1.0) + n * lam * np.eye(n)
    try:
        alpha = np.linalg.solve(K, yv)
    except np.linalg.LinAlgError as exc:
        raise ValueError("kernel ridge system singular") from exc
    return {"X": A, "alpha": alpha, "length": float(ell), "lam": float(lam)}


def predict_kernel_ridge(fit: dict[str, Array | float], X_new: Array) -> Array:
    A = _xmat(X_new, min_n=1)
    Ks = _rbf(A, np.asarray(fit["X"], dtype=float), float(fit["length"]), 1.0)
    return np.asarray(Ks @ np.asarray(fit["alpha"], dtype=float))


def kernel_pca(
    X: Array, n_components: int = 3, length: float | None = None
) -> dict[str, Array]:
    """Scholkopf et al. (1998) kernel PCA with RBF kernel.

    Returns principal coordinates and eigenvalues (explained variance
    shares). Centering is done in feature space via the double-centering
    trick on K.
    """
    A = _xmat(X)
    n = A.shape[0]
    if not (1 <= n_components < n):
        raise ValueError("n_components must be in [1, n)")
    ell = length if length is not None else _median_length(A)
    K = _rbf(A, A, ell, 1.0)
    one = np.ones((n, n)) / n
    Kc = K - one @ K - K @ one + one @ K @ one
    vals, vecs = np.linalg.eigh(Kc)
    order = np.argsort(vals)[::-1]
    vals = np.maximum(vals[order], 0.0)
    vecs = vecs[:, order]
    k = min(n_components, int(np.sum(vals > 1e-10)))
    if k == 0:
        raise ValueError("kernel PCA degenerate (no positive eigenvalues)")
    coords = vecs[:, :k] * np.sqrt(vals[:k])[None, :]
    shares = vals[:k] / max(vals.sum(), 1e-20)
    return {
        "components": coords,
        "eigenvalues": vals[:k],
        "explained_share": shares,
        "length": np.array([ell]),
    }
