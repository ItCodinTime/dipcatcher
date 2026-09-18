"""Covariance estimators and PSD repair."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from sklearn.covariance import OAS, LedoitWolf

from quant_fund.utils.logging import get_logger

Array = NDArray[np.float64]
log = get_logger(module="covariance")


def is_symmetric(sigma: Array, tol: float = 1e-10) -> bool:
    return bool(np.max(np.abs(sigma - sigma.T)) <= tol)


def min_eigenvalue(sigma: Array) -> float:
    return float(np.min(np.linalg.eigvalsh(0.5 * (sigma + sigma.T))))


def repair_psd(sigma: Array, tol: float = 1e-10) -> tuple[Array, dict[str, float]]:
    s = np.asarray(sigma, dtype=float)
    if s.ndim != 2 or s.shape[0] != s.shape[1] or s.shape[0] == 0:
        raise ValueError("sigma must be a non-empty square matrix")
    if not np.isfinite(s).all():
        raise ValueError("sigma must contain only finite values")
    s = 0.5 * (s + s.T)
    eig_min = float(np.min(np.linalg.eigvalsh(s)))
    if eig_min >= -tol and is_symmetric(sigma):
        return s, {"repaired": 0.0, "eig_min_before": eig_min, "eig_min_after": eig_min}
    # Work directly in the symmetric eigensystem.  ``cov_nearest`` can return
    # NaNs for finite covariance-like inputs with negative diagonal entries;
    # clipping eigenvalues is deterministic and guarantees a finite PSD result.
    eigenvalues, eigenvectors = np.linalg.eigh(s)
    clipped = np.maximum(eigenvalues, max(tol, 0.0))
    repaired = (eigenvectors * clipped) @ eigenvectors.T
    repaired = 0.5 * (repaired + repaired.T)
    after = float(np.min(np.linalg.eigvalsh(repaired)))
    fro = float(np.linalg.norm(repaired - s, "fro"))
    log.warning("psd_repair", eig_min_before=eig_min, eig_min_after=after, frobenius=fro)
    return repaired, {
        "repaired": 1.0,
        "eig_min_before": eig_min,
        "eig_min_after": after,
        "frobenius": fro,
    }


def _clean_returns(returns: Array, *, min_rows: int = 2) -> Array:
    x = np.asarray(returns, dtype=float)
    if x.ndim != 2 or x.shape[1] == 0:
        raise ValueError("returns must be a non-empty 2D array")
    x = x[np.isfinite(x).all(axis=1)]
    if x.shape[0] < min_rows:
        raise ValueError(f"at least {min_rows} finite return rows are required")
    return x


def _validate_lambda(lam: float) -> None:
    if not np.isfinite(lam) or not 0.0 <= lam <= 1.0:
        raise ValueError("lam must be finite and between 0 and 1")


def sample_cov(returns: Array) -> Array:
    x = _clean_returns(returns)
    return np.atleast_2d(np.asarray(np.cov(x, rowvar=False), dtype=float))


def ewma_cov(returns: Array, lam: float = 0.94) -> Array:
    _validate_lambda(lam)
    x = _clean_returns(returns, min_rows=1)
    n, k = x.shape
    cov = np.outer(x[0], x[0])
    for t in range(1, n):
        r = x[t - 1]
        cov = lam * cov + (1.0 - lam) * np.outer(r, r)
    return 0.5 * (cov + cov.T)


def ledoit_wolf_cov(returns: Array) -> Array:
    x = _clean_returns(returns)
    return np.asarray(LedoitWolf().fit(x).covariance_, dtype=float)


def oracle_approximating_shrinkage_cov(returns: Array) -> Array:
    """Estimate covariance with finite-sample Oracle Approximating Shrinkage.

    OAS is a linear shrinkage estimator, not nonlinear spectral shrinkage. It
    is useful when the asset dimension is close to or exceeds the observation
    count and follows the module's finite-input and PSD-repair contract.
    """
    x = _clean_returns(returns)
    sigma = np.asarray(OAS().fit(x).covariance_, dtype=float)
    if not np.isfinite(sigma).all():
        raise ValueError("OAS covariance produced non-finite values")
    repaired, _ = repair_psd(sigma)
    return repaired


def factor_cov(betas: Array, factor_cov: Array, idio_var: Array) -> Array:
    b = np.asarray(betas, dtype=float)
    f = np.asarray(factor_cov, dtype=float)
    dvar = np.asarray(idio_var, dtype=float)
    if b.ndim != 2 or f.shape != (b.shape[1], b.shape[1]) or dvar.shape != (b.shape[0],):
        raise ValueError("factor covariance inputs have incompatible shapes")
    if not np.isfinite(b).all() or not np.isfinite(f).all() or not np.isfinite(dvar).all():
        raise ValueError("factor covariance inputs must be finite")
    if not np.allclose(f, f.T) or np.any(dvar < 0):
        raise ValueError("factor covariance must be symmetric and idio_var non-negative")
    if float(np.min(np.linalg.eigvalsh(f))) < -1e-10:
        raise ValueError("factor covariance must be positive semidefinite")
    d = np.diag(dvar)
    return np.asarray(b @ f @ b.T + d, dtype=np.float64)


def dcc_gaussian(
    returns: Array, a0: float | None = None, b0: float | None = None
) -> tuple[Array, dict[str, float]]:
    """Two-stage Gaussian DCC(1,1). Returns last H_t and params.

    Stage 1: EWMA vols (fast, no per-name GARCH optimizer in the inner loop).
    Stage 2: QML on a, b with correlation targeting.
    """
    from scipy.optimize import minimize

    x = _clean_returns(returns, min_rows=3)
    t, n = x.shape
    if a0 is not None and (not np.isfinite(a0) or a0 < 0):
        raise ValueError("a0 must be finite and non-negative")
    if b0 is not None and (not np.isfinite(b0) or b0 < 0):
        raise ValueError("b0 must be finite and non-negative")
    vol = np.zeros_like(x)
    for j in range(n):
        v = ewma_variance_1d(x[:, j])
        vol[:, j] = np.sqrt(np.clip(v, 1e-16, None))
    z = x / np.clip(vol, 1e-12, None)
    with np.errstate(divide="ignore", invalid="ignore"):
        qbar = np.asarray(np.corrcoef(z, rowvar=False), dtype=float)
    if not np.isfinite(qbar).all():
        raise ValueError("DCC correlation target is non-finite; input has unusable variance")
    np.fill_diagonal(qbar, 1.0)
    qbar, _ = repair_psd(qbar, tol=1e-12)

    def nll(params: Array) -> float:
        a, b = float(params[0]), float(params[1])
        if a < 0 or b < 0 or a + b >= 0.999:
            return 1e12
        q = qbar.copy()
        ll = 0.0
        for i in range(1, t):
            q = (1 - a - b) * qbar + a * np.outer(z[i - 1], z[i - 1]) + b * q
            d = np.sqrt(np.clip(np.diag(q), 1e-12, None))
            r = q / np.outer(d, d)
            r = 0.5 * (r + r.T)
            np.fill_diagonal(r, 1.0)
            sign, logdet = np.linalg.slogdet(r)
            if sign <= 0:
                return 1e12
            try:
                quadratic = float(z[i] @ np.linalg.solve(r, z[i]))
            except np.linalg.LinAlgError:
                return 1e12
            ll += logdet + quadratic
        return float(ll / t)

    x0 = np.array([0.05 if a0 is None else a0, 0.9 if b0 is None else b0])
    x0 = np.clip(x0, 1e-6, 0.99)
    if x0.sum() >= 0.99:
        x0 *= 0.98 / x0.sum()
    res = minimize(
        nll,
        x0,
        bounds=[(1e-6, 0.5), (1e-6, 0.99)],
        constraints={"type": "ineq", "fun": lambda p: 0.999 - p[0] - p[1]},
        method="SLSQP",
    )
    candidate = np.asarray(res.x if res.success and np.isfinite(res.fun) else x0)
    a, b = float(candidate[0]), float(candidate[1])
    q = qbar.copy()
    for i in range(1, t):
        q = (1 - a - b) * qbar + a * np.outer(z[i - 1], z[i - 1]) + b * q
    d = np.sqrt(np.clip(np.diag(q), 1e-12, None))
    r = q / np.outer(d, d)
    d_last = np.diag(vol[-1])
    h = d_last @ r @ d_last
    h, _ = repair_psd(h)
    return h, {"a": a, "b": b, "success": float(res.success)}


def ewma_variance_1d(r: Array, lam: float = 0.94) -> Array:
    _validate_lambda(lam)
    r = np.asarray(r, dtype=float)
    if r.ndim != 1 or r.size == 0 or not np.isfinite(r).all():
        raise ValueError("r must be a non-empty finite 1D array")
    v = np.empty_like(r, dtype=float)
    v[0] = r[0] ** 2
    for t in range(1, r.size):
        v[t] = lam * v[t - 1] + (1.0 - lam) * r[t - 1] ** 2
    return v


def condition_number(sigma: Array) -> float:
    s = np.asarray(sigma, dtype=float)
    if s.ndim != 2 or s.shape[0] != s.shape[1] or s.shape[0] == 0:
        raise ValueError("sigma must be a non-empty square matrix")
    if not np.isfinite(s).all():
        raise ValueError("sigma must contain only finite values")
    eig = np.linalg.eigvalsh(0.5 * (s + s.T))
    if float(eig[0]) <= 1e-12:
        return float("inf")
    return float(eig.max() / eig.min())
