"""Covariance estimators and PSD repair."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from sklearn.covariance import LedoitWolf
from statsmodels.stats.correlation_tools import cov_nearest

from quant_fund.utils.logging import get_logger

Array = NDArray[np.float64]
log = get_logger(module="covariance")


def is_symmetric(sigma: Array, tol: float = 1e-10) -> bool:
    return bool(np.max(np.abs(sigma - sigma.T)) <= tol)


def min_eigenvalue(sigma: Array) -> float:
    return float(np.min(np.linalg.eigvalsh(0.5 * (sigma + sigma.T))))


def repair_psd(sigma: Array, tol: float = 1e-10) -> tuple[Array, dict[str, float]]:
    s = np.asarray(sigma, dtype=float)
    s = 0.5 * (s + s.T)
    eig_min = float(np.min(np.linalg.eigvalsh(s)))
    if eig_min >= -tol and is_symmetric(sigma):
        return s, {"repaired": 0.0, "eig_min_before": eig_min, "eig_min_after": eig_min}
    repaired = np.asarray(cov_nearest(s, method="clipped", threshold=max(tol, 0.0)), dtype=float)
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
    return np.asarray(np.cov(x, rowvar=False), dtype=float)


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
    x = np.asarray(returns, dtype=float)
    x = x[np.isfinite(x).all(axis=1)]
    return np.asarray(LedoitWolf().fit(x).covariance_, dtype=float)


def factor_cov(betas: Array, factor_cov: Array, idio_var: Array) -> Array:
    b = np.asarray(betas, dtype=float)
    f = np.asarray(factor_cov, dtype=float)
    d = np.diag(np.asarray(idio_var, dtype=float))
    return b @ f @ b.T + d


def dcc_gaussian(
    returns: Array, a0: float | None = None, b0: float | None = None
) -> tuple[Array, dict[str, float]]:
    """Two-stage Gaussian DCC(1,1). Returns last H_t and params.

    Stage 1: EWMA vols (fast, no per-name GARCH optimizer in the inner loop).
    Stage 2: QML on a, b with correlation targeting.
    """
    from scipy.optimize import minimize

    x = np.asarray(returns, dtype=float)
    x = x[np.isfinite(x).all(axis=1)]
    t, n = x.shape
    vol = np.zeros_like(x)
    for j in range(n):
        v = ewma_variance_1d(x[:, j])
        vol[:, j] = np.sqrt(np.clip(v, 1e-16, None))
    z = x / np.clip(vol, 1e-12, None)
    qbar = np.corrcoef(z, rowvar=False)
    qbar = np.nan_to_num(qbar, nan=0.0)
    np.fill_diagonal(qbar, 1.0)

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
            ll += logdet + z[i] @ np.linalg.pinv(r) @ z[i]
        return ll / t

    x0 = np.array([0.05 if a0 is None else a0, 0.9 if b0 is None else b0])
    res = minimize(nll, x0, bounds=[(1e-6, 0.5), (1e-6, 0.99)])
    a, b = float(res.x[0]), float(res.x[1])
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
    v = np.empty_like(r, dtype=float)
    v[0] = r[0] ** 2
    for t in range(1, r.size):
        v[t] = lam * v[t - 1] + (1.0 - lam) * r[t - 1] ** 2
    return v


def condition_number(sigma: Array) -> float:
    eig = np.linalg.eigvalsh(0.5 * (sigma + sigma.T))
    eig = eig[eig > 1e-12]
    if eig.size == 0:
        return float("inf")
    return float(eig.max() / eig.min())
