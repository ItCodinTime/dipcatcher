"""Bayesian VAR with the Minnesota (Litterman) prior.

Litterman (1986); Banbura, Giannone & Reichlin (2010) dummy-observation
form: prior moments are imposed as pseudo-observations stacked on the data,
so the posterior mean is the OLS of the augmented system — exact, no MCMC.

Prior on B = [c, A_1 .. A_p]':
- own-lag means: A_l[i,i] = 1 (l=1), 0 (l>1) — a random walk
- tightness theta (lambda_1): cross-equation shrinkage 1/l
- lag decay lambda_2: variance of A_l ~ (theta*sigma_i/(l^lambda_2*sigma_j))^2
- own-sum-of-coefficients mu: additional shrinkage toward unit persistence
- covariance treated via per-equation residual variances sigma_i^2
  (independent equations — the standard BGR shortcut)

``bvar_fit`` returns the posterior-mean coefficient matrix, residual
covariance, and augmented design; ``bvar_irf`` gives impulse responses from
the companion form (recursive orthogonalized via Cholesky of Sigma_u);
``bvar_fevd`` decomposes h-step forecast-error variance; ``bvar_forecast``
iterates the companion matrix.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


def _check_panel(y: Array, p: int) -> tuple[Array, int, int]:
    yy = np.asarray(y, dtype=float)
    if yy.ndim != 2:
        raise ValueError("y must be (T, N)")
    t, n = yy.shape
    if not np.isfinite(yy).all():
        raise ValueError("non-finite input")
    if p < 1 or t <= p + n * p + 2:
        raise ValueError("need p >= 1 and enough observations")
    return yy, t, n


def _ols_moments(yy: Array, p: int) -> tuple[Array, Array]:
    """Equation-by-equation OLS residual variances for prior scaling."""
    t, n = yy.shape
    x = np.column_stack(
        [np.ones(t - p)] + [yy[p - lag_i - 1 : t - lag_i - 1] for lag_i in range(p)]
    )
    dep = yy[p:]
    beta = np.linalg.lstsq(x, dep, rcond=None)[0]
    resid = dep - x @ beta
    sigma2 = np.diag(resid.T @ resid) / max(t - p - x.shape[1], 1)
    return beta, np.clip(sigma2, 1e-12, None)


def _dummy_observations(
    yy: Array, p: int, theta: float, decay: float, mu: float
) -> tuple[Array, Array]:
    """Minnesota prior as per-coefficient pseudo-observations.

    For coefficient A_l[i,j] the prior is N(m_ijl, v_ijl^2) with
    m_ijl = 1 if l == 1 and i == j else 0 (own random walk) and
    v_ijl = theta * sigma_i / (l^decay * sigma_j).  The dummy row encodes
    |x_d b - y_d|^2 = sigma_i^2/v_ijl^2 * (b - m)^2, i.e.
    x_d = sigma_i/v = l^decay * sigma_j / theta, y_d = x_d * m.
    """
    _, n = yy.shape
    _, sigma2 = _ols_moments(yy, p)
    sigma = np.sqrt(sigma2)
    mu0 = yy[:p].mean(axis=0)
    k = 1 + n * p
    rows_y: list[Array] = []
    rows_x: list[Array] = []
    for lag in range(1, p + 1):
        y_blk = np.zeros((n * n, n))
        x_blk = np.zeros((n * n, k))
        for i in range(n):
            for j in range(n):
                r = i * n + j
                x = (lag**decay) * sigma[j] / theta
                x_blk[r, 1 + (lag - 1) * n + j] = x
                if lag == 1 and i == j:
                    y_blk[r, i] = x
        rows_y.append(y_blk)
        rows_x.append(x_blk)
    # sum-of-own-lags (persistence) block: penalize sum_l A_l[i,i] - 1 by mu
    if mu > 0:
        y_mu = np.diag(mu0 * mu)
        x_mu = np.zeros((n, k))
        x_mu[:, 1 : 1 + n * p] = np.tile(np.diag(mu0 * mu), (1, p))
        rows_y.append(y_mu)
        rows_x.append(x_mu)
    return np.vstack(rows_y), np.vstack(rows_x)


def bvar_fit(
    y: Array,
    p: int = 2,
    theta: float = 0.2,
    decay: float = 1.0,
    mu: float = 0.0,
) -> dict[str, Array | int]:
    """Posterior-mean BVAR: coefficients B (1+Np, N), resid cov Sigma (N, N).

    Prior scales follow each equation's OLS residual variance; theta <-> 0
    freezes the prior exactly, theta -> inf returns OLS.
    """
    yy, t, n = _check_panel(y, p)
    if not (theta > 0.0 and np.isfinite(theta)):
        raise ValueError("theta must be positive and finite")
    if not np.isfinite(decay) or decay < 0:
        raise ValueError("decay must be non-negative")
    if not np.isfinite(mu) or mu < 0:
        raise ValueError("mu must be non-negative")
    x = np.column_stack(
        [np.ones(t - p)] + [yy[p - lag_i - 1 : t - lag_i - 1] for lag_i in range(p)]
    )
    dep = yy[p:]
    y_d, x_d = _dummy_observations(yy, p, theta, decay, mu)
    xa = np.vstack([x, x_d])
    ya = np.vstack([dep, y_d])
    b_hat = np.linalg.solve(xa.T @ xa + 1e-10 * np.eye(xa.shape[1]), xa.T @ ya)
    resid = dep - x @ b_hat
    dof = max(t - p - x.shape[1], 1)
    sigma_u = (resid.T @ resid) / dof
    return {
        "B": b_hat,
        "sigma_u": sigma_u,
        "resid": resid,
        "x_aug": xa,
        "y_aug": ya,
        "n_obs": t - p,
    }


def _companion(b: Array, n: int, p: int) -> Array:
    """Companion form F of VAR(p) given B rows [const, A_1'...A_p']."""
    f = np.zeros((n * p, n * p))
    for lag in range(p):
        f[:n, lag * n : (lag + 1) * n] = b[1 + lag * n : 1 + (lag + 1) * n].T
    f[n:, :-n] = np.eye(n * (p - 1))
    return f


def bvar_irf(b: Array, sigma_u: Array, n_steps: int = 20, p: int = 2) -> Array:
    """Cholesky-orthogonalized IRFs: out[h, i, j] = response of i to shock j."""
    b = np.asarray(b, dtype=float)
    sigma_u = np.asarray(sigma_u, dtype=float)
    n = sigma_u.shape[0]
    if b.shape != (1 + n * p, n):
        raise ValueError("B shape inconsistent with (n, p)")
    if n_steps < 1:
        raise ValueError("n_steps must be >= 1")
    try:
        chol = np.linalg.cholesky(sigma_u + 1e-12 * np.eye(n))
    except np.linalg.LinAlgError:
        eig, q = np.linalg.eigh(sigma_u)
        chol = q @ np.diag(np.sqrt(np.clip(eig, 0, None))) @ q.T
        chol = np.tril(chol)
    f = _companion(b, n, p)
    out = np.zeros((n_steps, n, n))
    out[0] = chol
    fk = np.eye(n * p)
    sel = np.zeros((n, n * p))
    sel[:, :n] = np.eye(n)
    for h in range(1, n_steps):
        fk = fk @ f
        out[h] = sel @ fk @ sel.T @ chol
    return out


def bvar_fevd(b: Array, sigma_u: Array, n_steps: int = 20, p: int = 2) -> Array:
    """Fraction of h-step FEV of variable i attributed to shock j."""
    irf = bvar_irf(b, sigma_u, n_steps, p)
    num = np.cumsum(irf**2, axis=0)  # (H, i, j)
    total = num.sum(axis=2)
    total[total == 0] = np.nan
    return num / total[:, :, None]


def bvar_forecast(b: Array, history: Array, p: int, n_steps: int) -> Array:
    """Iterate the companion form; history is the last >= p rows of y."""
    hist = np.asarray(history, dtype=float)
    b = np.asarray(b, dtype=float)
    n = b.shape[1]
    if hist.shape[0] < p or hist.shape[1] != n:
        raise ValueError("history must be (>=p, N)")
    if n_steps < 1:
        raise ValueError("n_steps must be >= 1")
    f = _companion(b, n, p)
    const = np.zeros(n * p)
    const[:n] = b[0]
    state = np.concatenate([hist[-1 - lag_i] for lag_i in range(p)])
    out = np.zeros((n_steps, n))
    for h in range(n_steps):
        state = const + f @ state
        out[h] = state[:n]
    return out
