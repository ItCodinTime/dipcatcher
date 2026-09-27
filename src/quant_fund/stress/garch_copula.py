"""GARCH(1,1) scenarios with a t copula or a t-copula C-vine.

The t-copula tail-dependence coefficient is the Demarta–McNeil (2005) formula.
The C-vine sampler follows the pair-copula recursion in Aas, Czado, Frigessi
and Bakken (2009). GARCH conditional variances use the standard recursion

    σ²_t = ω + α r²_{t-1} + β σ²_{t-1},

with unconditional variance ω / (1 − α − β) when α + β < 1 and innovations have
unit variance.

These are scenario generators for research. They are not a trading signal.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray
from scipy import stats

Array = NDArray[np.float64]


def garch11_unconditional_variance(omega: float, alpha: float, beta: float) -> float:
    """Stationary variance of a GARCH(1,1) with unit-variance innovations."""
    for name, value in (("omega", omega), ("alpha", alpha), ("beta", beta)):
        if not np.isfinite(value):
            raise ValueError(f"{name} must be finite")
    if omega <= 0.0 or alpha < 0.0 or beta < 0.0 or alpha + beta >= 1.0:
        raise ValueError("need omega > 0, alpha >= 0, beta >= 0, alpha + beta < 1")
    return float(omega / (1.0 - alpha - beta))


def simulate_garch11(
    omega: float,
    alpha: float,
    beta: float,
    innovations: Array,
    *,
    sigma0: float | None = None,
) -> tuple[Array, Array]:
    """Filter or simulate one GARCH(1,1) path. Returns ``(returns, sigma2)``."""
    _ = garch11_unconditional_variance(omega, alpha, beta)
    z = np.asarray(innovations, dtype=float).reshape(-1)
    if z.size < 1 or not np.isfinite(z).all():
        raise ValueError("innovations must be a non-empty finite vector")
    unconditional = garch11_unconditional_variance(omega, alpha, beta)
    prev_s = unconditional if sigma0 is None else float(sigma0) ** 2
    if not np.isfinite(prev_s) or prev_s <= 0.0:
        raise ValueError("sigma0 must be positive")
    prev_r2 = prev_s
    returns = np.empty(z.size, dtype=np.float64)
    sigma2 = np.empty(z.size, dtype=np.float64)
    for t in range(z.size):
        sigma2[t] = omega + alpha * prev_r2 + beta * prev_s
        if sigma2[t] <= 0.0:
            raise ValueError("GARCH variance became non-positive")
        returns[t] = math.sqrt(float(sigma2[t])) * float(z[t])
        prev_r2 = float(returns[t] ** 2)
        prev_s = float(sigma2[t])
    return returns, sigma2


def t_copula_tail_dependence(rho: float, nu: float) -> float:
    """Upper and lower tail dependence of the bivariate t copula.

    λ = 2 t_{ν+1}( −√((ν+1)(1−ρ)/(1+ρ)) ). Gaussian copulas are the ν → ∞ limit,
    where λ → 0 for ρ < 1.
    """
    if not np.isfinite(rho) or not -1.0 < float(rho) < 1.0:
        raise ValueError("rho must be in (-1, 1)")
    if not np.isfinite(nu) or float(nu) <= 2.0:
        raise ValueError("nu must be finite and > 2")
    arg = -math.sqrt((nu + 1.0) * (1.0 - rho) / (1.0 + rho))
    return float(2.0 * stats.t.cdf(arg, df=nu + 1.0))


def t_copula_kendall_tau(rho: float) -> float:
    """Kendall tau of an elliptical copula: (2/π) arcsin(ρ)."""
    if not np.isfinite(rho) or not -1.0 < float(rho) < 1.0:
        raise ValueError("rho must be in (-1, 1)")
    return float((2.0 / math.pi) * math.asin(rho))


def _as_correlation(corr: Array) -> Array:
    matrix = np.asarray(corr, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] < 2:
        raise ValueError("correlation must be a d x d matrix with d >= 2")
    if not np.isfinite(matrix).all():
        raise ValueError("correlation must be finite")
    if not np.allclose(matrix, matrix.T, atol=1e-8):
        raise ValueError("correlation must be symmetric")
    if not np.allclose(np.diag(matrix), 1.0, atol=1e-6):
        raise ValueError("correlation diagonal must be 1")
    smallest = float(np.linalg.eigvalsh(matrix)[0])
    if smallest <= 1e-10:
        raise ValueError("correlation must be positive definite")
    return np.asarray(matrix, dtype=np.float64)


def simulate_t_copula(
    corr: Array,
    nu: float,
    n: int,
    rng: np.random.Generator,
) -> tuple[Array, Array]:
    """Draw standardized t-copula innovations and the copula uniforms.

    The returned innovations have unit marginal variance (scaled by
    √((ν−2)/ν)). Uniforms are the t CDF of the unscaled draws.
    """
    matrix = _as_correlation(corr)
    if not np.isfinite(nu) or nu <= 2.0:
        raise ValueError("nu must be > 2")
    if isinstance(n, bool) or not isinstance(n, int) or n < 1:
        raise ValueError("n must be a positive integer")
    chol = np.linalg.cholesky(matrix)
    gaussian = rng.standard_normal((n, matrix.shape[0])) @ chol.T
    weight = rng.chisquare(nu, size=n) / nu
    raw = gaussian / np.sqrt(weight)[:, None]
    scale = math.sqrt((nu - 2.0) / nu)
    innovations = np.asarray(raw * scale, dtype=np.float64)
    uniforms = np.asarray(stats.t.cdf(raw, df=nu), dtype=np.float64)
    return innovations, uniforms


def _clip_unit(u: Array) -> Array:
    return np.asarray(np.clip(np.asarray(u, dtype=float), 1e-10, 1.0 - 1e-10), dtype=np.float64)


def t_h_function(u: Array, v: Array, rho: float, nu: float) -> Array:
    """Conditional CDF h(u | v) of the bivariate t copula."""
    if not np.isfinite(rho) or abs(rho) >= 1.0 or not np.isfinite(nu) or nu <= 2.0:
        raise ValueError("rho must be in (-1, 1) and nu > 2")
    uu = _clip_unit(u)
    vv = _clip_unit(v)
    x = stats.t.ppf(uu, df=nu)
    y = stats.t.ppf(vv, df=nu)
    scale = np.sqrt((nu + y**2) * (1.0 - rho**2) / (nu + 1.0))
    return np.asarray(stats.t.cdf((x - rho * y) / scale, df=nu + 1.0), dtype=np.float64)


def t_h_inverse(w: Array, v: Array, rho: float, nu: float) -> Array:
    """Inverse h-function of the bivariate t copula."""
    if not np.isfinite(rho) or abs(rho) >= 1.0 or not np.isfinite(nu) or nu <= 2.0:
        raise ValueError("rho must be in (-1, 1) and nu > 2")
    ww = _clip_unit(w)
    vv = _clip_unit(v)
    y = stats.t.ppf(vv, df=nu)
    z = stats.t.ppf(ww, df=nu + 1.0)
    scale = np.sqrt((nu + y**2) * (1.0 - rho**2) / (nu + 1.0))
    x = rho * y + z * scale
    return np.asarray(stats.t.cdf(x, df=nu), dtype=np.float64)


def simulate_c_vine_t(
    rhos: Array,
    nus: Array,
    n: int,
    rng: np.random.Generator,
) -> Array:
    """Simulate uniforms from a C-vine of bivariate t pair-copulas.

    ``rhos[j, i]`` and ``nus[j, i]`` for ``j < i`` are the pair-copula
    parameters used when variable ``i`` is conditioned on variable ``j`` in the
    Aas–Czado–Frigessi–Bakken sampling recursion. The unconditional copula of
    the first two coordinates is exactly the pair ``(0, 1)``.
    """
    rho = np.asarray(rhos, dtype=float)
    nu = np.asarray(nus, dtype=float)
    if rho.shape != nu.shape or rho.ndim != 2 or rho.shape[0] != rho.shape[1]:
        raise ValueError("rhos and nus must be square and the same shape")
    dim = rho.shape[0]
    if dim < 2:
        raise ValueError("C-vine dimension must be >= 2")
    if isinstance(n, bool) or not isinstance(n, int) or n < 1:
        raise ValueError("n must be a positive integer")
    independent = rng.random((n, dim))
    uniforms = np.zeros((n, dim), dtype=np.float64)
    uniforms[:, 0] = independent[:, 0]
    for i in range(1, dim):
        value = independent[:, i].copy()
        for j in range(i - 1, -1, -1):
            value = t_h_inverse(value, uniforms[:, j], float(rho[j, i]), float(nu[j, i]))
        uniforms[:, i] = value
    return uniforms


def simulate_garch_t_copula(
    omegas: Array,
    alphas: Array,
    betas: Array,
    corr: Array,
    nu: float,
    n: int,
    rng: np.random.Generator,
) -> Array:
    """Multi-asset GARCH(1,1) paths whose innovations follow a t copula.

    Shape ``(n, d)``. Each margin's unconditional variance is ω/(1−α−β).
    """
    omega = np.asarray(omegas, dtype=float).reshape(-1)
    alpha = np.asarray(alphas, dtype=float).reshape(-1)
    beta = np.asarray(betas, dtype=float).reshape(-1)
    matrix = _as_correlation(corr)
    dim = omega.size
    if not (alpha.size == beta.size == dim == matrix.shape[0]):
        raise ValueError("GARCH parameters and correlation dimension must match")
    innovations, _uniforms = simulate_t_copula(matrix, nu, n, rng)
    out = np.empty((n, dim), dtype=np.float64)
    for j in range(dim):
        returns, _sigma2 = simulate_garch11(
            float(omega[j]), float(alpha[j]), float(beta[j]), innovations[:, j]
        )
        out[:, j] = returns
    return out


def fit_variance_targeted_garch11(returns: Array) -> dict[str, float]:
    """Gaussian variance-targeted GARCH(1,1) on a coarse (α, β) grid.

    ω is set so the unconditional variance equals the sample second moment.
    This is a scenario calibration, not a claim that the grid is the MLE.
    """
    r = np.asarray(returns, dtype=float).reshape(-1)
    r = r[np.isfinite(r)]
    if r.size < 30:
        raise ValueError("GARCH calibration needs at least 30 finite returns")
    centered = r - float(r.mean())
    sample_var = float(np.mean(centered**2))
    if sample_var <= 0.0:
        raise ValueError("GARCH calibration needs positive variance")
    best_ll = -np.inf
    best = (0.05, 0.90)
    for alpha in np.linspace(0.02, 0.20, 10):
        for beta in np.linspace(0.70, 0.97, 10):
            if alpha + beta >= 0.995:
                continue
            omega = sample_var * (1.0 - alpha - beta)
            ll = _gaussian_garch_ll(centered, omega, float(alpha), float(beta))
            if ll > best_ll:
                best_ll = ll
                best = (float(alpha), float(beta))
    alpha_hat, beta_hat = best
    omega_hat = sample_var * (1.0 - alpha_hat - beta_hat)
    return {
        "omega": float(omega_hat),
        "alpha": alpha_hat,
        "beta": beta_hat,
        "unconditional_variance": float(sample_var),
        "loglik": float(best_ll),
    }


def _gaussian_garch_ll(returns: Array, omega: float, alpha: float, beta: float) -> float:
    prev_s = float(np.mean(np.asarray(returns, dtype=float) ** 2))
    prev_r2 = prev_s
    ll = 0.0
    for value in np.asarray(returns, dtype=float):
        sigma2 = omega + alpha * prev_r2 + beta * prev_s
        if sigma2 <= 0.0:
            return -np.inf
        ll += -0.5 * (math.log(2.0 * math.pi) + math.log(sigma2) + float(value) ** 2 / sigma2)
        prev_r2 = float(value) ** 2
        prev_s = sigma2
    return float(ll)


def residual_dependence(returns: Array, fits: list[dict[str, float]]) -> tuple[Array, float]:
    """Correlation of GARCH residuals and a moment-based t degrees of freedom.

    Excess kurtosis κ maps to ν = 4 + 6/κ when κ > 0.1, otherwise ν = 30
    (near-Gaussian). The correlation is the residual correlation.
    """
    panel = np.asarray(returns, dtype=float)
    if panel.ndim != 2 or panel.shape[1] != len(fits):
        raise ValueError("fits must match the panel's column count")
    residual = np.empty_like(panel, dtype=float)
    for j, fit in enumerate(fits):
        sigma = _filter_sigma(panel[:, j], fit["omega"], fit["alpha"], fit["beta"])
        residual[:, j] = panel[:, j] / sigma
    corr = np.corrcoef(residual, rowvar=False)
    corr = np.asarray(0.5 * (corr + corr.T), dtype=np.float64)
    np.fill_diagonal(corr, 1.0)
    centered = residual - residual.mean(axis=0, keepdims=True)
    fourth = np.mean(centered**4, axis=0)
    second = np.mean(centered**2, axis=0)
    excess = float(np.median(fourth / np.maximum(second**2, 1e-18) - 3.0))
    nu = 30.0 if excess <= 0.1 else float(min(30.0, max(4.1, 4.0 + 6.0 / excess)))
    return corr, nu


def _filter_sigma(returns: Array, omega: float, alpha: float, beta: float) -> Array:
    r = np.asarray(returns, dtype=float).reshape(-1)
    sigma = np.empty(r.size, dtype=np.float64)
    prev_s = float(np.mean(r**2))
    prev_r2 = prev_s
    for t, value in enumerate(r):
        sigma2 = omega + alpha * prev_r2 + beta * prev_s
        sigma[t] = math.sqrt(max(sigma2, 1e-18))
        prev_r2 = float(value) ** 2
        prev_s = sigma2
    return sigma
