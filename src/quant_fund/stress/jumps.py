"""Merton jump-diffusion scenario returns.

The univariate path reuses :func:`quant_fund.models.jump_diffusion.merton_jump_simulate`
and is checked against :func:`quant_fund.models.jump_diffusion.merton_log_moments`.
The multi-asset generator uses a correlated Brownian shock and idiosyncratic
compound-Poisson jumps, so each margin has the same log moments as the
univariate model.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

from quant_fund.models.jump_diffusion import merton_jump_simulate, merton_kappa, merton_log_moments

Array = NDArray[np.float64]


def merton_scenario_log_returns(
    *,
    tenor: float,
    rate: float,
    sigma: float,
    lam: float,
    mu_j: float,
    s_j: float,
    n_scenarios: int,
    rng: np.random.Generator,
) -> Array:
    """One-period log returns from the existing Merton simulator."""
    if isinstance(n_scenarios, bool) or not isinstance(n_scenarios, int) or n_scenarios < 1:
        raise ValueError("n_scenarios must be a positive integer")
    paths = merton_jump_simulate(
        spot=1.0,
        tenor=tenor,
        rate=rate,
        sigma=sigma,
        lam=lam,
        mu_j=mu_j,
        s_j=s_j,
        n_steps=1,
        n_paths=n_scenarios,
        rng=rng,
    )
    return np.asarray(np.log(paths[:, -1] / paths[:, 0]), dtype=np.float64)


def margin_log_moments(
    *,
    tenor: float,
    rate: float,
    sigma: float,
    lam: float,
    mu_j: float,
    s_j: float,
) -> dict[str, float]:
    """Analytic Merton log moments from the existing implementation."""
    return merton_log_moments(tenor, rate, sigma, lam, mu_j, s_j)


def correlated_merton_log_returns(
    sigmas: Array,
    corr: Array,
    *,
    tenor: float,
    rate: float,
    lam: float,
    mu_j: float,
    s_j: float,
    n_scenarios: int,
    rng: np.random.Generator,
) -> Array:
    """Multi-asset Merton log returns with correlated diffusion and idiosyncratic jumps.

    Jump parameters are shared across assets. Brownian correlation is ``corr``.
    Jumps are independent across assets, so the off-diagonal covariance is the
    diffusion term ``corr_ij * sigma_i * sigma_j * tenor``. Jump variance is
    added only on the diagonal. See :func:`idiosyncratic_jump_covariance`.
    """
    sigma = np.asarray(sigmas, dtype=float).reshape(-1)
    matrix = np.asarray(corr, dtype=float)
    dim = sigma.size
    if matrix.shape != (dim, dim):
        raise ValueError("corr must match the number of assets")
    if np.any(sigma <= 0.0) or not np.isfinite(sigma).all():
        raise ValueError("sigmas must be finite and positive")
    if tenor <= 0.0 or lam < 0.0 or s_j <= 0.0:
        raise ValueError("tenor > 0, lam >= 0, s_j > 0")
    if isinstance(n_scenarios, bool) or not isinstance(n_scenarios, int) or n_scenarios < 1:
        raise ValueError("n_scenarios must be a positive integer")
    kappa = merton_kappa(mu_j, s_j)
    chol = np.linalg.cholesky(matrix)
    gaussian = rng.standard_normal((n_scenarios, dim)) @ chol.T
    log_returns = np.empty((n_scenarios, dim), dtype=np.float64)
    for j in range(dim):
        drift = (rate - lam * kappa - 0.5 * sigma[j] ** 2) * tenor
        counts = rng.poisson(lam * tenor, size=n_scenarios)
        jumps = np.zeros(n_scenarios, dtype=float)
        active = counts > 0
        if np.any(active):
            jumps[active] = rng.normal(counts[active] * mu_j, np.sqrt(counts[active]) * s_j)
        log_returns[:, j] = drift + sigma[j] * math.sqrt(tenor) * gaussian[:, j] + jumps
    return log_returns


def idiosyncratic_jump_covariance(
    sigmas: Array,
    corr: Array,
    *,
    tenor: float,
    lam: float,
    mu_j: float,
    s_j: float,
) -> Array:
    """Covariance of multi-asset Merton log returns with independent jumps."""
    sigma = np.asarray(sigmas, dtype=float).reshape(-1)
    matrix = np.asarray(corr, dtype=float)
    jump_var = lam * tenor * (mu_j**2 + s_j**2)
    cov = tenor * (sigma[:, None] * matrix * sigma[None, :])
    cov = cov + np.diag(np.full(sigma.size, jump_var))
    return np.asarray(cov, dtype=np.float64)


def calibrate_merton_to_moments(
    returns: Array,
    *,
    jump_mean: float = -0.02,
    jump_vol: float = 0.015,
) -> dict[str, float]:
    """Match sample mean and variance with a restricted Merton specification.

    Jump mean and jump volatility are fixed inputs. Jump intensity absorbs up
    to half of the sample variance when excess kurtosis is positive, and is
    zero otherwise. The rate is then set so the analytic log mean matches the
    sample mean. Skewness is not matched.
    """
    r = np.asarray(returns, dtype=float).reshape(-1)
    r = r[np.isfinite(r)]
    if r.size < 30:
        raise ValueError("Merton calibration needs at least 30 finite returns")
    if not np.isfinite(jump_mean) or not np.isfinite(jump_vol) or jump_vol <= 0.0:
        raise ValueError("jump_mean must be finite and jump_vol positive")
    mean = float(r.mean())
    var = float(r.var(ddof=1))
    if var <= 0.0:
        raise ValueError("Merton calibration needs positive variance")
    centered = r - mean
    excess = float(np.mean(centered**4) / var**2 - 3.0)
    jump_second = jump_mean**2 + jump_vol**2
    if excess <= 0.0:
        lam = 0.0
    else:
        share = min(0.5, excess / (excess + 3.0))
        lam = share * var / jump_second
    sigma2 = var - lam * jump_second
    if sigma2 <= 1e-12:
        raise ValueError("jump specification absorbs the whole variance")
    sigma = math.sqrt(sigma2)
    kappa = merton_kappa(jump_mean, jump_vol)
    # mean = rate - lam*kappa - 0.5*sigma^2 + lam*mu_j   (tenor = 1)
    rate = mean + lam * kappa + 0.5 * sigma2 - lam * jump_mean
    analytic = merton_log_moments(1.0, rate, sigma, lam, jump_mean, jump_vol)
    return {
        "tenor": 1.0,
        "rate": float(rate),
        "sigma": float(sigma),
        "lam": float(lam),
        "mu_j": float(jump_mean),
        "s_j": float(jump_vol),
        "sample_mean": mean,
        "sample_var": var,
        "excess_kurtosis": excess,
        "analytic_mean": float(analytic["mean"]),
        "analytic_var": float(analytic["var"]),
    }
