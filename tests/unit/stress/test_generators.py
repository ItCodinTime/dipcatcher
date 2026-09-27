"""Moment and dependence checks for the scenario generators.

Draws are SYNTHETIC correctness checks. Tolerances are sampling bounds
around the analytic target, not fitted market benchmarks.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import stats

from quant_fund.models.copula import kendall_tau
from quant_fund.stress.bootstrap import (
    path_moments,
    resolve_block_length,
    stationary_bootstrap_paths,
)
from quant_fund.stress.garch_copula import (
    fit_variance_targeted_garch11,
    garch11_unconditional_variance,
    residual_dependence,
    simulate_c_vine_t,
    simulate_garch11,
    simulate_garch_t_copula,
    simulate_t_copula,
    t_copula_kendall_tau,
    t_copula_tail_dependence,
    t_h_function,
    t_h_inverse,
)
from quant_fund.stress.jumps import (
    calibrate_merton_to_moments,
    correlated_merton_log_returns,
    idiosyncratic_jump_covariance,
    margin_log_moments,
    merton_scenario_log_returns,
)
from quant_fund.stress.regimes import (
    GaussianRegimeSpec,
    simulate_gaussian_hmm,
    spec_from_univariate_fit,
    stationary_distribution,
    unconditional_moments,
)

pytestmark = pytest.mark.synthetic


def _lag1(series: np.ndarray) -> float:
    y = np.asarray(series, dtype=float)
    y = y - y.mean()
    denom = float(y @ y)
    if denom == 0.0:
        return 0.0
    return float(y[:-1] @ y[1:] / denom)


def test_stationary_bootstrap_keeps_dependence_and_cross_section() -> None:
    rng = np.random.default_rng(0)
    n = 600
    eps = rng.normal(size=n)
    ar = np.zeros(n)
    for t in range(1, n):
        ar[t] = 0.8 * ar[t - 1] + eps[t]
    paths = stationary_bootstrap_paths(ar, n_paths=40, horizon=n, mean_block=25.0, seed=1)
    assert paths.shape == (40, n, 1)
    boot_acf = float(np.mean([_lag1(paths[i, :, 0]) for i in range(paths.shape[0])]))
    iid = ar[rng.integers(0, n, size=(40, n))]
    iid_acf = float(np.mean([_lag1(iid[i]) for i in range(iid.shape[0])]))
    assert boot_acf > 0.45
    assert abs(iid_acf) < 0.12

    z = rng.normal(size=(400, 2))
    z[:, 1] = 0.75 * z[:, 0] + math.sqrt(1.0 - 0.75**2) * z[:, 1]
    panel_paths = stationary_bootstrap_paths(z, n_paths=30, horizon=200, mean_block=8.0, seed=2)
    pooled = path_moments(panel_paths)
    sample_corr = float(np.corrcoef(z, rowvar=False)[0, 1])
    boot_corr = float(pooled["cov"][0, 1] / math.sqrt(pooled["cov"][0, 0] * pooled["cov"][1, 1]))
    assert boot_corr == pytest.approx(sample_corr, abs=0.08)
    assert resolve_block_length(np.ones((30, 1)), None) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        stationary_bootstrap_paths(ar, n_paths=1, horizon=n + 1)
    with pytest.raises(ValueError):
        stationary_bootstrap_paths(ar, n_paths=0)


def test_garch_variance_and_t_copula_moments() -> None:
    omega, alpha, beta = 1.0e-6, 0.05, 0.50
    target = garch11_unconditional_variance(omega, alpha, beta)
    rng = np.random.default_rng(3)
    innovations = rng.normal(size=8000)
    returns, sigma2 = simulate_garch11(omega, alpha, beta, innovations)
    assert float(sigma2.mean()) == pytest.approx(target, rel=0.05)
    assert float(returns.var(ddof=1)) == pytest.approx(target, rel=0.08)
    with pytest.raises(ValueError):
        garch11_unconditional_variance(1.0, 0.5, 0.6)

    rho = 0.55
    nu = 8.0
    z, uniforms = simulate_t_copula(np.array([[1.0, rho], [rho, 1.0]]), nu, 6000, rng)
    assert z.var(axis=0, ddof=1) == pytest.approx(1.0, rel=0.08)
    assert uniforms.mean(axis=0) == pytest.approx(0.5, abs=0.02)
    assert uniforms.var(axis=0, ddof=1) == pytest.approx(1.0 / 12.0, rel=0.08)
    tau = kendall_tau(uniforms)
    assert tau == pytest.approx(t_copula_kendall_tau(rho), abs=0.04)
    assert t_copula_kendall_tau(rho) == pytest.approx((2.0 / math.pi) * math.asin(rho))


def test_t_copula_tail_dependence_formula_and_monotone_limits() -> None:
    rho, nu = 0.4, 5.0
    arg = -math.sqrt((nu + 1.0) * (1.0 - rho) / (1.0 + rho))
    assert t_copula_tail_dependence(rho, nu) == pytest.approx(2.0 * stats.t.cdf(arg, df=nu + 1.0))
    rhos = (-0.5, 0.0, 0.5, 0.9)
    lambdas = [t_copula_tail_dependence(value, 4.0) for value in rhos]
    assert lambdas == sorted(lambdas)
    nus = (3.0, 6.0, 30.0)
    by_nu = [t_copula_tail_dependence(0.5, value) for value in nus]
    assert by_nu == sorted(by_nu, reverse=True)
    assert t_copula_tail_dependence(0.5, 400.0) < 0.01
    assert t_copula_tail_dependence(0.999, 4.0) > 0.8
    with pytest.raises(ValueError):
        t_copula_tail_dependence(1.0, 5.0)


def test_c_vine_first_pair_and_h_inverse() -> None:
    rng = np.random.default_rng(4)
    rho, nu = 0.45, 6.0
    w = rng.uniform(0.05, 0.95, size=40)
    v = rng.uniform(0.05, 0.95, size=40)
    recovered = t_h_function(t_h_inverse(w, v, rho, nu), v, rho, nu)
    assert recovered == pytest.approx(w, abs=1e-6)
    rhos = np.array([[1.0, rho, 0.2], [rho, 1.0, 0.3], [0.2, 0.3, 1.0]])
    nus = np.array([[0.0, nu, 7.0], [nu, 0.0, 8.0], [7.0, 8.0, 0.0]])
    uniforms = simulate_c_vine_t(rhos, nus, 6000, rng)
    assert kendall_tau(uniforms[:, :2]) == pytest.approx(t_copula_kendall_tau(rho), abs=0.04)
    with pytest.raises(ValueError):
        simulate_c_vine_t(np.eye(1), np.eye(1), 10, rng)


def test_garch_t_copula_matches_unconditional_variance() -> None:
    rng = np.random.default_rng(5)
    omegas = np.array([2.0e-6, 4.0e-6])
    alphas = np.array([0.04, 0.06])
    betas = np.array([0.60, 0.50])
    corr = np.array([[1.0, 0.3], [0.3, 1.0]])
    draws = simulate_garch_t_copula(omegas, alphas, betas, corr, 10.0, 7000, rng)
    for j in range(2):
        target = garch11_unconditional_variance(float(omegas[j]), float(alphas[j]), float(betas[j]))
        assert float(draws[:, j].var(ddof=1)) == pytest.approx(target, rel=0.12)
    fit = fit_variance_targeted_garch11(draws[:, 0])
    assert fit["unconditional_variance"] == pytest.approx(
        float(np.mean((draws[:, 0] - draws[:, 0].mean()) ** 2))
    )
    assert fit["alpha"] + fit["beta"] < 1.0
    flat = {"omega": 1.0, "alpha": 0.0, "beta": 0.0}
    noise = rng.normal(size=(5000, 2)) @ np.linalg.cholesky(corr).T
    corr_hat, nu_hat = residual_dependence(noise, [flat, flat])
    assert np.allclose(np.diag(corr_hat), 1.0)
    assert corr_hat[0, 1] == pytest.approx(0.3, abs=0.06)
    assert nu_hat >= 4.1


def test_hmm_matches_stationary_moments() -> None:
    means = np.array([[-0.01], [0.02]])
    covs = np.array([[[1.0e-4]], [[4.0e-4]]])
    transition = np.array([[0.9, 0.1], [0.2, 0.8]])
    spec = GaussianRegimeSpec(means=means, covs=covs, transition=transition)
    pi = stationary_distribution(transition)
    assert pi == pytest.approx([2.0 / 3.0, 1.0 / 3.0])
    analytic = unconditional_moments(spec)
    rng = np.random.default_rng(6)
    drawn, states = simulate_gaussian_hmm(spec, 12000, rng)
    se = math.sqrt(float(analytic["cov"][0, 0]) / drawn.shape[0])
    assert float(drawn.mean()) == pytest.approx(float(analytic["mean"][0]), abs=4.0 * se)
    assert float(drawn.var(ddof=1)) == pytest.approx(float(analytic["cov"][0, 0]), rel=0.12)
    freq = np.bincount(states, minlength=2) / states.size
    assert freq == pytest.approx(analytic["pi"], abs=0.03)
    wrapped = spec_from_univariate_fit(
        {"means": np.array([-0.01, 0.02]), "sigmas": np.array([0.01, 0.02]), "P": transition}
    )
    assert wrapped.means.shape == (2, 1)
    with pytest.raises(ValueError):
        GaussianRegimeSpec(means=means, covs=covs, transition=np.array([[0.5, 0.5], [0.5, 0.4]]))


def test_merton_margins_correlation_and_calibration() -> None:
    rng = np.random.default_rng(7)
    params = {"tenor": 1.0, "rate": 0.01, "sigma": 0.2, "lam": 1.5, "mu_j": -0.02, "s_j": 0.04}
    analytic = margin_log_moments(**params)
    n = 16000
    drawn = merton_scenario_log_returns(**params, n_scenarios=n, rng=rng)
    se = math.sqrt(analytic["var"] / n)
    assert float(drawn.mean()) == pytest.approx(analytic["mean"], abs=4.0 * se)
    assert float(drawn.var(ddof=1)) == pytest.approx(analytic["var"], rel=0.08)

    sigmas = np.array([0.2, 0.3])
    corr = np.array([[1.0, 0.6], [0.6, 1.0]])
    quiet = correlated_merton_log_returns(
        sigmas,
        corr,
        tenor=1.0,
        rate=0.01,
        lam=0.0,
        mu_j=-0.02,
        s_j=0.015,
        n_scenarios=8000,
        rng=rng,
    )
    assert float(np.corrcoef(quiet, rowvar=False)[0, 1]) == pytest.approx(0.6, abs=0.04)
    jumped = correlated_merton_log_returns(
        sigmas,
        corr,
        tenor=1.0,
        rate=0.01,
        lam=2.0,
        mu_j=-0.02,
        s_j=0.03,
        n_scenarios=12000,
        rng=rng,
    )
    target = idiosyncratic_jump_covariance(sigmas, corr, tenor=1.0, lam=2.0, mu_j=-0.02, s_j=0.03)
    sample = np.cov(jumped, rowvar=False, ddof=1)
    assert sample == pytest.approx(target, rel=0.1, abs=1e-4)
    assert target[0, 1] == pytest.approx(1.0 * 0.2 * 0.6 * 0.3)
    assert target[0, 0] > 1.0 * 0.2**2

    flat = rng.uniform(-0.01, 0.01, size=800)
    calm = calibrate_merton_to_moments(flat)
    assert calm["excess_kurtosis"] < 0.0
    assert calm["lam"] == 0.0
    assert calm["analytic_mean"] == pytest.approx(calm["sample_mean"], abs=1e-10)
    assert calm["analytic_var"] == pytest.approx(calm["sample_var"], rel=1e-8)
    heavy = rng.standard_t(5, size=2000) * 0.01
    fitted = calibrate_merton_to_moments(heavy)
    assert fitted["lam"] > 0.0
    assert fitted["analytic_mean"] == pytest.approx(fitted["sample_mean"], abs=1e-10)
    assert fitted["analytic_var"] == pytest.approx(fitted["sample_var"], rel=1e-8)
    with pytest.raises(ValueError):
        calibrate_merton_to_moments(np.zeros(10))
