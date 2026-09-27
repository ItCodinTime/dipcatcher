"""Larger Monte Carlo checks. Marked slow so the smoke job can skip them."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import stats

from quant_fund.models.copula import empirical_tail_dependence
from quant_fund.stress.garch_copula import (
    garch11_unconditional_variance,
    simulate_garch11,
    simulate_t_copula,
    t_copula_tail_dependence,
)
from quant_fund.stress.jumps import margin_log_moments, merton_scenario_log_returns
from quant_fund.stress.regimes import (
    GaussianRegimeSpec,
    simulate_gaussian_hmm,
    unconditional_moments,
)

pytestmark = [pytest.mark.slow, pytest.mark.synthetic]


def _finite_threshold_tail(rho: float, nu: float, p: float, n: int = 250_000) -> float:
    """P(X2 <= q | X1 <= q) for a bivariate t, with q = t_ν^{-1}(p).

    This is the finite-threshold conditional probability. It sits above the
    Demarta–McNeil limit and is the quantity the empirical estimator targets.
    """
    corr = np.array([[1.0, rho], [rho, 1.0]])
    raw = stats.multivariate_t.rvs(shape=corr, df=nu, size=n, random_state=21)
    q = float(stats.t.ppf(p, df=nu))
    cond = raw[:, 0] <= q
    return float(np.mean(raw[cond, 1] <= q))


def test_empirical_t_copula_tail_dependence() -> None:
    rng = np.random.default_rng(11)
    rho, nu = 0.6, 5.0
    n = 30_000
    k = 900
    _z, uniforms = simulate_t_copula(np.array([[1.0, rho], [rho, 1.0]]), nu, n, rng)
    lower, upper = empirical_tail_dependence(uniforms, k=k)
    p = (k + 1) / n
    finite = _finite_threshold_tail(rho, nu, p)
    limit = t_copula_tail_dependence(rho, nu)
    se = math.sqrt(finite * (1.0 - finite) / (n * p))
    assert finite > limit
    assert lower == pytest.approx(finite, abs=4.0 * se)
    assert upper == pytest.approx(finite, abs=4.0 * se)
    assert lower > p * 2.0


def test_long_garch_variance() -> None:
    omega, alpha, beta = 2.0e-6, 0.05, 0.70
    target = garch11_unconditional_variance(omega, alpha, beta)
    returns, sigma2 = simulate_garch11(
        omega, alpha, beta, np.random.default_rng(12).normal(size=20000)
    )
    assert float(sigma2.mean()) == pytest.approx(target, rel=0.03)
    assert float(returns.var(ddof=1)) == pytest.approx(target, rel=0.05)


def test_hmm_moments_tighter_sample() -> None:
    means = np.array([[0.0, 0.0], [0.01, -0.02]])
    covs = np.array(
        [
            [[1.0e-4, 2.0e-5], [2.0e-5, 1.0e-4]],
            [[2.0e-4, -1.0e-5], [-1.0e-5, 2.0e-4]],
        ]
    )
    transition = np.array([[0.95, 0.05], [0.10, 0.90]])
    spec = GaussianRegimeSpec(means=means, covs=covs, transition=transition)
    analytic = unconditional_moments(spec)
    drawn, _states = simulate_gaussian_hmm(spec, 25000, np.random.default_rng(13))
    assert drawn.mean(axis=0) == pytest.approx(analytic["mean"], abs=0.0015)
    assert np.cov(drawn, rowvar=False, ddof=1) == pytest.approx(analytic["cov"], rel=0.12, abs=2e-5)


def test_merton_skew_matches_analytic() -> None:
    params = {"tenor": 1.0, "rate": 0.0, "sigma": 0.15, "lam": 3.0, "mu_j": -0.03, "s_j": 0.02}
    analytic = margin_log_moments(**params)
    drawn = merton_scenario_log_returns(**params, n_scenarios=40000, rng=np.random.default_rng(14))
    sample_skew = float(stats_skew(drawn))
    se = math.sqrt(6.0 / drawn.size)
    assert sample_skew == pytest.approx(analytic["skew"], abs=max(0.08, 8.0 * se))


def stats_skew(values: np.ndarray) -> float:
    x = np.asarray(values, dtype=float)
    z = (x - x.mean()) / x.std(ddof=1)
    return float(np.mean(z**3))
