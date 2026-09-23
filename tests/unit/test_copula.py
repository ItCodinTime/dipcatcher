"""Tests for copula module."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats as sstats

from quant_fund.models.copula import (
    clayton_copula_sim,
    copula_tail_dependence_theory,
    empirical_tail_dependence,
    fit_clayton,
    fit_gaussian_copula,
    fit_gumbel,
    fit_t_copula,
    gaussian_copula_sim,
    gumbel_copula_sim,
    kendall_tau,
    pseudo_observations,
    t_copula_sim,
)


class TestTransforms:
    def test_pseudo_observations_uniform(self):
        rng = np.random.default_rng(0)
        u = pseudo_observations(rng.normal(size=(500, 3)))
        assert u.shape == (500, 3)
        assert np.all(u > 0.0) and np.all(u < 1.0)
        # Uniform margins -> each column ~ Uniform.
        ks = sstats.kstest(u[:, 0], "uniform").pvalue
        assert ks > 0.01

    def test_kendall_tau(self):
        rng = np.random.default_rng(1)
        x = rng.normal(size=300)
        y = 0.9 * x + np.sqrt(1 - 0.81) * rng.normal(size=300)
        u = pseudo_observations(np.column_stack([x, y]))
        tau = kendall_tau(u)
        assert tau > 0.5

    def test_failclosed(self):
        with pytest.raises(ValueError):
            pseudo_observations(np.ones((5, 4)))
        with pytest.raises(ValueError):
            kendall_tau(np.ones((20, 2)) * 1.5)  # outside [0,1]
        with pytest.raises(ValueError):
            empirical_tail_dependence(np.ones((20, 2)) * 0.5, k=0)


class TestFits:
    def test_gaussian_fit_recovers_rho(self):
        sim = gaussian_copula_sim(0.6, 3000, seed=0)
        rho_hat = fit_gaussian_copula(sim)
        assert abs(rho_hat - 0.6) < 0.08

    def test_clayton_inversion(self):
        sim = clayton_copula_sim(2.0, 3000, seed=1)
        theta_hat = fit_clayton(sim)
        assert abs(theta_hat - 2.0) < 0.5

    def test_gumbel_inversion(self):
        sim = gumbel_copula_sim(2.0, 3000, seed=2)
        alpha_hat = fit_gumbel(sim)
        assert abs(alpha_hat - 2.0) < 0.4

    def test_t_fit(self):
        sim = t_copula_sim(0.5, 5.0, 3000, seed=3)
        rho_hat, nu_hat = fit_t_copula(sim)
        assert abs(rho_hat - 0.5) < 0.12
        assert 2.0 < nu_hat <= 30.0

    def test_failclosed(self):
        with pytest.raises(ValueError):
            fit_gaussian_copula(np.ones((20, 3)))
        with pytest.raises(ValueError):
            fit_t_copula(np.ones((20, 2)), nu_grid=np.array([1.0]))


class TestSimulations:
    def test_gaussian_sim_uniform(self):
        u = gaussian_copula_sim(0.3, 2000, seed=4)
        assert u.shape == (2000, 2)
        assert np.all((u >= 0) & (u <= 1))
        ks = sstats.kstest(u[:, 0], "uniform").pvalue
        assert ks > 0.01
        # Positive rho -> positive tau
        assert kendall_tau(u) > 0.1

    def test_t_sim_tail_dependence(self):
        u = t_copula_sim(0.5, 4.0, 5000, seed=5)
        lo, hi = empirical_tail_dependence(u)
        # t copula has symmetric tail dependence > 0.
        assert lo > 0.05 and hi > 0.05

    def test_clayton_lower_tail(self):
        u = clayton_copula_sim(3.0, 5000, seed=6)
        lo, hi = empirical_tail_dependence(u)
        assert lo > hi  # Clayton: lower tail only

    def test_gumbel_upper_tail(self):
        u = gumbel_copula_sim(2.5, 5000, seed=7)
        lo, hi = empirical_tail_dependence(u)
        assert hi > lo  # Gumbel: upper tail only
        tau = kendall_tau(u)
        assert abs(tau - (1.0 - 1.0 / 2.5)) < 0.1


class TestTheory:
    def test_t_values(self):
        lo, hi = copula_tail_dependence_theory("t", 0.5, nu=4.0)
        assert lo == hi and 0.0 < lo < 1.0
        # Higher nu -> less tail dependence.
        lo2, _ = copula_tail_dependence_theory("t", 0.5, nu=20.0)
        assert lo2 < lo

    def test_archimedean(self):
        lo, hi = copula_tail_dependence_theory("clayton", 2.0)
        assert abs(lo - 2.0**-0.5) < 1e-12 and hi == 0.0
        lo2, hi2 = copula_tail_dependence_theory("gumbel", 2.0)
        assert lo2 == 0.0 and abs(hi2 - (2.0 - np.sqrt(2.0))) < 1e-12

    def test_gaussian_zero(self):
        assert copula_tail_dependence_theory("gaussian", 0.9) == (0.0, 0.0)

    def test_failclosed(self):
        with pytest.raises(ValueError):
            copula_tail_dependence_theory("clayton", -1.0)
        with pytest.raises(ValueError):
            copula_tail_dependence_theory("gumbel", 0.5)
        with pytest.raises(ValueError):
            copula_tail_dependence_theory("frank", 1.0)
