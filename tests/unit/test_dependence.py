"""Tests for nonlinear dependence measures."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.metrics.dependence import (
    chatterjee_xi,
    distance_correlation,
    hsic,
    mmd,
    mutual_information_knn,
)


class TestDistanceCorrelation:
    def test_independent_low(self):
        rng = np.random.default_rng(0)
        x = rng.normal(size=200)
        y = rng.normal(size=200)
        out = distance_correlation(x, y)
        assert out["dcor"] < 0.25

    def test_nonlinear_dependence_detected(self):
        rng = np.random.default_rng(1)
        x = rng.uniform(-2, 2, 300)
        y = x**2 + rng.normal(scale=0.2, size=300)  # pearson ~ 0
        out = distance_correlation(x, y)
        assert out["dcor"] > 0.4
        # Pearson would miss this (symmetric nonlinearity).
        assert abs(np.corrcoef(x, y)[0, 1]) < 0.2

    def test_failclosed(self):
        with pytest.raises(ValueError):
            distance_correlation(np.ones(50), np.ones(50))


class TestHSIC:
    def test_dependence_detected(self):
        rng = np.random.default_rng(2)
        x = rng.normal(size=120)
        y = np.sin(3 * x) + rng.normal(scale=0.1, size=120)
        out = hsic(x, y, n_perm=100, seed=3)
        assert out["pvalue"] < 0.05

    def test_independent(self):
        rng = np.random.default_rng(3)
        out = hsic(rng.normal(size=100), rng.normal(size=100), n_perm=100, seed=4)
        assert out["pvalue"] > 0.02  # not a rejection at conventional levels

    def test_failclosed(self):
        with pytest.raises(ValueError):
            hsic(np.ones(10), np.ones(10))


class TestMMD:
    def test_same_distribution(self):
        rng = np.random.default_rng(5)
        x = rng.normal(size=150)
        y = rng.normal(size=150)
        out = mmd(x, y)
        assert out["pvalue"] > 0.1

    def test_different_distribution(self):
        rng = np.random.default_rng(6)
        x = rng.normal(size=150)
        y = rng.normal(loc=1.5, size=150)
        out = mmd(x, y)
        assert out["mmd2"] > 0.05
        assert out["pvalue"] < 0.05

    def test_failclosed(self):
        with pytest.raises(ValueError):
            mmd(np.ones(10), np.ones(9))


class TestChatterjee:
    def test_monotone_detected(self):
        rng = np.random.default_rng(7)
        x = rng.uniform(0, 5, 300)
        y = np.log1p(x) + rng.normal(scale=0.05, size=300)
        out = chatterjee_xi(x, y)
        assert out["xi"] > 0.6
        assert out["pvalue"] < 0.01

    def test_independent_near_zero(self):
        rng = np.random.default_rng(8)
        out = chatterjee_xi(rng.normal(size=300), rng.normal(size=300))
        assert abs(out["xi"]) < 0.25

    def test_nonmonotone_function(self):
        rng = np.random.default_rng(9)
        x = rng.uniform(-3, 3, 400)
        y = np.cos(2 * x) + rng.normal(scale=0.05, size=400)
        out = chatterjee_xi(x, y)
        assert out["xi"] > 0.4

    def test_failclosed(self):
        with pytest.raises(ValueError):
            chatterjee_xi(np.ones(40), np.ones(40))


class TestKraskovMI:
    def test_dependent_positive(self):
        rng = np.random.default_rng(10)
        x = rng.normal(size=200)
        y = x + rng.normal(scale=0.3, size=200)
        out = mutual_information_knn(x, y, k=4)
        assert out["mi"] > 0.3

    def test_independent_near_zero(self):
        rng = np.random.default_rng(11)
        out = mutual_information_knn(rng.normal(size=200), rng.normal(size=200), k=4)
        assert out["mi"] < 0.25

    def test_failclosed(self):
        with pytest.raises(ValueError):
            mutual_information_knn(np.ones(50), np.ones(50))
