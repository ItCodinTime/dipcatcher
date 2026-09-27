"""Tests for bootstrap and subsampling schemes."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.metrics.bootstrap import (
    circular_block_indices,
    pairs_bootstrap,
    rademacher_bootstrap,
    sieve_bootstrap_residuals,
    subsample_statistic,
    wild_bootstrap_residuals,
)


class TestWildBootstrap:
    def test_mammen_moments(self):
        e = np.linspace(-2, 2, 200)
        boot = wild_bootstrap_residuals(e, n_boot=2000, seed=0)
        assert boot.shape == (2000, 200)
        # Wild bootstrap preserves heteroskedastic scale: E[e* | e] = 0.
        assert np.all(np.abs(boot.mean(axis=0)) < 0.3)
        # E[e*^2] ~ e^2 approximately.
        var_ratio = (boot**2).mean(axis=0) / np.maximum(e**2, 1e-9)
        mid = var_ratio[40:160]
        assert np.abs(mid.mean() - 1.0) < 0.35

    def test_rademacher(self):
        e = np.random.default_rng(0).normal(size=150)
        boot = rademacher_bootstrap(e, n_boot=500, seed=1)
        assert boot.shape == (500, 150)
        assert np.all(np.abs(boot.mean(axis=0)) < 0.4)

    def test_failclosed(self):
        with pytest.raises(ValueError):
            wild_bootstrap_residuals(np.ones(3))


class TestPairsBootstrap:
    def test_recovers_beta_distribution(self):
        rng = np.random.default_rng(2)
        n = 200
        x = np.column_stack([np.ones(n), rng.normal(size=n)])
        beta_true = np.array([1.0, 2.0])
        y = x @ beta_true + rng.normal(scale=0.5, size=n)
        boot = pairs_bootstrap(y, x, n_boot=300, seed=3)
        assert boot.shape == (300, 2)
        assert np.abs(boot[:, 1].mean() - 2.0) < 0.15
        assert boot[:, 1].std() > 0.01  # nonzero dispersion

    def test_failclosed(self):
        with pytest.raises(ValueError):
            pairs_bootstrap(np.ones(5), np.ones((4, 2)))


class TestSieveBootstrap:
    def test_preserves_autocorrelation(self):
        rng = np.random.default_rng(4)
        n = 300
        e = np.empty(n)
        e[0] = 0.0
        for t in range(1, n):
            e[t] = 0.7 * e[t - 1] + rng.normal()
        boot = sieve_bootstrap_residuals(e, order=1, n_boot=100, seed=5)
        assert boot.shape == (100, n)
        ac1 = [np.corrcoef(b[:-1], b[1:])[0, 1] for b in boot]
        assert np.mean(ac1) > 0.4  # AR(1) structure preserved

    def test_failclosed(self):
        with pytest.raises(ValueError):
            sieve_bootstrap_residuals(np.ones(10), order=5)


class TestSubsampling:
    def test_mean_statistic(self):
        rng = np.random.default_rng(6)
        v = rng.normal(loc=1.5, size=400)
        out = subsample_statistic(v, lambda b: float(b.mean()), block_len=50)
        assert out["n_blocks"] == 351
        assert abs(out["full_stat"] - 1.5) < 0.2
        assert out["q025"] < out["full_stat"] < out["q975"]

    def test_nonoverlapping(self):
        v = np.random.default_rng(7).normal(size=300)
        out = subsample_statistic(v, lambda b: float(b.std()), block_len=50, overlapping=False)
        assert out["n_blocks"] == 6

    def test_failclosed(self):
        with pytest.raises(ValueError):
            subsample_statistic(np.ones(20), lambda b: 0.0, block_len=25)


class TestCircularBlocks:
    def test_indices_valid(self):
        idx = circular_block_indices(n=100, block_len=10, n_boot=20, seed=8)
        assert idx.shape == (20, 100)
        assert np.all(idx >= 0) and np.all(idx < 100)

    def test_failclosed(self):
        with pytest.raises(ValueError):
            circular_block_indices(50, 0, 5)
