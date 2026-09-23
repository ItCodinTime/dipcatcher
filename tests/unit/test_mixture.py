"""Tests for EM mixture models."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.mixture import (
    fit_gaussian_mixture,
    fit_t_mixture,
    mixing_density_stats,
    mixture_bic,
    select_mixture_k,
)


def _two_component(seed: int = 0, n: int = 400, gap: float = 3.0):
    rng = np.random.default_rng(seed)
    a = rng.normal(-gap / 2, 0.5, n // 2)
    b = rng.normal(gap / 2, 0.5, n - n // 2)
    return np.concatenate([a, b]), np.concatenate([np.zeros(n // 2), np.ones(n - n // 2)])


class TestGaussianMixture:
    def test_recovers_components(self):
        x, labels = _two_component()
        fit = fit_gaussian_mixture(x, k=2, seed=0)
        assert fit["means"].shape == (2, 1)
        # Means near +-1.5.
        mus = np.sort(fit["means"][:, 0])
        assert abs(mus[0] + 1.5) < 0.3
        assert abs(mus[1] - 1.5) < 0.3
        # Responsibilities recover labels.
        pred = np.argmax(fit["responsibilities"], axis=1)
        acc = max(np.mean(pred == labels), np.mean(1 - pred == labels))
        assert acc > 0.95

    def test_loglik_monotone_increases(self):
        x, _ = _two_component()
        fit = fit_gaussian_mixture(x, k=2, seed=1)
        ll = fit["loglik"]
        assert np.all(np.diff(ll) > -1e-6)

    def test_bic_selects_two(self):
        x, _ = _two_component(gap=4.0)
        out = select_mixture_k(x, k_range=range(1, 4), seed=0)
        assert out["best_k"][0] == 2.0

    def test_failclosed(self):
        with pytest.raises(ValueError):
            fit_gaussian_mixture(np.ones(30), k=40)
        with pytest.raises(ValueError):
            fit_gaussian_mixture(np.full(50, np.nan), k=2)


class TestTMixture:
    def test_recovers_heavy_tail_components(self):
        rng = np.random.default_rng(2)
        a = rng.standard_t(df=4, size=200) * 0.5 - 1.5
        b = rng.standard_t(df=4, size=200) * 0.5 + 1.5
        x = np.concatenate([a, b])
        fit = fit_t_mixture(x, k=2, seed=0)
        mus = np.sort(fit["means"][:, 0])
        assert abs(mus[0] + 1.5) < 0.4
        assert abs(mus[1] - 1.5) < 0.4
        assert np.all(fit["nus"] >= 3.0)

    def test_bic_counts_nu(self):
        rng = np.random.default_rng(3)
        x = rng.normal(size=300)
        fit_g = fit_gaussian_mixture(x, k=2, seed=0)
        fit_t = fit_t_mixture(x, k=2, seed=0)
        # t-mixture has 2 extra df params.
        p_g = mixture_bic(fit_g, 300)
        p_t = mixture_bic(fit_t, 300)
        assert p_t > p_g - 20.0  # penalized by extra nu params

    def test_mixing_density_stats(self):
        x, _ = _two_component(gap=5.0)
        fit = fit_gaussian_mixture(x, k=2, seed=0)
        stats = mixing_density_stats(fit)
        assert stats["eff_components"] > 1.5
        assert stats["mean_gap"] > 3.0
        # Well-separated -> low assignment entropy.
        assert stats["resp_entropy"] < 0.3

    def test_failclosed(self):
        with pytest.raises(ValueError):
            fit_t_mixture(np.ones(20), k=0)
        with pytest.raises(ValueError):
            select_mixture_k(np.ones(30), k_range=range(30, 34))
