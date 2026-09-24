"""Tests for forecast combinations and classic bandits."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.bandits import (
    EXP3,
    UCB1,
    BayesUCB,
    EpsilonGreedy,
    ThompsonBernoulli,
    ThompsonGaussian,
    simulate_bandit,
)
from quant_fund.models.ensemble import (
    combine,
    covariance_weights,
    equal_weights,
    granger_ramanathan,
    inverse_mse_weights,
    median_combination,
    rolling_inverse_mse,
    trimmed_mean_combination,
)


def _panel(rng: np.random.Generator, t: int = 200):
    y = rng.normal(size=t)
    f1 = y + rng.normal(scale=0.5, size=t)  # good  (MSE ~ 0.25)
    f2 = y + rng.normal(scale=1.2, size=t)  # poor  (MSE ~ 1.44)
    f3 = rng.normal(scale=1.5, size=t)  # useless (MSE ~ var(y) + 2.25)
    return np.column_stack([f1, f2, f3]), y


class TestCombinations:
    def test_inverse_mse_prefers_best(self):
        rng = np.random.default_rng(0)
        f, y = _panel(rng)
        w = inverse_mse_weights(f, y)
        assert np.isclose(w.sum(), 1.0)
        assert w[0] > w[1] > w[2]

    def test_covariance_weights_sum(self):
        rng = np.random.default_rng(1)
        f, y = _panel(rng)
        w = covariance_weights(f, y)
        assert np.isfinite(w).all()

    def test_granger_ramanathan(self):
        rng = np.random.default_rng(2)
        f, y = _panel(rng)
        w, a = granger_ramanathan(f, y)
        assert np.isclose(w.sum(), 1.0, atol=1e-4)
        assert np.all(w >= -1e-6)
        assert w[0] > 0.5
        assert abs(a) < 0.5

    def test_median_and_trimmed(self):
        rng = np.random.default_rng(3)
        f, _ = _panel(rng)
        med = median_combination(f)
        assert med.shape == (f.shape[0],)
        tm = trimmed_mean_combination(f, trim=0.34)  # drop 1 of 3 each side
        # Trimmed mean of 3 with trim 0.34 = median.
        assert np.allclose(tm, med, atol=1e-9)

    def test_combine_and_rolling(self):
        rng = np.random.default_rng(4)
        f, y = _panel(rng)
        w = inverse_mse_weights(f, y)
        fc = combine(f, w)
        assert fc.shape == (200,)
        rw = rolling_inverse_mse(f, y, window=50)
        assert rw.shape == (200, 3)
        # Early rows are equal weight; later rows adapt.
        assert np.allclose(rw[:50], 1 / 3)
        assert rw[-1, 0] > rw[-1, 1]

    def test_failclosed(self):
        with pytest.raises(ValueError):
            inverse_mse_weights(np.ones((3, 4)), np.ones(3))
        with pytest.raises(ValueError):
            equal_weights(0)
        with pytest.raises(ValueError):
            trimmed_mean_combination(np.ones((10, 3)), trim=0.6)
        with pytest.raises(ValueError):
            combine(np.ones((10, 3)), np.ones(2))


class TestBandits:
    MEANS = np.array([0.2, 0.5, 0.8])

    def _regret_check(self, bandit, steps: int = 3000, seed: int = 0):
        out = simulate_bandit(bandit, self.MEANS, steps, sigma=0.3, seed=seed)
        # Best-arm share in the last third should dominate.
        last = out["arms"][steps * 2 // 3 :]
        share_best = float(np.mean(last == 2))
        return share_best, out["regret"]

    def test_ucb1_finds_best(self):
        share, _ = self._regret_check(UCB1(3))
        assert share > 0.6

    def test_epsilon_greedy(self):
        share, _ = self._regret_check(EpsilonGreedy(3, epsilon=0.1, seed=1))
        assert share > 0.6

    def test_thompson_gaussian(self):
        share, _ = self._regret_check(ThompsonGaussian(3, sigma=0.3, seed=2))
        assert share > 0.8

    def test_thompson_bernoulli(self):
        # Bernoulli TS with clipped Gaussian rewards still prefers the max arm.
        share, _ = self._regret_check(ThompsonBernoulli(3, seed=3))
        assert share > 0.5

    def test_bayes_ucb(self):
        share, _ = self._regret_check(BayesUCB(3), seed=4)
        assert share > 0.5

    def test_exp3_runs_and_leans(self):
        rng = np.random.default_rng(5)
        b = EXP3(3, gamma=0.2, seed=6)
        arms = np.zeros(2000, dtype=np.intp)
        for i in range(2000):
            a = b.select()
            r = float(rng.random() < (self.MEANS[a] * 0.5 + 0.5))
            b.update(a, r)
            arms[i] = a
        share = float(np.mean(arms[-600:] == 2))
        assert share > 0.35  # adversarial bandit: softer guarantee

    def test_failclosed(self):
        with pytest.raises(ValueError):
            UCB1(1)
        with pytest.raises(ValueError):
            EpsilonGreedy(3, epsilon=1.5)
        with pytest.raises(ValueError):
            EXP3(3, gamma=0.0)
        b = UCB1(3)
        with pytest.raises(ValueError):
            b.update(0, float("nan"))
        e = EXP3(3, seed=0)
        chosen = e.select()
        wrong_arm = (chosen + 1) % 3
        with pytest.raises(ValueError):
            e.update(wrong_arm, 0.5)  # update must match selected arm
