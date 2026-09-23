"""Classic multi-armed bandit algorithms (non-contextual).

``models/rl.py`` already has contextual LinUCB; this module covers the
canonical reward-only bandits used for model/parameter selection.

References:
- Auer, Cesa-Bianchi, Fischer (2002). Finite-time analysis of the
  multiarmed bandit problem (UCB1). *Machine Learning* 47.
- Auer et al. (2002). The nonstochastic multiarmed bandit problem (EXP3).
  *SIAM Journal on Computing* 32.
- Thompson (1933). On the likelihood that one unknown probability exceeds
  another — Beta-Bernoulli Thompson sampling.
- Kaufmann, Cappé, Garivier (2012). On Bayesian upper confidence bounds
  for bandit problems (Bayes-UCB). *AISTATS*.
- Sutton, Barto (2018). *Reinforcement Learning* — epsilon-greedy.
- Gittins (1979) — conceptual basis for index policies (not implemented).
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


class _BaseBandit:
    """Shared state: counts, empirical means, total pulls."""

    def __init__(self, n_arms: int) -> None:
        if isinstance(n_arms, bool) or not isinstance(n_arms, int) or n_arms < 2:
            raise ValueError("n_arms must be an integer >= 2")
        self.n_arms = n_arms
        self.counts = np.zeros(n_arms)
        self.values = np.zeros(n_arms)  # running mean reward
        self.t = 0

    def _update(self, arm: int, reward: float) -> None:
        if not np.isfinite(reward):
            raise ValueError("reward must be finite")
        self.t += 1
        self.counts[arm] += 1.0
        self.values[arm] += (reward - self.values[arm]) / self.counts[arm]

    def select(self) -> int:  # pragma: no cover - abstract
        raise NotImplementedError

    def update(self, arm: int, reward: float) -> None:  # pragma: no cover
        raise NotImplementedError


class UCB1(_BaseBandit):
    """Auer et al. (2002) UCB1: ``mean_i + c * sqrt(ln t / n_i)``."""

    def __init__(self, n_arms: int, c: float = 2.0) -> None:
        super().__init__(n_arms)
        if not np.isfinite(c) or c <= 0.0:
            raise ValueError("c must be positive and finite")
        self.c = c

    def select(self) -> int:
        unpulled = np.flatnonzero(self.counts == 0)
        if unpulled.size:
            return int(unpulled[0])
        bonus = self.c * np.sqrt(np.log(max(self.t, 2)) / self.counts)
        return int(np.argmax(self.values + bonus))

    def update(self, arm: int, reward: float) -> None:
        self._update(arm, reward)


class EpsilonGreedy(_BaseBandit):
    """Sutton–Barto ε-greedy; supports decaying ε via ``epsilon0 / sqrt(t)``."""

    def __init__(
        self,
        n_arms: int,
        epsilon: float = 0.1,
        decay: bool = False,
        seed: int | None = None,
    ) -> None:
        super().__init__(n_arms)
        if not np.isfinite(epsilon) or not (0.0 <= epsilon <= 1.0):
            raise ValueError("epsilon must be in [0, 1]")
        self.epsilon0 = epsilon
        self.decay = decay
        self.rng = np.random.default_rng(seed)

    def _eps(self) -> float:
        if not self.decay or self.t == 0:
            return self.epsilon0
        return min(1.0, self.epsilon0 * self.n_arms / self.t * 5.0)

    def select(self) -> int:
        unpulled = np.flatnonzero(self.counts == 0)
        if unpulled.size:
            return int(unpulled[0])
        if self.rng.random() < self._eps():
            return int(self.rng.integers(self.n_arms))
        return int(np.argmax(self.values))

    def update(self, arm: int, reward: float) -> None:
        self._update(arm, reward)


class ThompsonBernoulli(_BaseBandit):
    """Thompson (1933) sampling with Beta(1,1) prior; rewards clipped to [0,1]."""

    def __init__(self, n_arms: int, seed: int | None = None) -> None:
        super().__init__(n_arms)
        self.rng = np.random.default_rng(seed)
        self.alpha = np.ones(n_arms)
        self.beta = np.ones(n_arms)

    def select(self) -> int:
        draws = self.rng.beta(self.alpha, self.beta)
        return int(np.argmax(draws))

    def update(self, arm: int, reward: float) -> None:
        if not np.isfinite(reward):
            raise ValueError("reward must be finite")
        r = min(max(reward, 0.0), 1.0)
        self.t += 1
        self.counts[arm] += 1.0
        self.values[arm] += (r - self.values[arm]) / self.counts[arm]
        self.alpha[arm] += r
        self.beta[arm] += 1.0 - r


class ThompsonGaussian(_BaseBandit):
    """Gaussian Thompson sampling with known reward std ``sigma``.

    Posterior on each arm's mean: N(empirical mean, sigma^2 / n).
    """

    def __init__(self, n_arms: int, sigma: float = 1.0, seed: int | None = None) -> None:
        super().__init__(n_arms)
        if not np.isfinite(sigma) or sigma <= 0.0:
            raise ValueError("sigma must be positive and finite")
        self.sigma = sigma
        self.rng = np.random.default_rng(seed)

    def select(self) -> int:
        unpulled = np.flatnonzero(self.counts == 0)
        if unpulled.size:
            return int(unpulled[0])
        post_sd = self.sigma / np.sqrt(self.counts)
        draws = self.values + post_sd * self.rng.standard_normal(self.n_arms)
        return int(np.argmax(draws))

    def update(self, arm: int, reward: float) -> None:
        self._update(arm, reward)


class EXP3:
    """Auer et al. (2002) EXP3 for adversarial bandits.

    Maintains importance-weighted reward estimates and samples arms from an
    exponential-weight distribution.  Rewards should lie in [0,1] (rewards
    outside are clipped for the loss update — documented behavior).
    """

    def __init__(
        self, n_arms: int, gamma: float = 0.1, seed: int | None = None
    ) -> None:
        if isinstance(n_arms, bool) or not isinstance(n_arms, int) or n_arms < 2:
            raise ValueError("n_arms must be an integer >= 2")
        if not np.isfinite(gamma) or not (0.0 < gamma <= 1.0):
            raise ValueError("gamma must be in (0, 1]")
        self.n_arms = n_arms
        self.gamma = gamma
        self.rng = np.random.default_rng(seed)
        self.w = np.ones(n_arms)
        self.last_probs = np.full(n_arms, 1.0 / n_arms)
        self.last_arm = -1

    def select(self) -> int:
        w_sum = self.w.sum()
        p = (1.0 - self.gamma) * self.w / w_sum + self.gamma / self.n_arms
        self.last_probs = p
        arm = int(self.rng.choice(self.n_arms, p=p))
        self.last_arm = arm
        return arm

    def update(self, arm: int, reward: float) -> None:
        if arm != self.last_arm:
            raise ValueError("EXP3 update must follow the same arm just selected")
        if not np.isfinite(reward):
            raise ValueError("reward must be finite")
        r = min(max(reward, 0.0), 1.0)
        p_a = max(self.last_probs[arm], 1e-12)
        x_hat = r / p_a  # importance-weighted reward estimate
        self.w[arm] *= math.exp(self.gamma * x_hat / self.n_arms)


class BayesUCB(_BaseBandit):
    """Kaufmann et al. (2012) Bayes-UCB for Bernoulli rewards.

    Selects the arm maximizing the posterior ``1 - 1/(t * log^c t)``
    quantile of Beta(alpha_i, beta_i).
    """

    def __init__(self, n_arms: int, c: float = 0.0, seed: int | None = None) -> None:
        super().__init__(n_arms)
        if not np.isfinite(c) or c < 0.0:
            raise ValueError("c must be non-negative and finite")
        self.c = c
        self.alpha = np.ones(n_arms)
        self.beta = np.ones(n_arms)

    def select(self) -> int:
        from scipy.stats import beta as beta_dist

        t = max(self.t, 1)
        quantile = 1.0 - 1.0 / (t * max(math.log(t), 1.0) ** self.c)
        qs = beta_dist.ppf(min(quantile, 1.0 - 1e-12), self.alpha, self.beta)
        return int(np.argmax(qs))

    def update(self, arm: int, reward: float) -> None:
        if not np.isfinite(reward):
            raise ValueError("reward must be finite")
        r = min(max(reward, 0.0), 1.0)
        self.t += 1
        self.counts[arm] += 1.0
        self.values[arm] += (r - self.values[arm]) / self.counts[arm]
        self.alpha[arm] += r
        self.beta[arm] += 1.0 - r


def simulate_bandit(
    bandit: _BaseBandit | EXP3,
    reward_means: Array,
    n_steps: int,
    sigma: float = 1.0,
    seed: int | None = None,
) -> dict[str, Array | NDArray[np.intp]]:
    """Simulate ``n_steps`` of bandit play against Gaussian arms.

    Returns chosen arms, rewards, and cumulative regret vs the best arm.
    """
    mu = _as_reward_means(reward_means)
    if isinstance(n_steps, bool) or not isinstance(n_steps, int) or n_steps < 1:
        raise ValueError("n_steps must be a positive integer")
    if not np.isfinite(sigma) or sigma <= 0.0:
        raise ValueError("sigma must be positive and finite")
    if isinstance(bandit, _BaseBandit) and bandit.n_arms != mu.size:
        raise ValueError("bandit n_arms must match reward_means")
    if isinstance(bandit, EXP3) and bandit.n_arms != mu.size:
        raise ValueError("bandit n_arms must match reward_means")
    rng = np.random.default_rng(seed)
    arms = np.zeros(n_steps, dtype=np.intp)
    rewards = np.zeros(n_steps)
    for i in range(n_steps):
        arm = bandit.select()
        r = float(rng.normal(mu[arm], sigma))
        bandit.update(arm, r)
        arms[i] = arm
        rewards[i] = r
    regret = np.cumsum(mu.max() - mu[arms])
    return {"arms": arms, "rewards": rewards, "regret": regret}


def _as_reward_means(mu: Array) -> Array:
    v = np.asarray(mu, dtype=float).reshape(-1)
    if v.size < 2 or not np.all(np.isfinite(v)):
        raise ValueError("reward_means must be a finite vector of length >= 2")
    return v
