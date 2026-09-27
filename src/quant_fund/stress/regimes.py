"""Gaussian regime-switching scenario generator.

The simulator draws a Markov chain and, in each state, a Gaussian vector.
Unconditional moments use the stationary distribution of the transition
matrix. A univariate fit can delegate to
:func:`quant_fund.models.regime_switch.fit_markov_switching_mean`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


@dataclass(frozen=True)
class GaussianRegimeSpec:
    """Parameters of a K-state Gaussian HMM. ``transition[i, j]`` is i → j."""

    means: Array
    covs: Array
    transition: Array

    def __post_init__(self) -> None:
        means = np.asarray(self.means, dtype=float)
        covs = np.asarray(self.covs, dtype=float)
        transition = np.asarray(self.transition, dtype=float)
        if means.ndim == 1:
            means = means.reshape(-1, 1)
        if means.ndim != 2:
            raise ValueError("means must have shape (n_states, n_assets)")
        n_states, n_assets = means.shape
        if n_states < 2 or n_assets < 1:
            raise ValueError("need at least two states and one asset")
        if covs.shape != (n_states, n_assets, n_assets):
            raise ValueError("covs must have shape (n_states, n_assets, n_assets)")
        if transition.shape != (n_states, n_states):
            raise ValueError("transition must be (n_states, n_states)")
        if not np.isfinite(means).all() or not np.isfinite(covs).all():
            raise ValueError("means and covs must be finite")
        if not np.isfinite(transition).all() or np.any(transition < 0.0):
            raise ValueError("transition probabilities must be finite and non-negative")
        if not np.allclose(transition.sum(axis=1), 1.0, atol=1e-8):
            raise ValueError("transition rows must sum to 1")
        for state in range(n_states):
            cov = covs[state]
            if not np.allclose(cov, cov.T, atol=1e-8):
                raise ValueError("state covariance must be symmetric")
            if float(np.linalg.eigvalsh(cov)[0]) <= 0.0:
                raise ValueError("state covariance must be positive definite")
        object.__setattr__(self, "means", np.asarray(means, dtype=np.float64))
        object.__setattr__(self, "covs", np.asarray(covs, dtype=np.float64))
        object.__setattr__(self, "transition", np.asarray(transition, dtype=np.float64))


def stationary_distribution(transition: Array) -> Array:
    """Solve π P = π with π summing to one."""
    matrix = np.asarray(transition, dtype=float)
    n_states = matrix.shape[0]
    system = np.eye(n_states) - matrix.T
    system[-1, :] = 1.0
    rhs = np.zeros(n_states)
    rhs[-1] = 1.0
    pi = np.linalg.solve(system, rhs)
    if not np.isfinite(pi).all() or np.any(pi < -1e-8):
        raise ValueError("stationary distribution is not a probability vector")
    pi = np.clip(pi, 0.0, None)
    total = float(pi.sum())
    if total <= 0.0:
        raise ValueError("stationary distribution is degenerate")
    return np.asarray(pi / total, dtype=np.float64)


def unconditional_moments(spec: GaussianRegimeSpec) -> dict[str, Array]:
    """Stationary mean and covariance of the Gaussian mixture induced by π."""
    pi = stationary_distribution(spec.transition)
    mean = np.zeros(spec.means.shape[1], dtype=np.float64)
    second = np.zeros((spec.means.shape[1], spec.means.shape[1]), dtype=np.float64)
    for state, weight in enumerate(pi):
        mu = spec.means[state]
        mean = mean + weight * mu
        second = second + weight * (spec.covs[state] + np.outer(mu, mu))
    cov = second - np.outer(mean, mean)
    return {
        "mean": np.asarray(mean, dtype=np.float64),
        "cov": np.asarray(cov, dtype=np.float64),
        "pi": pi,
    }


def simulate_gaussian_hmm(
    spec: GaussianRegimeSpec,
    n: int,
    rng: np.random.Generator,
) -> tuple[Array, NDArray[np.int64]]:
    """Simulate ``n`` draws starting from the stationary distribution."""
    if isinstance(n, bool) or not isinstance(n, int) or n < 1:
        raise ValueError("n must be a positive integer")
    moments = unconditional_moments(spec)
    pi = moments["pi"]
    n_states = spec.transition.shape[0]
    state = int(rng.choice(n_states, p=pi))
    returns = np.empty((n, spec.means.shape[1]), dtype=np.float64)
    states = np.empty(n, dtype=np.int64)
    for t in range(n):
        states[t] = state
        returns[t] = rng.multivariate_normal(spec.means[state], spec.covs[state])
        state = int(rng.choice(n_states, p=spec.transition[state]))
    return returns, states


def spec_from_univariate_fit(fit: dict[str, Array]) -> GaussianRegimeSpec:
    """Wrap a ``fit_markov_switching_mean`` result as a one-asset HMM spec."""
    means = np.asarray(fit["means"], dtype=float).reshape(-1, 1)
    sigmas = np.asarray(fit["sigmas"], dtype=float).reshape(-1)
    covs = np.array([[[float(sigma) ** 2]] for sigma in sigmas], dtype=np.float64)
    transition = np.asarray(fit["P"], dtype=float)
    return GaussianRegimeSpec(means=means, covs=covs, transition=transition)
