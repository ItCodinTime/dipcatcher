"""Discrete first-order HMM — Jurafsky & Martin SLP3 Appendix A.

Rabiner's three problems:

1. Likelihood — forward (A.11–A.12)
2. Decoding — Viterbi (A.13–A.14)
3. Learning — Baum–Welch / forward–backward (A.15–A.28)

0-based arrays. Textbook 1-based ``α_t(j)`` is ``alpha[t, j]``.
Linear-space matches the Eisner ice-cream arithmetic. ``log_*`` variants
are for longer sequences. Research only; not a live book.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypedDict

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]
IntArray = NDArray[np.intp]
_EPS = 1e-15


class DiscreteHMMDict(TypedDict):
    """Serialized HMM matrices and initial-state probabilities."""

    A: list[list[float]]
    B: list[list[float]]
    pi: list[float]


@dataclass
class DiscreteHMM:
    """λ = (A, B, π). Rows of A and B and π sum to 1."""

    A: Array
    B: Array
    pi: Array

    def __post_init__(self) -> None:
        self.A = np.asarray(self.A, dtype=float)
        self.B = np.asarray(self.B, dtype=float)
        self.pi = np.asarray(self.pi, dtype=float)
        n = self.A.shape[0]
        if self.A.shape != (n, n):
            raise ValueError("A must be square")
        if self.B.ndim != 2 or self.B.shape[0] != n:
            raise ValueError("B must be (n_states, n_obs)")
        if self.pi.shape != (n,):
            raise ValueError("pi must be (n_states,)")
        if np.any(self.A < 0) or np.any(self.B < 0) or np.any(self.pi < 0):
            raise ValueError("HMM parameters must be non-negative")
        if not np.all(np.isfinite(self.A)) or not np.all(np.isfinite(self.B)):
            raise ValueError("HMM parameters must be finite")
        if not np.all(np.isfinite(self.pi)):
            raise ValueError("HMM parameters must be finite")
        tol = 1e-9
        if not np.allclose(self.A.sum(axis=1), 1.0, atol=tol):
            raise ValueError("rows of A must sum to 1")
        if not np.allclose(self.B.sum(axis=1), 1.0, atol=tol):
            raise ValueError("rows of B must sum to 1")
        if not np.isclose(self.pi.sum(), 1.0, atol=tol):
            raise ValueError("pi must sum to 1")

    @property
    def n_states(self) -> int:
        return int(self.A.shape[0])

    @property
    def n_obs(self) -> int:
        return int(self.B.shape[1])

    def as_dict(self) -> DiscreteHMMDict:
        return {
            "A": self.A.tolist(),
            "B": self.B.tolist(),
            "pi": self.pi.tolist(),
        }


def _as_obs(observations: list[int] | IntArray, n_obs: int) -> IntArray:
    o = np.asarray(observations, dtype=int)
    if o.ndim != 1 or o.size == 0:
        raise ValueError("observations must be a non-empty 1-D sequence")
    if np.any(o < 0) or np.any(o >= n_obs):
        raise ValueError("observation index out of vocabulary")
    return o


def forward(model: DiscreteHMM, observations: list[int] | IntArray) -> tuple[Array, float]:
    """P(O|λ) and the forward trellis α_t(j) = P(o_1..o_t, q_t=j | λ)."""
    o = _as_obs(observations, model.n_obs)
    t_len, n = o.size, model.n_states
    alpha = np.zeros((t_len, n), dtype=float)
    alpha[0] = model.pi * model.B[:, o[0]]
    for t in range(1, t_len):
        alpha[t] = (alpha[t - 1] @ model.A) * model.B[:, o[t]]
    likelihood = float(alpha[-1].sum())
    return alpha, likelihood


def backward(model: DiscreteHMM, observations: list[int] | IntArray) -> Array:
    """β_t(i) = P(o_{t+1}..o_T | q_t=i, λ). β_T(i) = 1."""
    o = _as_obs(observations, model.n_obs)
    t_len, n = o.size, model.n_states
    beta = np.zeros((t_len, n), dtype=float)
    beta[-1] = 1.0
    for t in range(t_len - 2, -1, -1):
        beta[t] = model.A @ (model.B[:, o[t + 1]] * beta[t + 1])
    return beta


def likelihood(model: DiscreteHMM, observations: list[int] | IntArray) -> float:
    _, p = forward(model, observations)
    return p


def _log_params(model: DiscreteHMM) -> tuple[Array, Array, Array]:
    eps = 1e-300
    return (
        np.log(np.clip(model.A, eps, None)),
        np.log(np.clip(model.B, eps, None)),
        np.log(np.clip(model.pi, eps, None)),
    )


def log_forward(model: DiscreteHMM, observations: list[int] | IntArray) -> tuple[Array, float]:
    """Scaled-free forward in log space — safe for arbitrarily long sequences."""
    from scipy.special import logsumexp

    o = _as_obs(observations, model.n_obs)
    log_a, log_b, log_pi = _log_params(model)
    t_len, n = o.size, model.n_states
    alpha = np.empty((t_len, n), dtype=float)
    alpha[0] = log_pi + log_b[:, o[0]]
    for t in range(1, t_len):
        alpha[t] = logsumexp(alpha[t - 1][:, None] + log_a, axis=0) + log_b[:, o[t]]
    return alpha, float(logsumexp(alpha[-1]))


def log_backward(model: DiscreteHMM, observations: list[int] | IntArray) -> Array:
    """Backward trellis in log space; beta[T-1] = log(1) = 0."""
    from scipy.special import logsumexp

    o = _as_obs(observations, model.n_obs)
    log_a, log_b, _ = _log_params(model)
    t_len, n = o.size, model.n_states
    beta = np.zeros((t_len, n), dtype=float)
    for t in range(t_len - 2, -1, -1):
        beta[t] = logsumexp(log_a + (log_b[:, o[t + 1]] + beta[t + 1])[None, :], axis=1)
    return beta


def log_likelihood(model: DiscreteHMM, observations: list[int] | IntArray) -> float:
    """log P(O|λ) — usable where linear `likelihood` underflows."""
    _, ll = log_forward(model, observations)
    return ll


def log_gamma_xi(
    model: DiscreteHMM, observations: list[int] | IntArray
) -> tuple[Array, Array, float]:
    """Posterior occupancies/transitions in log space (long sequences)."""

    o = _as_obs(observations, model.n_obs)
    log_a, log_b, _ = _log_params(model)
    alpha, ll = log_forward(model, o)
    beta = log_backward(model, o)
    if not np.isfinite(ll):
        raise ValueError("forward likelihood is zero; cannot form posteriors")
    gamma = np.exp(alpha + beta - ll)
    t_len, n = o.size, model.n_states
    xi = np.zeros((max(t_len - 1, 0), n, n), dtype=float)
    for t in range(t_len - 1):
        raw = alpha[t][:, None] + log_a + log_b[:, o[t + 1]][None, :] + beta[t + 1][None, :]
        xi[t] = np.exp(raw - ll)
    return gamma, xi, ll


def log_viterbi(model: DiscreteHMM, observations: list[int] | IntArray) -> tuple[IntArray, float]:
    """Most probable state path in log space; returns (path, log P)."""
    o = _as_obs(observations, model.n_obs)
    log_a, log_b, log_pi = _log_params(model)
    t_len, n = o.size, model.n_states
    vit = np.empty((t_len, n), dtype=float)
    back = np.zeros((t_len, n), dtype=np.intp)
    vit[0] = log_pi + log_b[:, o[0]]
    for t in range(1, t_len):
        scores = vit[t - 1][:, None] + log_a + log_b[:, o[t]][None, :]
        back[t] = np.argmax(scores, axis=0)
        vit[t] = scores[back[t], np.arange(n)]
    last = int(np.argmax(vit[-1]))
    path = np.empty(t_len, dtype=np.intp)
    path[-1] = last
    for t in range(t_len - 2, -1, -1):
        path[t] = back[t + 1, path[t + 1]]
    return path, float(vit[-1, last])


def viterbi(model: DiscreteHMM, observations: list[int] | IntArray) -> tuple[IntArray, float]:
    """Most probable state path and its joint probability (A.13–A.14)."""
    o = _as_obs(observations, model.n_obs)
    t_len, n = o.size, model.n_states
    vit = np.zeros((t_len, n), dtype=float)
    back = np.zeros((t_len, n), dtype=np.intp)
    vit[0] = model.pi * model.B[:, o[0]]
    back[0] = 0
    for t in range(1, t_len):
        scores = vit[t - 1][:, None] * model.A * model.B[:, o[t]][None, :]
        back[t] = np.argmax(scores, axis=0)
        vit[t] = scores[back[t], np.arange(n)]
    last = int(np.argmax(vit[-1]))
    path_prob = float(vit[-1, last])
    path = np.empty(t_len, dtype=np.intp)
    path[-1] = last
    for t in range(t_len - 2, -1, -1):
        path[t] = back[t + 1, path[t + 1]]
    return path, path_prob


def gamma_xi(model: DiscreteHMM, observations: list[int] | IntArray) -> tuple[Array, Array, float]:
    """E-step occupancies γ_t(j) and transitions ξ_t(i,j)."""
    o = _as_obs(observations, model.n_obs)
    alpha, p_o = forward(model, o)
    beta = backward(model, o)
    if p_o <= _EPS:
        raise ValueError("forward likelihood is zero; cannot form posteriors")
    gamma = alpha * beta / p_o
    t_len, n = o.size, model.n_states
    xi = np.zeros((max(t_len - 1, 0), n, n), dtype=float)
    for t in range(t_len - 1):
        raw = alpha[t][:, None] * model.A * model.B[:, o[t + 1]][None, :] * beta[t + 1][None, :]
        xi[t] = raw / p_o
    return gamma, xi, p_o


def baum_welch(
    observations: list[int] | IntArray,
    *,
    n_states: int,
    n_obs: int,
    n_iter: int = 20,
    seed: int = 7,
    model: DiscreteHMM | None = None,
    log_space: bool = False,
) -> tuple[DiscreteHMM, list[float]]:
    """Forward–backward (Fig. A.14). Likelihood is non-decreasing on one sequence.

    ``log_space=True`` runs the E-step on the log-space trellis so training
    does not underflow on long sequences; the returned history is then in
    log-likelihood units.
    """
    if n_states < 1 or n_obs < 1:
        raise ValueError("n_states and n_obs must be >= 1")
    o = np.asarray(observations, dtype=int)
    _as_obs(o, n_obs)
    if model is None:
        rng = np.random.default_rng(int(seed))
        A = rng.random((n_states, n_states))
        B = rng.random((n_states, n_obs))
        pi = rng.random(n_states)
        A = A / A.sum(axis=1, keepdims=True)
        B = B / B.sum(axis=1, keepdims=True)
        pi = pi / pi.sum()
        model = DiscreteHMM(A, B, pi)
    history: list[float] = []
    current = model
    for _ in range(int(n_iter)):
        if log_space:
            gamma, xi, p_o = log_gamma_xi(current, o)
        else:
            gamma, xi, p_o = gamma_xi(current, o)
        history.append(p_o)
        if xi.size:
            A_hat = xi.sum(axis=0)
            row_mass = A_hat.sum(axis=1, keepdims=True)
            occupied = row_mass[:, 0] > _EPS
            with np.errstate(divide="ignore", invalid="ignore"):
                A_norm = np.divide(
                    A_hat, row_mass, out=np.zeros_like(A_hat), where=occupied[:, None]
                )
            # States never occupied keep their prior row — the M-step has no
            # evidence to re-estimate them.
            A_hat = np.where(occupied[:, None], A_norm, current.A)
        else:
            A_hat = current.A
        B_hat = np.zeros((current.n_states, current.n_obs), dtype=float)
        for vk in range(current.n_obs):
            mask = o == vk
            B_hat[:, vk] = gamma[mask].sum(axis=0) if mask.any() else 0.0
        state_mass = gamma.sum(axis=0)[:, None]
        with np.errstate(divide="ignore", invalid="ignore"):
            B_norm = np.divide(B_hat, state_mass, out=np.zeros_like(B_hat), where=state_mass > _EPS)
        B_hat = np.where(state_mass > _EPS, B_norm, current.B)
        pi_hat = gamma[0] / max(float(gamma[0].sum()), _EPS)
        current = DiscreteHMM(A_hat, B_hat, pi_hat)
    if log_space:
        history.append(log_likelihood(current, o))
    else:
        _, last = forward(current, o)
        history.append(last)
    return current, history


def mle_supervised(
    states: list[int] | IntArray,
    observations: list[int] | IntArray,
    *,
    n_states: int,
    n_obs: int,
) -> DiscreteHMM:
    """Fully visible MLE for A, B, π (Appendix A.5 warm-up)."""
    q = np.asarray(states, dtype=int)
    o = _as_obs(observations, n_obs)
    if q.shape != o.shape:
        raise ValueError("states and observations must align")
    if np.any(q < 0) or np.any(q >= n_states):
        raise ValueError("state index out of range")
    A = np.zeros((n_states, n_states), dtype=float)
    B = np.zeros((n_states, n_obs), dtype=float)
    pi = np.zeros(n_states, dtype=float)
    pi[q[0]] += 1.0
    for t in range(len(q) - 1):
        A[q[t], q[t + 1]] += 1.0
    for t in range(len(q)):
        B[q[t], o[t]] += 1.0
    a_mass = A.sum(axis=1, keepdims=True)
    b_mass = B.sum(axis=1, keepdims=True)
    # MLE is undefined for a state with no outgoing transitions (or no
    # emissions — e.g. the sequence's terminal state): complete those rows
    # with the uniform maximum-entropy distribution rather than emitting a
    # non-stochastic model.
    A = np.where(
        a_mass > 0,
        np.divide(A, a_mass, out=np.zeros_like(A), where=a_mass > 0),
        1.0 / n_states,
    )
    B = np.where(
        b_mass > 0,
        np.divide(B, b_mass, out=np.zeros_like(B), where=b_mass > 0),
        1.0 / n_obs,
    )
    pi = pi / max(float(pi.sum()), _EPS)
    return DiscreteHMM(A, B, pi)
