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

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]
IntArray = NDArray[np.intp]
_EPS = 1e-15


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

    @property
    def n_states(self) -> int:
        return int(self.A.shape[0])

    @property
    def n_obs(self) -> int:
        return int(self.B.shape[1])

    def as_dict(self) -> dict[str, list]:
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
) -> tuple[DiscreteHMM, list[float]]:
    """Forward–backward (Fig. A.14). Likelihood is non-decreasing on one sequence."""
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
        gamma, xi, p_o = gamma_xi(current, o)
        history.append(p_o)
        if xi.size:
            A_hat = xi.sum(axis=0)
            A_hat = A_hat / np.maximum(A_hat.sum(axis=1, keepdims=True), _EPS)
        else:
            A_hat = current.A
        B_hat = np.zeros((current.n_states, current.n_obs), dtype=float)
        for vk in range(current.n_obs):
            mask = o == vk
            B_hat[:, vk] = gamma[mask].sum(axis=0) if mask.any() else 0.0
        B_hat = B_hat / np.maximum(gamma.sum(axis=0)[:, None], _EPS)
        pi_hat = gamma[0] / max(float(gamma[0].sum()), _EPS)
        current = DiscreteHMM(A_hat, B_hat, pi_hat)
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
    A = A / np.maximum(A.sum(axis=1, keepdims=True), _EPS)
    B = B / np.maximum(B.sum(axis=1, keepdims=True), _EPS)
    pi = pi / max(float(pi.sum()), _EPS)
    return DiscreteHMM(A, B, pi)
