"""Conformal test martingales for exchangeability / drift monitoring.

Vovk, Gammerman & Shafer (2005, "Algorithmic Learning in a Random World",
ch. 7); Vovk, Petej, Nouretdinov, Ahlberg, Carlsson & Gammerman (2021,
"Retrain or not retrain: conformal test martingales for change-point
detection", COPA, PMLR 152 — the Simple Jumper); Vovk (2021, "Testing
randomness online", Statistical Science 36(4)); Prinster et al. (2025,
WATCH — weighted-conformal test martingales for changepoint monitoring
under covariate shift, arXiv:2505.04608).

Under the null of exchangeability the SEQUENTIAL conformal p-values

    p_t = ( #{i <= t : s_i > s_t} + U_t * #{i <= t : s_i = s_t} ) / t,
    U_t ~ Uniform(0, 1) i.i.d.,

are i.i.d. Uniform(0, 1) (Vovk et al. 2005, Thm 8.1) whatever the
nonconformity score ``s``. Any betting function ``f: [0,1] -> [0, inf)``
with integral 1 makes ``M_t = prod_{i <= t} f_i(p_i)`` a nonnegative test
martingale with M_0 = 1, so by Ville's inequality
``P(sup_t M_t >= 1/alpha) <= alpha`` — an anytime-valid alarm at any
threshold. Implemented betting rules:

* ``power_martingale``   — f(p) = eps p^(eps - 1), fixed eps in (0, 1).
* ``mixture_martingale`` — integral over eps in (0, 1) of the power
  martingales (the "simple mixture"), evaluated by quadrature on the
  cumulative log p; parameter-free.
* ``simple_jumper``      — Vovk et al. 2021: capital spread over eps in
  {-1, 0, 1} for f(p) = 1 + eps (p - 1/2) with a jump rate J; adapts to
  the direction (small or large p-values) of the alternative.

``WatchMonitor`` follows the WATCH protocol: weighted conformal p-values
(likelihood-ratio weights for covariate shift; uniform weights recover the
plain conformal test martingale), a jumper martingale, an alarm when the
martingale first crosses ``1/alpha``, and a RESET of the calibration window
and the wealth after each alarm so the procedure keeps monitoring for the
next changepoint. Detection delay and false-alarm rate are the reported
diagnostics — no P&L. All functions fail closed on empty/non-finite input.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]

__all__ = [
    "WatchMonitor",
    "WatchResult",
    "conformal_p_values",
    "martingale_alarm",
    "mixture_martingale",
    "power_martingale",
    "simple_jumper",
    "weighted_conformal_p_value",
]


def _check_scores(scores: Array) -> Array:
    s = np.asarray(scores, dtype=float).ravel()
    if s.size == 0:
        raise ValueError("scores must be non-empty")
    if not np.all(np.isfinite(s)):
        raise ValueError("scores must be finite")
    return s


def _check_p(p: Array) -> Array:
    p = np.asarray(p, dtype=float).ravel()
    if p.size == 0:
        raise ValueError("p-values must be non-empty")
    if not np.all(np.isfinite(p)) or np.any(p < 0.0) or np.any(p > 1.0):
        raise ValueError("p-values must lie in [0, 1]")
    return p


def _jumper_update(capital: Array, eps: Array, jump: float, p: float) -> Array:
    """One Simple-Jumper step: mix ``jump`` of wealth, then multiply by ``f_eps(p)``."""
    total = float(np.sum(capital))
    k = int(capital.shape[0])
    mixed = (1.0 - jump) * capital + (jump * total / k)
    return np.asarray(mixed * (1.0 + eps * (p - 0.5)), dtype=float)


def conformal_p_values(scores: Array, seed: int = 0) -> Array:
    """Sequential smoothed conformal p-values; i.i.d. U(0,1) under exchangeability."""
    s = _check_scores(scores)
    rng = np.random.default_rng(seed)
    u = rng.uniform(size=s.size)
    p = np.empty(s.size)
    for t in range(s.size):
        past = s[: t + 1]
        greater = np.count_nonzero(past > s[t])
        ties = np.count_nonzero(past == s[t])
        p[t] = (greater + u[t] * ties) / (t + 1)
    return p


def weighted_conformal_p_value(
    past_scores: Array, new_score: float, weights: Array, u: float
) -> float:
    """Weighted conformal p-value (Tibshirani et al. 2019; WATCH eq. 3).

    ``weights`` are nonnegative likelihood-ratio weights for the past scores
    and the new one (``len(weights) == len(past_scores) + 1``); normalized
    internally.
    """
    past = np.asarray(past_scores, dtype=float).ravel()
    w = np.asarray(weights, dtype=float).ravel()
    if past.size and not np.all(np.isfinite(past)):
        raise ValueError("past_scores must be finite")
    if w.size != past.size + 1:
        raise ValueError("weights must have len(past_scores) + 1 entries")
    if np.any(w < 0.0) or not np.all(np.isfinite(w)) or w.sum() <= 0.0:
        raise ValueError("weights must be nonnegative, finite, with positive mass")
    if not 0.0 <= u <= 1.0:
        raise ValueError("u must be in [0, 1]")
    if not np.isfinite(new_score):
        raise ValueError("new_score must be finite")
    w = w / w.sum()
    all_s = np.concatenate([past, [new_score]])
    greater = float(np.sum(w[all_s > new_score]))
    ties = float(np.sum(w[all_s == new_score]))
    return float(greater + u * ties)


# Flooring p away from 0 understates p^(eps-1) (eps-1 < 0), so alarms stay conservative.
_P_FLOOR = 1e-300


def power_martingale(p: Array, eps: float = 0.5) -> Array:
    """M_t = prod_i eps * p_i^(eps-1); returned as the full path."""
    p = _check_p(p)
    if not 0.0 < eps < 1.0:
        raise ValueError("eps must be in (0, 1)")
    pc = np.clip(p, _P_FLOOR, 1.0)
    log_m = np.cumsum(np.log(eps) + (eps - 1.0) * np.log(pc))
    return np.asarray(np.exp(log_m), dtype=float)


def mixture_martingale(p: Array, n_grid: int = 200) -> Array:
    """Simple-mixture martingale: integral_0^1 prod eps p_i^(eps-1) d eps (path)."""
    p = _check_p(p)
    if n_grid < 8:
        raise ValueError("n_grid must be >= 8")
    pc = np.clip(p, _P_FLOOR, 1.0)
    cum_log_p = np.cumsum(np.log(pc))  # (t,)
    t = np.arange(1, p.size + 1, dtype=float)
    # midpoint rule on eps in (0, 1)
    eps = (np.arange(n_grid) + 0.5) / n_grid
    log_terms = t[:, None] * np.log(eps)[None, :] + (eps - 1.0)[None, :] * cum_log_p[:, None]
    m = np.max(log_terms, axis=1, keepdims=True)
    log_mix = m[:, 0] + np.log(np.mean(np.exp(log_terms - m), axis=1))
    return np.asarray(np.exp(log_mix), dtype=float)


def simple_jumper(
    p: Array, jump: float = 0.01, epsilons: tuple[float, ...] = (-1.0, 0.0, 1.0)
) -> Array:
    """Vovk et al. (2021) Simple Jumper martingale path.

    Capital C_eps over epsilons; each step first mixes a fraction ``jump`` of
    total capital uniformly across epsilons, then multiplies C_eps by
    f_eps(p) = 1 + eps (p - 1/2). The total capital is a test martingale.
    """
    p = _check_p(p)
    if not 0.0 <= jump <= 1.0:
        raise ValueError("jump must be in [0, 1]")
    eps = np.asarray(epsilons, dtype=float)
    if eps.size == 0 or not np.all(np.isfinite(eps)) or np.any(np.abs(eps) > 2.0):
        raise ValueError("epsilons must be non-empty, finite, and in [-2, 2] to keep f >= 0")
    k = int(eps.size)
    capital = np.full(k, 1.0 / k)
    path = np.empty(p.size)
    for i, pi in enumerate(p):
        capital = _jumper_update(capital, eps, jump, float(pi))
        path[i] = float(np.sum(capital))
    return path


def martingale_alarm(path: Array, alpha: float = 0.05) -> dict[str, object]:
    """First crossing of 1/alpha (Ville): index or None, max wealth, anytime p-value."""
    m = np.asarray(path, dtype=float).ravel()
    if m.size == 0 or not np.all(np.isfinite(m)) or np.any(m < 0.0):
        raise ValueError("path must be non-empty, finite, nonnegative")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    thr = 1.0 / alpha
    hits = np.flatnonzero(m >= thr)
    first = int(hits[0]) if hits.size else None
    running_max = float(np.max(m))
    return {
        "alarm_index": first,
        "alarmed": first is not None,
        "threshold": thr,
        "max_wealth": running_max,
        "anytime_p_value": float(min(1.0, 1.0 / running_max)) if running_max > 0 else 1.0,
    }


@dataclass
class WatchResult:
    p_values: Array
    wealth: Array
    alarms: list[int] = field(default_factory=list)


class WatchMonitor:
    """WATCH-style sequential monitor with reset-on-alarm.

    Parameters
    ----------
    alpha:
        Alarm level; alarm when wealth >= 1/alpha. Per-segment false alarm
        probability <= alpha under exchangeability (Ville).
    warmup:
        Minimum calibration scores held before betting starts after each
        reset; p-values during warm-up are still recorded, but wealth is
        held at 1.
    jump:
        Simple Jumper jump rate.
    max_window:
        Optional cap on the calibration window length (sliding window).
    """

    def __init__(
        self,
        alpha: float = 0.05,
        warmup: int = 20,
        jump: float = 0.01,
        max_window: int | None = None,
        seed: int = 0,
    ) -> None:
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must be in (0, 1)")
        if warmup < 1:
            raise ValueError("warmup must be >= 1")
        if not 0.0 <= jump <= 1.0:
            raise ValueError("jump must be in [0, 1]")
        if max_window is not None and max_window < warmup:
            raise ValueError("max_window must be >= warmup")
        self.alpha = float(alpha)
        self.warmup = int(warmup)
        self.jump = float(jump)
        self.max_window = max_window
        self._rng = np.random.default_rng(seed)
        self._eps = np.array([-1.0, 0.0, 1.0])
        self._reset()

    def _reset(self) -> None:
        self._window: list[float] = []
        self._weights: list[float] = []
        k = int(self._eps.shape[0])
        self._capital = np.full(k, 1.0 / k)
        self.wealth = 1.0

    def step(self, score: float, weight: float = 1.0) -> tuple[float, float, bool]:
        """Feed one nonconformity score (and optional LR weight). Returns (p, wealth, alarmed)."""
        if not np.isfinite(score):
            raise ValueError("score must be finite")
        if not np.isfinite(weight) or weight < 0.0:
            raise ValueError("weight must be nonnegative and finite")
        u = float(self._rng.uniform())
        w_all = np.asarray(self._weights + [weight], dtype=float)
        if w_all.sum() <= 0.0:
            p = u
        else:
            p = weighted_conformal_p_value(np.asarray(self._window), score, w_all, u)
        self._window.append(float(score))
        self._weights.append(float(weight))
        if self.max_window is not None and len(self._window) > self.max_window:
            self._window.pop(0)
            self._weights.pop(0)
        alarmed = False
        if len(self._window) > self.warmup:
            self._capital = _jumper_update(self._capital, self._eps, self.jump, p)
            self.wealth = float(np.sum(self._capital))
            if self.wealth >= 1.0 / self.alpha:
                alarmed = True
        out = (p, self.wealth, alarmed)
        if alarmed:
            self._reset()
        return out

    def run(self, scores: Array, weights: Array | None = None) -> WatchResult:
        s = _check_scores(scores)
        if weights is None:
            w = np.ones_like(s)
        else:
            w = np.asarray(weights, dtype=float).ravel()
            if w.shape != s.shape:
                raise ValueError("weights must match scores length")
        pv = np.empty(s.size)
        wealth = np.empty(s.size)
        alarms: list[int] = []
        for i in range(s.size):
            p, m, a = self.step(float(s[i]), float(w[i]))
            pv[i] = p
            wealth[i] = m
            if a:
                alarms.append(i)
        return WatchResult(p_values=pv, wealth=wealth, alarms=alarms)
