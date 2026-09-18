"""Quantile Thompson sampling for cross-sectional ranking.

Shared linear pinball models (IRLS ridge) with posterior draws on a tau-grid.
Reward is the scientific target (e.g. next residual), not a portfolio path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


@dataclass
class QuantileBanditTrace:
    dates: list[str]
    policy_reward: NDArray[np.float64]
    oracle_reward: NDArray[np.float64]
    uniform_reward: NDArray[np.float64]
    cumulative_regret: NDArray[np.float64]


def _as_key(date: object) -> str:
    if isinstance(date, np.datetime64):
        return str(date)
    iso = getattr(date, "isoformat", None)
    if callable(iso):
        return str(iso())
    return str(date)


def _sort_value(date: object) -> tuple[int, float | str]:
    if isinstance(date, np.datetime64):
        return (0, float(date.astype("datetime64[ns]").astype(np.int64)))
    if isinstance(date, (np.integer, int)):
        return (0, float(date))
    if isinstance(date, (np.floating, float)):
        value = float(date)
        if np.isfinite(value):
            return (0, value)
        return (1, str(date))
    stamp = getattr(date, "timestamp", None)
    if callable(stamp):
        try:
            return (0, float(stamp()))
        except (OSError, TypeError, ValueError):
            pass
    text = _as_key(date)
    try:
        return (0, float(text))
    except ValueError:
        return (1, text)


def _ordered_groups(dates: NDArray[Any] | list[object]) -> tuple[list[str], list[str]]:
    keys = [_as_key(d) for d in dates]
    first: dict[str, object] = {}
    for key, raw in zip(keys, dates, strict=False):
        if key not in first:
            first[key] = raw
    order = sorted(first, key=lambda key: _sort_value(first[key]))
    return keys, order


def _finite_row(x: NDArray[np.float64]) -> NDArray[np.float64]:
    v = np.asarray(x, dtype=float).reshape(-1)
    return np.where(np.isfinite(v), v, 0.0)


def _irls_pinball(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    tau: float,
    ridge: float,
    beta0: NDArray[np.float64],
    *,
    n_iter: int = 8,
    eps: float = 1e-4,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Asymmetric IRLS for ridge pinball / linear quantile regression."""
    d = int(x.shape[1])
    beta = np.asarray(beta0, dtype=float).copy()
    eye = np.eye(d)
    prec = ridge * eye
    for _ in range(n_iter):
        resid = y - x @ beta
        weight = np.where(resid >= 0.0, tau, 1.0 - tau) / np.maximum(np.abs(resid), eps)
        xtw = x.T * weight
        prec = xtw @ x + ridge * eye
        beta = np.linalg.solve(prec, xtw @ y)
    return beta, prec


def _draw_gaussian(
    mean: NDArray[np.float64],
    prec: NDArray[np.float64],
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    """Draw z ~ N(mean, prec^{-1}) via Cholesky of the precision."""
    d = int(mean.shape[0])
    jittered = prec + 1e-8 * np.eye(d)
    try:
        lower = np.linalg.cholesky(jittered)
    except np.linalg.LinAlgError:
        return mean
    noise = rng.standard_normal(d)
    return mean + np.linalg.solve(lower.T, noise)


def _topk_indices(scores: NDArray[np.float64], k: int) -> NDArray[np.intp]:
    k = max(1, min(int(k), int(scores.size)))
    return np.argsort(scores)[-k:]


def _topk_mean(y: NDArray[np.float64], scores: NDArray[np.float64], k: int) -> float:
    idx = _topk_indices(scores, k)
    return float(np.mean(y[idx]))


class QuantileThompson:
    """Shared linear quantile Thompson sampler (sibling to LinUCB).

    For each tau on a uniform grid, fit ridge pinball by IRLS, warm-started
    from Bayesian linear regression. ``select`` draws one tau and a Gaussian
    posterior of that quantile's coefficients, then returns top-k indices.
    """

    def __init__(self, n_quantiles: int = 9, ridge: float = 1.0, seed: int = 0) -> None:
        if int(n_quantiles) < 1:
            raise ValueError("n_quantiles must be >= 1")
        self.n_quantiles = int(n_quantiles)
        self.ridge = float(ridge)
        self.seed = int(seed)
        m = self.n_quantiles
        self.taus = (np.arange(m, dtype=float) + 1.0) / (m + 1.0)
        self.rng = np.random.default_rng(self.seed)
        self._uniform_rng = np.random.default_rng(self.seed + 1)
        self._d: int | None = None
        self._xs: list[NDArray[np.float64]] = []
        self._ys: list[float] = []
        self._a: NDArray[np.float64] | None = None
        self._b: NDArray[np.float64] | None = None
        self._beta: NDArray[np.float64] | None = None
        self._prec: NDArray[np.float64] | None = None
        self._dirty = False

    def _ensure_dim(self, d: int) -> None:
        if self._d is None:
            self._d = int(d)
            eye = self.ridge * np.eye(self._d)
            self._a = eye.copy()
            self._b = np.zeros(self._d)
            self._beta = np.zeros((self.n_quantiles, self._d))
            self._prec = np.stack([eye.copy() for _ in range(self.n_quantiles)])
            self._dirty = False
            return
        if int(d) != self._d:
            raise ValueError(f"expected {self._d} features, got {d}")

    def _mean_theta(self) -> NDArray[np.float64]:
        if self._a is None or self._b is None:
            raise RuntimeError("quantile bandit has no initialized posterior")
        return np.linalg.solve(self._a, self._b)

    def _refit(self) -> None:
        if not self._dirty:
            return
        self._dirty = False
        if self._d is None:
            raise RuntimeError("quantile bandit has no initialized feature dimension")
        if not self._xs:
            return
        x = np.stack(self._xs, axis=0)
        y = np.asarray(self._ys, dtype=float)
        theta = self._mean_theta()
        betas = np.empty((self.n_quantiles, self._d), dtype=float)
        precs = np.empty((self.n_quantiles, self._d, self._d), dtype=float)
        for i, tau in enumerate(self.taus):
            beta, prec = _irls_pinball(x, y, float(tau), self.ridge, theta)
            betas[i] = beta
            precs[i] = prec
        self._beta = betas
        self._prec = precs

    def select(self, context_matrix: NDArray[np.float64], k: int) -> NDArray[np.intp]:
        xx = np.asarray(context_matrix, dtype=float)
        if xx.ndim == 1:
            xx = xx.reshape(1, -1)
        xx = np.where(np.isfinite(xx), xx, 0.0)
        self._ensure_dim(int(xx.shape[1]))
        self._refit()
        if self._beta is None or self._prec is None:
            raise RuntimeError("quantile bandit refit produced incomplete state")
        tau_idx = int(self.rng.integers(0, self.n_quantiles))
        beta = _draw_gaussian(self._beta[tau_idx], self._prec[tau_idx], self.rng)
        scores = xx @ beta
        return _topk_indices(scores, k)

    def update(self, context: NDArray[np.float64], reward: float) -> None:
        if not np.isfinite(reward):
            return
        v = _finite_row(context)
        self._ensure_dim(int(v.size))
        if self._a is None or self._b is None:
            raise RuntimeError("quantile bandit has no initialized posterior")
        self._a = self._a + np.outer(v, v)
        self._b = self._b + float(reward) * v
        self._xs.append(v)
        self._ys.append(float(reward))
        self._dirty = True

    def run_panel(
        self,
        x: NDArray[np.float64],
        y: NDArray[np.float64],
        dates: NDArray[Any],
        k: int,
        oracle: NDArray[np.float64] | None = None,
    ) -> QuantileBanditTrace:
        """Walk dates in time order. Score each group, then update the chosen arms.

        ``oracle`` is a score column aligned to ``y`` (planted when present).
        When omitted, regret uses hindsight y-greedy — same construction as LinUCB.
        """
        keys, order = _ordered_groups(dates)
        pol: list[float] = []
        ora: list[float] = []
        uni: list[float] = []
        kept: list[str] = []
        xx_all = np.asarray(x, dtype=float)
        yy_all = np.asarray(y, dtype=float)
        oracle_all = None if oracle is None else np.asarray(oracle, dtype=float)
        for key in order:
            mask = np.array([item == key for item in keys], dtype=bool)
            if int(mask.sum()) < max(int(k) * 2, 4):
                continue
            xx, yy = xx_all[mask], yy_all[mask]
            finite = np.isfinite(yy) & np.isfinite(xx).all(axis=1)
            if int(finite.sum()) < max(int(k) * 2, 4):
                continue
            xx, yy = xx[finite], yy[finite]
            chosen = self.select(xx, k)
            pol.append(float(np.mean(yy[chosen])))
            if oracle_all is not None:
                oo = oracle_all[mask][finite]
                ora.append(_topk_mean(yy, oo, k))
            else:
                ora.append(_topk_mean(yy, yy, k))
            uni.append(_topk_mean(yy, self._uniform_rng.normal(size=yy.size), k))
            for i in chosen:
                self.update(xx[i], float(yy[i]))
            kept.append(key)
        policy = np.asarray(pol, dtype=float)
        oracle_r = np.asarray(ora, dtype=float)
        uniform = np.asarray(uni, dtype=float)
        regret = np.cumsum(oracle_r - policy)
        return QuantileBanditTrace(
            dates=kept,
            policy_reward=policy,
            oracle_reward=oracle_r,
            uniform_reward=uniform,
            cumulative_regret=regret,
        )
