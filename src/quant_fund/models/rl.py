"""Contextual bandit for cross-sectional ranking. Research-only reward path.

LinUCB (Li et al., 2010) treats each name as an arm with shared linear
parameters. Reward is the scientific target (e.g. next-day idiosyncratic
return), not a trading P&L path. Trace fields never include live return metrics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from quant_fund.metrics.cross_section import _date_keys


@dataclass
class BanditTrace:
    dates: list[str]
    policy_reward: NDArray[np.float64]
    oracle_reward: NDArray[np.float64]
    uniform_reward: NDArray[np.float64]
    cumulative_regret: NDArray[np.float64]


class LinUCBRanker:
    def __init__(self, n_features: int, alpha: float = 1.0) -> None:
        d = int(n_features)
        if d <= 0:
            raise ValueError(f"n_features must be > 0, got {n_features}")
        a = float(alpha)
        if not np.isfinite(a) or a < 0.0:
            raise ValueError(f"alpha must be finite and >= 0, got {alpha}")
        self.n_features = d
        self.alpha = a
        self.a = np.eye(d)
        self.b = np.zeros(d)

    def _check_x(self, x: NDArray[np.float64], *, allow_batch: bool) -> NDArray[np.float64]:
        arr = np.asarray(x, dtype=float)
        if allow_batch:
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            if arr.ndim != 2 or arr.shape[1] != self.n_features:
                raise ValueError(
                    f"expected (*, {self.n_features}) features, got shape {np.asarray(x).shape}"
                )
        else:
            if arr.ndim != 1 or arr.shape[0] != self.n_features:
                raise ValueError(
                    f"expected {self.n_features} features, got shape {np.asarray(x).shape}"
                )
        return arr

    def scores(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        xx = self._check_x(x, allow_batch=True)
        xx = np.where(np.isfinite(xx), xx, 0.0)
        a_inv = np.linalg.inv(self.a)
        theta = a_inv @ self.b
        bonus = np.sqrt(np.clip(np.sum((xx @ a_inv) * xx, axis=1), 0.0, None))
        return xx @ theta + self.alpha * bonus

    def update(self, x: NDArray[np.float64], reward: float) -> None:
        v = self._check_x(x, allow_batch=False)
        v = np.where(np.isfinite(v), v, 0.0)
        self.a = self.a + np.outer(v, v)
        self.b = self.b + float(reward) * v


class LinearThompsonRanker(LinUCBRanker):
    """Bayesian linear Thompson sampler for contextual ranking.

    The posterior mean is ridge regression and the exploration covariance is
    proportional to ``A^-1``. Sampling is deterministic under the supplied
    seed, making this suitable for reproducible research diagnostics.
    """

    def __init__(self, n_features: int, alpha: float = 1.0, seed: int = 7) -> None:
        super().__init__(n_features, alpha)
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)

    def scores(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        xx = self._check_x(x, allow_batch=True)
        xx = np.where(np.isfinite(xx), xx, 0.0)
        a_inv = np.linalg.inv(self.a)
        mean = a_inv @ self.b
        covariance = (self.alpha**2) * a_inv
        sample = self.rng.multivariate_normal(mean, covariance, check_valid="ignore")
        return xx @ sample


def _topk_mean(y: NDArray[np.float64], scores: NDArray[np.float64], k: int) -> float:
    k = max(1, min(int(k), int(y.size)))
    idx = np.argsort(scores)[-k:]
    return float(np.mean(y[idx]))


def run_linucb_panel(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    dates: NDArray[Any],
    *,
    oracle: NDArray[np.float64] | None = None,
    top_k: int = 3,
    alpha: float = 1.0,
    seed: int = 7,
) -> BanditTrace:
    """Online LinUCB vs oracle and uniform. One update pass, chronological."""
    xx = np.asarray(x, dtype=float)
    yy = np.asarray(y, dtype=float)
    if xx.ndim != 2:
        raise ValueError(f"x must be 2-d panel, got shape {xx.shape}")
    if yy.ndim != 1 or yy.shape[0] != xx.shape[0]:
        raise ValueError(f"y length must match x rows ({xx.shape[0]}), got {yy.shape}")
    dates_arr = np.asarray(dates)
    if dates_arr.shape[0] != xx.shape[0]:
        raise ValueError(
            f"dates length must match x rows ({xx.shape[0]}), got {dates_arr.shape[0]}"
        )
    if oracle is not None:
        oo_all = np.asarray(oracle, dtype=float)
        if oo_all.shape[0] != xx.shape[0]:
            raise ValueError(
                f"oracle length must match x rows ({xx.shape[0]}), got {oo_all.shape[0]}"
            )
    else:
        oo_all = None
    if xx.shape[0] == 0:
        empty = np.asarray([], dtype=float)
        return BanditTrace(
            dates=[],
            policy_reward=empty,
            oracle_reward=empty,
            uniform_reward=empty,
            cumulative_regret=empty,
        )
    if xx.shape[1] <= 0:
        raise ValueError(f"x must have >=1 feature columns, got shape {xx.shape}")
    keys = _date_keys(dates_arr)
    order = sorted(set(keys))
    bandit = LinUCBRanker(xx.shape[1], alpha=alpha)
    rng = np.random.default_rng(seed)
    pol: list[float] = []
    ora: list[float] = []
    uni: list[float] = []
    kept: list[str] = []
    for key in order:
        mask = np.array([k == key for k in keys], dtype=bool)
        if int(mask.sum()) < max(top_k * 2, 4):
            continue
        x_day, y_day = xx[mask], yy[mask]
        finite = np.isfinite(y_day) & np.isfinite(x_day).all(axis=1)
        if int(finite.sum()) < max(top_k * 2, 4):
            continue
        x_day, y_day = x_day[finite], y_day[finite]
        scores = bandit.scores(x_day)
        pol.append(_topk_mean(y_day, scores, top_k))
        if oo_all is not None:
            oo = oo_all[mask][finite]
            ora.append(_topk_mean(y_day, oo, top_k))
        else:
            ora.append(_topk_mean(y_day, y_day, top_k))
        uni.append(_topk_mean(y_day, rng.normal(size=y_day.size), top_k))
        chosen = np.argsort(scores)[-top_k:]
        for i in chosen:
            bandit.update(x_day[i], float(y_day[i]))
        kept.append(key)
    policy = np.asarray(pol, dtype=float)
    oracle_r = np.asarray(ora, dtype=float)
    uniform = np.asarray(uni, dtype=float)
    regret = np.cumsum(oracle_r - policy)
    return BanditTrace(
        dates=kept,
        policy_reward=policy,
        oracle_reward=oracle_r,
        uniform_reward=uniform,
        cumulative_regret=regret,
    )


def run_thompson_panel(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    dates: NDArray[Any],
    *,
    oracle: NDArray[np.float64] | None = None,
    top_k: int = 3,
    alpha: float = 1.0,
    seed: int = 7,
) -> BanditTrace:
    """Online Thompson-sampling analogue of :func:`run_linucb_panel`."""
    xx = np.asarray(x, dtype=float)
    yy = np.asarray(y, dtype=float)
    dates_arr = np.asarray(dates)
    if xx.ndim != 2 or yy.ndim != 1 or yy.shape[0] != xx.shape[0]:
        raise ValueError("x must be 2-D and y must align with x")
    if dates_arr.shape[0] != xx.shape[0]:
        raise ValueError("dates must align with x")
    oo_all = None if oracle is None else np.asarray(oracle, dtype=float)
    if oo_all is not None and oo_all.shape[0] != xx.shape[0]:
        raise ValueError("oracle must align with x")
    if xx.shape[0] == 0:
        empty = np.asarray([], dtype=float)
        return BanditTrace([], empty, empty, empty, empty)
    keys = _date_keys(dates_arr)
    policy = LinearThompsonRanker(xx.shape[1], alpha=alpha, seed=seed)
    rng = np.random.default_rng(seed)
    pol: list[float] = []
    ora: list[float] = []
    uni: list[float] = []
    kept: list[str] = []
    for key in sorted(set(keys)):
        mask = np.array([k == key for k in keys], dtype=bool)
        if int(mask.sum()) < max(top_k * 2, 4):
            continue
        x_day, y_day = xx[mask], yy[mask]
        finite = np.isfinite(y_day) & np.isfinite(x_day).all(axis=1)
        if int(finite.sum()) < max(top_k * 2, 4):
            continue
        x_day, y_day = x_day[finite], y_day[finite]
        scores = policy.scores(x_day)
        pol.append(_topk_mean(y_day, scores, top_k))
        oo = y_day if oo_all is None else oo_all[mask][finite]
        ora.append(_topk_mean(y_day, oo, top_k))
        uni.append(_topk_mean(y_day, rng.normal(size=y_day.size), top_k))
        for i in np.argsort(scores)[-top_k:]:
            policy.update(x_day[i], float(y_day[i]))
        kept.append(key)
    policy_reward = np.asarray(pol, dtype=float)
    oracle_reward = np.asarray(ora, dtype=float)
    uniform = np.asarray(uni, dtype=float)
    return BanditTrace(
        kept,
        policy_reward,
        oracle_reward,
        uniform,
        np.cumsum(oracle_reward - policy_reward),
    )
