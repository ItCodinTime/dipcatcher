"""Contextual bandit for cross-sectional ranking. No portfolio Sharpe.

LinUCB (Li et al., 2010) treats each name as an arm with shared linear
parameters. Reward is the scientific target (e.g. next-day idiosyncratic
return), not a trading P&L path.
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
        self.alpha = float(alpha)
        self.a = np.eye(d)
        self.b = np.zeros(d)

    def scores(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        xx = np.where(np.isfinite(x), x, 0.0)
        a_inv = np.linalg.inv(self.a)
        theta = a_inv @ self.b
        bonus = np.sqrt(np.clip(np.sum((xx @ a_inv) * xx, axis=1), 0.0, None))
        return xx @ theta + self.alpha * bonus

    def update(self, x: NDArray[np.float64], reward: float) -> None:
        v = np.where(np.isfinite(x), x, 0.0)
        self.a = self.a + np.outer(v, v)
        self.b = self.b + float(reward) * v


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
    keys = _date_keys(dates)
    order = sorted(set(keys))
    bandit = LinUCBRanker(x.shape[1], alpha=alpha)
    rng = np.random.default_rng(seed)
    pol: list[float] = []
    ora: list[float] = []
    uni: list[float] = []
    kept: list[str] = []
    for key in order:
        mask = np.array([k == key for k in keys], dtype=bool)
        if int(mask.sum()) < max(top_k * 2, 4):
            continue
        xx, yy = x[mask], y[mask]
        finite = np.isfinite(yy) & np.isfinite(xx).all(axis=1)
        if int(finite.sum()) < max(top_k * 2, 4):
            continue
        xx, yy = xx[finite], yy[finite]
        scores = bandit.scores(xx)
        pol.append(_topk_mean(yy, scores, top_k))
        if oracle is not None:
            oo = oracle[mask][finite]
            ora.append(_topk_mean(yy, oo, top_k))
        else:
            ora.append(_topk_mean(yy, yy, top_k))
        uni.append(_topk_mean(yy, rng.normal(size=yy.size), top_k))
        chosen = np.argsort(scores)[-top_k:]
        for i in chosen:
            bandit.update(xx[i], float(yy[i]))
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
