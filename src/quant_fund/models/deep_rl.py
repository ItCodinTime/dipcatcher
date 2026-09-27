"""Optional neural contextual-bandit policy.

This module is deliberately separate from execution: rewards are research
targets, dates are processed chronologically, and no output is a live-P&L
claim. PyTorch is imported lazily so the core package remains usable without
the optional ``nn`` dependency.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class PolicyGradientTrace:
    dates: list[str]
    policy_reward: NDArray[np.float64]
    uniform_reward: NDArray[np.float64]
    cumulative_regret: NDArray[np.float64]


class PolicyGradientRanker:
    """Small MLP policy trained on cross-sectional expected reward.

    The objective maximizes the expected reward of a softmax top-k policy with
    entropy regularization. This is a contextual policy-gradient surrogate,
    not a portfolio optimizer or trading simulator.
    """

    def __init__(
        self,
        n_features: int,
        *,
        hidden: tuple[int, int] = (64, 32),
        learning_rate: float = 1e-3,
        epochs: int = 80,
        entropy_weight: float = 1e-3,
        seed: int = 7,
    ) -> None:
        if int(n_features) <= 0 or int(epochs) < 1:
            raise ValueError("n_features must be positive and epochs must be positive")
        if not np.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive")
        if not np.isfinite(entropy_weight) or entropy_weight < 0:
            raise ValueError("entropy_weight must be finite and non-negative")
        import torch
        from torch import nn

        torch.manual_seed(int(seed))
        self.n_features = int(n_features)
        self.epochs = int(epochs)
        self.entropy_weight = float(entropy_weight)
        self.seed = int(seed)
        self.model = nn.Sequential(
            nn.Linear(self.n_features, int(hidden[0])),
            nn.Tanh(),
            nn.Linear(int(hidden[0]), int(hidden[1])),
            nn.Tanh(),
            nn.Linear(int(hidden[1]), 1),
        )
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=float(learning_rate))

    def fit(
        self,
        x: NDArray[np.float64],
        y: NDArray[np.float64],
        dates: NDArray[Any],
        *,
        top_k: int = 3,
    ) -> PolicyGradientRanker:
        import torch

        xx = np.asarray(x, dtype=np.float32)
        yy = np.asarray(y, dtype=np.float32).reshape(-1)
        dd = np.asarray(dates)
        if xx.ndim != 2 or xx.shape[1] != self.n_features or yy.shape[0] != xx.shape[0]:
            raise ValueError("policy-gradient inputs have incompatible shapes")
        if dd.shape[0] != xx.shape[0]:
            raise ValueError("dates must align with policy-gradient inputs")
        if int(top_k) < 1:
            raise ValueError("top_k must be positive")
        groups = []
        for date in sorted(set(dd.tolist())):
            mask = dd == date
            finite = np.isfinite(yy[mask]) & np.isfinite(xx[mask]).all(axis=1)
            if int(finite.sum()) >= max(2 * int(top_k), 4):
                groups.append((xx[mask][finite], yy[mask][finite]))
        if not groups:
            raise ValueError("policy-gradient training produced no evaluable dates")
        self.model.train()
        for _ in range(self.epochs):
            losses = []
            for x_day, y_day in groups:
                features = torch.as_tensor(x_day)
                rewards = torch.as_tensor(y_day)
                logits = self.model(features).reshape(-1)
                probs = torch.softmax(logits, dim=0)
                expected = torch.sum(probs * rewards)
                entropy = -torch.sum(probs * torch.log(probs.clamp_min(1e-8)))
                losses.append(-expected - self.entropy_weight * entropy)
            self.optimizer.zero_grad()
            torch.stack(losses).mean().backward()
            self.optimizer.step()
        return self

    def predict(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        import torch

        xx = np.asarray(x, dtype=np.float32)
        if xx.ndim != 2 or xx.shape[1] != self.n_features:
            raise ValueError("prediction features do not match policy shape")
        self.model.eval()
        with torch.no_grad():
            scores = self.model(torch.as_tensor(np.nan_to_num(xx))).reshape(-1)
        return np.asarray(scores.cpu().numpy(), dtype=float)

    def metadata(self) -> dict[str, Any]:
        return {
            "family": "reinforcement",
            "name": "policy_gradient",
            "seed": self.seed,
            "research_only": True,
            "live_pnl_claim": False,
        }


def run_policy_gradient_panel(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    dates: NDArray[Any],
    *,
    top_k: int = 3,
    seed: int = 7,
    epochs: int = 80,
) -> PolicyGradientTrace:
    """Evaluate a policy with strictly pre-date training data.

    Each date is scored by a policy fit only on earlier eligible dates. The
    current date is appended to history only after its reward is recorded.
    This prevents the apparently strong but invalid in-sample result that a
    fit-on-all-rows evaluator would produce.
    """
    xx = np.asarray(x, dtype=float)
    yy = np.asarray(y, dtype=float).reshape(-1)
    dd = np.asarray(dates)
    rng = np.random.default_rng(seed)
    history_x: list[NDArray[np.float64]] = []
    history_y: list[NDArray[np.float64]] = []
    history_d: list[NDArray[Any]] = []
    dates_kept: list[str] = []
    policy_rewards: list[float] = []
    uniform_rewards: list[float] = []
    for date in sorted(set(dd.tolist())):
        mask = dd == date
        finite = np.isfinite(yy[mask]) & np.isfinite(xx[mask]).all(axis=1)
        if int(finite.sum()) < max(2 * top_k, 4):
            continue
        x_day, y_day = xx[mask][finite], yy[mask][finite]
        if history_x:
            train_x = np.concatenate(history_x)
            train_y = np.concatenate(history_y)
            train_d = np.concatenate(history_d)
            try:
                policy = PolicyGradientRanker(xx.shape[1], seed=seed, epochs=epochs).fit(
                    train_x, train_y, train_d, top_k=top_k
                )
            except ValueError:
                policy = None
        else:
            policy = None
        if policy is None:
            history_x.append(x_day)
            history_y.append(y_day)
            history_d.append(np.full(x_day.shape[0], date, dtype=object))
            continue
        scores = policy.predict(x_day)
        chosen = np.argsort(scores)[-top_k:]
        random_choice = np.argsort(rng.normal(size=y_day.size))[-top_k:]
        policy_rewards.append(float(np.mean(y_day[chosen])))
        uniform_rewards.append(float(np.mean(y_day[random_choice])))
        dates_kept.append(str(date))
        history_x.append(x_day)
        history_y.append(y_day)
        history_d.append(np.full(x_day.shape[0], date, dtype=object))
    policy_reward = np.asarray(policy_rewards, dtype=float)
    uniform = np.asarray(uniform_rewards, dtype=float)
    return PolicyGradientTrace(
        dates_kept,
        policy_reward,
        uniform,
        np.cumsum(uniform - policy_reward),
    )
