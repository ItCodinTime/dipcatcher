"""Optuna may only see training data. Holdout is not passed in."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import optuna


def tune_on_train(
    objective_factory: Callable[[np.ndarray, np.ndarray], Callable[[optuna.Trial], float]],
    x_train: np.ndarray,
    y_train: np.ndarray,
    n_trials: int,
    seed: int,
    *,
    x_holdout: np.ndarray | None = None,
    y_holdout: np.ndarray | None = None,
) -> dict[str, Any]:
    """Run Optuna on train only; fail closed on holdout leak / empty / bad space."""
    if x_holdout is not None or y_holdout is not None:
        raise ValueError("Holdout data must not be passed to tune_on_train")
    if int(n_trials) < 1:
        raise ValueError("n_trials must be >= 1")
    x = np.asarray(x_train)
    y = np.asarray(y_train).reshape(-1)
    if x.size == 0 or y.size == 0:
        raise ValueError("empty train arrays are not allowed for Optuna")
    if x.ndim < 1:
        raise ValueError("x_train must be at least 1-d")
    if x.shape[0] != y.shape[0]:
        raise ValueError("x_train and y_train length mismatch")
    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective_factory(x, y), n_trials=int(n_trials), show_progress_bar=False)
    if len(study.trials) == 0:
        raise ValueError("empty Optuna study after optimize")
    return {
        "best_params": study.best_params,
        "best_value": study.best_value,
        "n_trials": int(n_trials),
        "seed": seed,
    }
