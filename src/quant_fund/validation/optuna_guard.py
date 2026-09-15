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
) -> dict[str, Any]:
    if x_holdout is not None:
        raise ValueError("Holdout data must not be passed to tune_on_train")
    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective_factory(x_train, y_train), n_trials=n_trials, show_progress_bar=False)
    return {
        "best_params": study.best_params,
        "best_value": study.best_value,
        "n_trials": n_trials,
        "seed": seed,
    }
