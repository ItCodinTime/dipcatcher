import numpy as np
import pytest

from quant_fund.validation.optuna_guard import tune_on_train


def test_tune_on_train_rejects_holdout_input() -> None:
    with pytest.raises(ValueError, match="Holdout data"):
        tune_on_train(
            lambda _x, _y: lambda _trial: 0.0,
            np.ones((4, 1)),
            np.ones(4),
            1,
            7,
            x_holdout=np.ones((1, 1)),
        )


def test_tune_on_train_rejects_y_holdout() -> None:
    with pytest.raises(ValueError, match="Holdout data"):
        tune_on_train(
            lambda _x, _y: lambda _trial: 0.0,
            np.ones((4, 1)),
            np.ones(4),
            1,
            7,
            y_holdout=np.ones(1),
        )


def test_tune_on_train_rejects_empty_train() -> None:
    with pytest.raises(ValueError, match="empty train"):
        tune_on_train(
            lambda _x, _y: lambda _trial: 0.0,
            np.ones((0, 1)),
            np.ones(0),
            2,
            1,
        )


def test_tune_on_train_rejects_length_mismatch() -> None:
    with pytest.raises(ValueError, match="length mismatch"):
        tune_on_train(
            lambda _x, _y: lambda _trial: 0.0,
            np.ones((4, 1)),
            np.ones(3),
            2,
            1,
        )


def test_tune_on_train_rejects_bad_n_trials() -> None:
    with pytest.raises(ValueError, match="n_trials"):
        tune_on_train(
            lambda _x, _y: lambda _trial: 0.0,
            np.ones((4, 1)),
            np.ones(4),
            0,
            1,
        )
    with pytest.raises(ValueError, match="n_trials"):
        tune_on_train(
            lambda _x, _y: lambda _trial: 0.0,
            np.ones((4, 1)),
            np.ones(4),
            -2,
            1,
        )


def test_tune_on_train_records_trial_provenance() -> None:
    def objective_factory(x: np.ndarray, y: np.ndarray):
        assert x.shape == (4, 1)
        assert y.shape == (4,)

        def objective(trial):
            return float(trial.suggest_float("score", 0.0, 1.0))

        return objective

    result = tune_on_train(objective_factory, np.ones((4, 1)), np.ones(4), 3, 11)
    assert result["n_trials"] == 3
    assert result["seed"] == 11
    assert "score" in result["best_params"]
