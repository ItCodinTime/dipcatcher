"""Tests for models/bayesian.py — NIG conjugate OLS + BMA."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.bayesian import (
    bayes_log_ml,
    bayes_ols,
    bayes_predictive,
    bayesian_model_averaging,
)


def _fixture(n: int = 200, seed: int = 7) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, 3))
    beta = np.array([1.5, -2.0, 0.0])
    y = 0.5 + x @ beta + 0.5 * rng.standard_normal(n)
    return y, x, beta


def test_posterior_mean_near_ols() -> None:
    y, x, beta = _fixture()
    post = bayes_ols(y, x)
    # weak-ish prior -> posterior mean close to truth
    assert post["b_n"][1] == pytest.approx(beta[0], abs=0.15)
    assert post["b_n"][2] == pytest.approx(beta[1], abs=0.15)
    assert abs(post["b_n"][3]) < 0.2  # zero coefficient shrunk to 0


def test_predictive_covers() -> None:
    y, x, _ = _fixture()
    post = bayes_ols(y, x)
    pred = bayes_predictive(post, x[:50])
    cover = float(np.mean((y[:50] >= pred["lo_95"]) & (y[:50] <= pred["hi_95"])))
    assert cover > 0.8  # 95% intervals should mostly cover in-sample
    assert (pred["scale"] > 0).all()


def test_stronger_prior_shrinks() -> None:
    y, x, _ = _fixture()
    weak = bayes_ols(y, x, g=1e6)
    strong = bayes_ols(y, x, g=0.01)
    assert abs(strong["b_n"][1]) < abs(weak["b_n"][1])


def test_bma_finds_true_model() -> None:
    y, x, _ = _fixture(n=400, seed=13)
    out = bayesian_model_averaging(y, x)
    pip = np.asarray(out["pip"])
    # columns 0 and 1 are signal, column 2 is noise
    assert pip[0] > 0.9 and pip[1] > 0.9
    assert pip[2] < 0.5
    probs = np.asarray(out["model_probs"])
    assert probs.sum() == pytest.approx(1.0)
    best = int(np.argmax(probs))
    sub = np.asarray(out["subsets"])[best]
    assert set(sub[sub >= 0]) == {0, 1}


def test_log_ml_prefers_signal() -> None:
    y, x, _ = _fixture()
    ml_sig = bayes_log_ml(y, x[:, :1])
    ml_noise = bayes_log_ml(y, x[:, 2:3])
    assert ml_sig > ml_noise


def test_fail_closed() -> None:
    y, x, _ = _fixture(n=30)
    with pytest.raises(ValueError):
        bayes_ols(y[:10], x)  # mismatched rows
    with pytest.raises(ValueError):
        bayes_ols(np.full(30, np.nan), x)
    with pytest.raises(ValueError):
        bayes_ols(y, np.column_stack([x[:, 0], x[:, 0]]))  # singular design
    with pytest.raises(ValueError):
        bayesian_model_averaging(y, np.random.default_rng(0).standard_normal((30, 11)))  # p > 10
    with pytest.raises(ValueError):
        bayes_ols(y, x, a0=-1.0)
