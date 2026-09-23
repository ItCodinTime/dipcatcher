"""Tests for models/ordered.py — ordered probit/logit."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.ordered import ordered_fit, ordered_predict


def _ordered_data(
    n: int = 800, seed: int = 0, link: str = "probit"
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = np.column_stack([rng.standard_normal(n)])  # no intercept: cuts absorb it
    beta = np.array([0.9])
    cuts = np.array([-0.5, 0.8, 1.8])
    xb = x @ beta
    e = rng.standard_normal(n) if link == "probit" else rng.logistic(size=n)
    ys = xb + e
    y = np.digitize(ys, cuts).astype(float)
    return y, x, beta, cuts


def test_ordered_probit_recovers() -> None:
    y, x, beta, cuts = _ordered_data()
    out = ordered_fit(y, x, link="probit")
    coef = np.asarray(out["coef"])
    assert abs(coef[0] - beta[0]) < 0.3
    est_cuts = np.asarray(out["cutpoints"])
    assert np.diff(est_cuts).min() > 0  # monotone
    assert np.abs(est_cuts - cuts).max() < 0.4


def test_ordered_logit() -> None:
    y, x, beta, _ = _ordered_data(seed=1, link="logit")
    out = ordered_fit(y, x, link="logit")
    assert abs(float(np.asarray(out["coef"])[0]) - beta[0]) < 0.4
    probs = np.asarray(out["probs"])
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-6)


def test_predict_probs() -> None:
    y, x, _, _ = _ordered_data(seed=2)
    out = ordered_fit(y, x)
    p = ordered_predict(out, x[:10])
    assert p.shape == (10, 4)
    assert np.allclose(p.sum(axis=1), 1.0)


def test_fail_closed() -> None:
    y, x, _, _ = _ordered_data()
    with pytest.raises(ValueError):
        ordered_fit((y > 1).astype(float), x)  # binary -> use probit
    with pytest.raises(ValueError):
        ordered_fit(y, np.column_stack([x, x[:, 0]]))  # rank-def
    with pytest.raises(ValueError):
        ordered_fit(y, x, link="bogus")
    with pytest.raises(ValueError):
        ordered_predict(ordered_fit(y, x), np.zeros((5, 3)))  # dim mismatch
