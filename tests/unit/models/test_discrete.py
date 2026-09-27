"""Tests for models/discrete.py — probit, logit, Tobit."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from quant_fund.models.discrete import logit_fit, probit_fit, tobit_fit


def _binary_data(n: int = 800, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = np.column_stack([np.ones(n), rng.standard_normal(n)])
    beta = np.array([-0.2, 1.2])
    p = stats.norm.cdf(x @ beta)
    y = (rng.random(n) < p).astype(float)
    return y, x


def test_probit_recovers_beta() -> None:
    y, x = _binary_data()
    out = probit_fit(y, x)
    coef = np.asarray(out["coef"])
    assert abs(coef[1] - 1.2) < 0.35
    assert 0.0 < out["mcfadden_r2"] < 0.9
    assert (np.asarray(out["se"]) > 0).all()


def test_logit_recovers_beta() -> None:
    rng = np.random.default_rng(1)
    n = 800
    x = np.column_stack([np.ones(n), rng.standard_normal(n)])
    beta = np.array([0.3, -1.5])
    p = 1.0 / (1.0 + np.exp(-(x @ beta)))
    y = (rng.random(n) < p).astype(float)
    out = logit_fit(y, x)
    coef = np.asarray(out["coef"])
    assert abs(coef[1] + 1.5) < 0.4
    fitted = np.asarray(out["fitted"])
    assert ((fitted > 0) & (fitted < 1)).all()


def test_tobit_recovers() -> None:
    rng = np.random.default_rng(2)
    n = 1000
    x = np.column_stack([np.ones(n), rng.standard_normal(n)])
    beta = np.array([0.5, 0.8])
    sig = 0.7
    ys = x @ beta + sig * rng.standard_normal(n)
    y = np.maximum(ys, 0.0)
    out = tobit_fit(y, x)
    coef = np.asarray(out["coef"])
    assert abs(coef[1] - 0.8) < 0.15
    assert abs(float(out["sigma"]) - sig) < 0.15
    assert 0 < out["censored_share"] < 1


def test_mcfadden_bounds() -> None:
    y, x = _binary_data(seed=4)
    for fit in (probit_fit, logit_fit):
        r2 = float(fit(y, x)["mcfadden_r2"])
        assert 0.0 <= r2 <= 1.0


def test_fail_closed() -> None:
    y, x = _binary_data()
    with pytest.raises(ValueError):
        probit_fit(y * 2.0, x)  # not binary
    with pytest.raises(ValueError):
        logit_fit(np.ones_like(y), x)  # single class
    with pytest.raises(ValueError):
        probit_fit(y, np.column_stack([x, x[:, 1]]))  # collinear
    unc = np.abs(np.random.default_rng(0).standard_normal(200)) + 0.5
    with pytest.raises(ValueError):
        tobit_fit(unc, x[:200])  # no censored mass
