"""Tests for models/fractional.py — Papke-Wooldridge fractional response."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.fractional import fractional_fit


def _frac_data(n: int = 600, seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = np.column_stack([np.ones(n), rng.standard_normal(n)])
    beta = np.array([0.2, 0.8])
    mu = 1.0 / (1.0 + np.exp(-(x @ beta)))
    # fractional outcome: Beta-distributed around mu
    conc = 20.0
    y = rng.beta(mu * conc, (1 - mu) * conc)
    return y, x, beta


def test_fractional_recovers() -> None:
    y, x, beta = _frac_data()
    out = fractional_fit(y, x)
    coef = np.asarray(out["coef"])
    assert np.abs(coef - beta).max() < 0.35
    assert out["r2"] > 0.4
    fitted = np.asarray(out["fitted"])
    assert ((fitted > 0) & (fitted < 1)).all()


def test_probit_link() -> None:
    y, x, _ = _frac_data(seed=1)
    out = fractional_fit(y, x, link="probit")
    assert np.isfinite(out["loglik"])
    assert out["r2"] > 0.3


def test_binary_outcome_compatible() -> None:
    # QMLE is also valid for pure 0/1 data
    rng = np.random.default_rng(2)
    n = 400
    x = np.column_stack([np.ones(n), rng.standard_normal(n)])
    y = (rng.random(n) < 1.0 / (1.0 + np.exp(-(x @ np.array([0.0, 1.0]))))).astype(float)
    out = fractional_fit(y, x)
    assert abs(float(np.asarray(out["coef"])[1]) - 1.0) < 0.4


def test_fail_closed() -> None:
    y, x, _ = _frac_data()
    with pytest.raises(ValueError):
        fractional_fit(y * 1.5, x)  # outside [0,1]
    with pytest.raises(ValueError):
        fractional_fit(y, np.column_stack([x, x[:, 1]]))  # rank-def
    with pytest.raises(ValueError):
        fractional_fit(y, x, link="bogus")
    with pytest.raises(ValueError):
        fractional_fit(y[:20], x[:20])
