"""Tests for models/count.py — Poisson, NB2, ZIP."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.count import nb2_fit, poisson_fit, zip_fit


def _poisson_data(n: int = 500, seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = np.column_stack([np.ones(n), rng.standard_normal(n)])
    beta = np.array([0.8, 0.5])
    mu = np.exp(x @ beta)
    return rng.poisson(mu).astype(float), x, beta


def test_poisson_recovers() -> None:
    y, x, beta = _poisson_data()
    out = poisson_fit(y, x)
    coef = np.asarray(out["coef"])
    assert np.abs(coef - beta).max() < 0.25
    assert (np.asarray(out["se"]) > 0).all()
    assert np.isfinite(out["aic"])


def test_nb2_overdispersion() -> None:
    # NB data (alpha>0): Poisson fit should show dispersion > 1
    rng = np.random.default_rng(1)
    n = 600
    x = np.column_stack([np.ones(n), rng.standard_normal(n)])
    mu = np.exp(np.array([1.0, 0.4]) @ x.T)
    r = 3.0  # alpha = 1/r
    y = rng.negative_binomial(r, r / (r + mu)).astype(float)
    out_nb = nb2_fit(y, x)
    out_pois = poisson_fit(y, x)
    assert out_nb["alpha"] > 0.1
    assert out_nb["aic"] < out_pois["aic"]  # NB2 fits overdispersed data better


def test_zip_inflation() -> None:
    rng = np.random.default_rng(2)
    n = 800
    x = np.column_stack([np.ones(n), rng.standard_normal(n)])
    mu = np.exp(x @ np.array([1.0, 0.3]))
    w_true = 0.35
    y = np.where(rng.random(n) < w_true, 0, rng.poisson(mu)).astype(float)
    out = zip_fit(y, x)
    assert abs(out["w"] - w_true) < 0.15
    assert abs(float(np.asarray(out["coef"])[0]) - 1.0) < 0.3


def test_fail_closed() -> None:
    y, x, _ = _poisson_data()
    with pytest.raises(ValueError):
        poisson_fit(y + 0.5, x)  # non-integer
    with pytest.raises(ValueError):
        poisson_fit(-y, x)  # negative
    with pytest.raises(ValueError):
        nb2_fit(y, np.column_stack([x, x[:, 1]]))  # rank-def
    with pytest.raises(ValueError):
        zip_fit(y[:20], x[:20])  # too small
