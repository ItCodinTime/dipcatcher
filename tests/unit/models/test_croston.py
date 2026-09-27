"""Tests for models/croston.py — Croston / SBA / TSB intermittent demand."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.croston import croston_fit, croston_forecast


def _intermittent(n: int = 2000, p: float = 0.3, mu: float = 4.0, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    occur = rng.random(n) < p
    sizes = rng.lognormal(mean=np.log(mu), sigma=0.3, size=n)
    y = np.where(occur, sizes, 0.0)
    if not y.any():
        y[0] = mu
    return y


def test_croston_rate_matches_mean_demand() -> None:
    y = _intermittent()
    fit = croston_fit(y, alpha=0.1, variant="croston")
    assert fit.rate > 0
    # per-period rate should be close to the empirical mean demand
    assert abs(fit.rate - y.mean()) < 0.4 * y.mean()
    fc = croston_forecast(fit, 7)
    assert fc.shape == (7,)
    assert np.allclose(fc, fit.rate)


def test_sba_is_bias_deflated() -> None:
    y = _intermittent(seed=2)
    croston = croston_fit(y, alpha=0.2, variant="croston")
    sba = croston_fit(y, alpha=0.2, variant="sba")
    assert sba.rate < croston.rate
    assert np.isclose(sba.rate, (1.0 - 0.2 / 2.0) * croston.rate)


def test_tsb_runs_and_is_positive() -> None:
    y = _intermittent(seed=5)
    fit = croston_fit(y, alpha=0.1, variant="tsb")
    assert fit.rate > 0
    assert 0.0 <= fit.prob_level <= 1.0
    assert np.isfinite(fit.fitted).all()


def test_fail_closed() -> None:
    with pytest.raises(ValueError):
        croston_fit(np.array([1.0, -1.0, 0.0, 2.0]))  # negative
    with pytest.raises(ValueError):
        croston_fit(np.zeros(10))  # all zero
    with pytest.raises(ValueError):
        croston_fit(_intermittent(), alpha=1.5)  # bad alpha
    with pytest.raises(ValueError):
        croston_fit(_intermittent(), variant="nope")
