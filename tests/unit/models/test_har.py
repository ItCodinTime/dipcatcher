"""Tests for models/har.py — Corsi HAR-RV + BPQ HARQ."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.har import har_forecast, har_rv_fit, harq_fit, harq_forecast


def _rv_series(n: int = 400, seed: int = 0) -> np.ndarray:
    """Simulated RV with multi-scale persistence."""
    rng = np.random.default_rng(seed)
    rv = np.empty(n)
    slow = np.empty(n)
    slow[0] = rv[0] = 1.0
    for t in range(1, n):
        slow[t] = 0.95 * slow[t - 1] + 0.05 * 1.0 + 0.1 * rng.standard_normal()
        slow[t] = max(slow[t], 0.1)
        rv[t] = 0.3 * rv[t - 1] + 0.4 * slow[t] + 0.3 * rng.lognormal(0, 0.3)
    return np.maximum(rv, 0.01)


def test_har_fit_recovers_structure() -> None:
    rv = _rv_series()
    out = har_rv_fit(rv)
    coef = np.asarray(out["coef"])
    assert coef.shape == (4,)
    assert out["r2"] > 0.5
    assert (np.asarray(out["se"]) > 0).all()
    # daily + weekly + monthly should carry the load
    assert coef[1] + coef[2] + coef[3] > 0.5


def test_har_forecast() -> None:
    rv = _rv_series(seed=1)
    out = har_rv_fit(rv[:-30])
    f = har_forecast(out, rv[:-30])
    assert f > 0.0 and np.isfinite(f)


def test_harq_fit() -> None:
    rv = _rv_series(seed=2)
    rng = np.random.default_rng(2)
    rq = rv * (0.5 + 0.5 * rng.random(rv.size))  # synthetic quarticity
    out = harq_fit(rv, rq)
    coef = np.asarray(out["coef"])
    assert coef.shape == (5,)
    assert np.isfinite(out["r2"])
    f = harq_forecast(out, rv, rq)
    assert f > 0.0


def test_fail_closed() -> None:
    with pytest.raises(ValueError):
        har_rv_fit(np.random.default_rng(0).standard_normal(200))  # rv <= 0
    with pytest.raises(ValueError):
        har_rv_fit(np.abs(np.random.default_rng(0).standard_normal(50)) + 0.1)
    rv = _rv_series()
    with pytest.raises(ValueError):
        harq_fit(rv, np.full(rv.size, np.nan))
    out = har_rv_fit(rv)
    with pytest.raises(ValueError):
        harq_forecast(out, rv, rv)  # HAR fit passed to HARQ forecaster
