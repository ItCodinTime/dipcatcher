"""Wave 44: risk VaR/ES empty / bad-alpha / all-NaN edges — honest NaN or ValueError.

Research / risk diagnostics only — not a live P&L claim.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from quant_fund.metrics.risk import (
    gaussian_es,
    gaussian_var,
    historical_es,
    historical_var,
    losses_from_returns,
    var_es_from_return_quantiles,
)


def test_losses_from_returns_empty_and_nan() -> None:
    empty = losses_from_returns(np.array([]))
    assert empty.size == 0
    out = losses_from_returns(np.array([0.1, -0.2, np.nan]))
    assert out[0] == pytest.approx(-0.1)
    assert out[1] == pytest.approx(0.2)
    assert math.isnan(out[2])


@pytest.mark.parametrize(
    "fn",
    [historical_var, historical_es, gaussian_var, gaussian_es],
)
def test_risk_bad_alpha_fail_closed(fn) -> None:
    x = np.array([0.01, 0.02, 0.03, 0.04])
    for alpha in (0.0, 1.0, -0.1, 1.5, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="alpha"):
            fn(x, alpha)


@pytest.mark.parametrize("fn", [historical_var, historical_es])
def test_historical_empty_all_nan(fn) -> None:
    assert math.isnan(fn(np.array([])))
    assert math.isnan(fn(np.array([np.nan, np.nan, np.inf, -np.inf])))


@pytest.mark.parametrize("fn", [gaussian_var, gaussian_es])
def test_gaussian_short_and_all_nan(fn) -> None:
    assert math.isnan(fn(np.array([])))
    assert math.isnan(fn(np.array([0.01])))
    assert math.isnan(fn(np.array([np.nan, np.inf])))
    # n=2 finite → finite diagnostic
    v = fn(np.array([0.01, 0.03]))
    assert math.isfinite(v)


def test_historical_var_es_finite_path() -> None:
    losses = np.array([0.01, 0.02, 0.05, 0.03, 0.04])
    var = historical_var(losses, 0.8)
    es = historical_es(losses, 0.8)
    assert math.isfinite(var) and math.isfinite(es)
    assert es >= var - 1e-12


def test_var_es_from_return_quantiles_edges() -> None:
    with pytest.raises(ValueError, match="alpha"):
        var_es_from_return_quantiles({0.05: -1.0}, alpha=0.0)
    with pytest.raises(ValueError, match="alpha"):
        var_es_from_return_quantiles({0.05: -1.0}, alpha=float("nan"))

    v, e = var_es_from_return_quantiles({}, 0.95)
    assert math.isnan(v) and math.isnan(e)

    v2, e2 = var_es_from_return_quantiles({0.05: float("nan"), 0.1: float("nan")}, 0.95)
    assert math.isnan(v2) and math.isnan(e2)

    # Non-monotone duplicate taus collapse after sort → NaN via non-positive diff
    # (dict can't hold duplicate keys; use single tau only → still finite interp path)
    v3, e3 = var_es_from_return_quantiles({0.05: -2.0}, 0.95)
    assert math.isfinite(v3) and math.isfinite(e3)
