"""Wave 13: stylized_stress + HistoricalTail / ScaledHistoricalTail edge fixtures."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.metrics.risk import historical_es, historical_var, losses_from_returns
from quant_fund.models.tail import (
    GaussianTail,
    HistoricalTail,
    ScaledHistoricalTail,
    stylized_stress,
)


def test_stylized_stress_defaults_and_custom() -> None:
    d = stylized_stress()
    assert d["equity_shock"] == -0.10
    assert d["vol_multiplier"] == 2.0
    assert d["correlation"] == 0.9
    assert d["note"] == "stylized_hypothetical"
    # Not a named crisis replay — key label is explicit
    assert "crisis" not in str(d["note"]).lower()

    custom = stylized_stress(equity_shock=-0.25, vol_mult=3.5, corr=0.99)
    assert custom["equity_shock"] == -0.25
    assert custom["vol_multiplier"] == 3.5
    assert custom["correlation"] == 0.99
    assert custom["note"] == "stylized_hypothetical"


def test_stylized_stress_boundary_values() -> None:
    zero = stylized_stress(equity_shock=0.0, vol_mult=1.0, corr=0.0)
    assert zero["equity_shock"] == 0.0
    assert zero["vol_multiplier"] == 1.0
    assert zero["correlation"] == 0.0
    assert set(zero) == {"equity_shock", "vol_multiplier", "correlation", "note"}


def test_historical_tail_unfitted_raises() -> None:
    m = HistoricalTail(0.95)
    with pytest.raises(RuntimeError, match="not been fitted"):
        m.predict_var_es()
    with pytest.raises(RuntimeError, match="not been fitted"):
        m.predict(np.zeros((3, 1)))


def test_historical_tail_closed_form_vs_helpers() -> None:
    # Known returns → losses = -R; VaR/ES must match risk helpers exactly
    y = np.array([0.01, -0.02, 0.03, -0.05, 0.0, -0.01, 0.02, -0.04, 0.015, -0.03])
    x = np.zeros((y.size, 1))
    alpha = 0.90
    m = HistoricalTail(alpha).fit(x, y)
    var, es = m.predict_var_es()
    losses = losses_from_returns(y)
    assert var == pytest.approx(historical_var(losses, alpha))
    assert es == pytest.approx(historical_es(losses, alpha))
    assert es >= var - 1e-12
    pred = m.predict(np.zeros((4, 1)))
    assert pred.shape == (4, 2)
    assert np.allclose(pred[:, 0], var) and np.allclose(pred[:, 1], es)


def test_historical_tail_empty_and_nan_returns() -> None:
    # All-nonfinite y → empty finite losses → nan VaR/ES (helpers)
    y = np.array([np.nan, np.inf, -np.inf])
    m = HistoricalTail(0.95).fit(np.zeros((3, 1)), y)
    var, es = m.predict_var_es()
    assert np.isnan(var) and np.isnan(es)


def test_historical_tail_invalid_alpha() -> None:
    y = np.array([0.01, -0.02, 0.03, -0.01])
    for bad in (0.0, 1.0, -0.1, 1.5):
        m = HistoricalTail(bad).fit(np.zeros((4, 1)), y)
        with pytest.raises(ValueError, match="alpha"):
            m.predict_var_es()


def test_gaussian_tail_unfitted_and_metadata() -> None:
    g = GaussianTail(0.95)
    with pytest.raises(RuntimeError, match="not been fitted"):
        g.predict_var_es()
    y = np.random.default_rng(0).normal(size=80)
    g.fit(np.zeros((80, 1)), y)
    var, es = g.predict_var_es()
    assert np.isfinite(var) and np.isfinite(es)
    assert es >= var - 1e-9
    assert g.metadata().name == "gaussian"
    assert HistoricalTail(0.95).metadata().name == "historical"


def test_scaled_historical_tail_empty_and_length_mismatch() -> None:
    m = ScaledHistoricalTail(0.95)
    # Empty / all-nan → z_var/z_es = 0
    m.fit(np.array([np.nan, np.nan]), np.array([1.0, 1.0]))
    assert m.z_var == 0.0 and m.z_es == 0.0
    var, es = m.predict_var_es(np.array([2.0, 3.0]))
    assert np.allclose(var, 0.0) and np.allclose(es, 0.0)

    with pytest.raises(ValueError, match="same length"):
        ScaledHistoricalTail(0.95).fit(np.array([0.1, -0.1]), np.array([1.0]))


def test_scaled_historical_tail_scales_linearly() -> None:
    rng = np.random.default_rng(7)
    y = rng.normal(0.0, 1.0, size=200)
    sc_train = np.ones(200)
    m = ScaledHistoricalTail(0.95).fit(y, sc_train)
    sc = np.array([0.5, 1.0, 2.0])
    var, es = m.predict_var_es(sc)
    assert var[1] == pytest.approx(m.z_var)
    assert es[1] == pytest.approx(m.z_es)
    assert var[2] == pytest.approx(2.0 * var[1])
    assert es[0] == pytest.approx(0.5 * es[1])
    assert np.all(es >= var - 1e-12)
