"""QARDistribution — QAR(1) challenger head contract on SYNTHETIC data.

Shapes/monotone/finite output, AR(1) coefficient recovery on planted
structure, fail-closed edges, and train.py catalog acceptance via the
early-raise pattern from test_distribution_challengers.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import norm

from quant_fund.config import load_config
from quant_fund.metrics.scoring import mean_pinball
from quant_fund.models.qar import QARDistribution
from quant_fund.pipeline.train import train_distribution

TAUS = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]


def _x(n: int) -> np.ndarray:
    return np.ones((n, 2))


def _ar1(phi: float = 0.9, sigma: float = 0.01, n: int = 3000, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    y = np.empty(n)
    y[0] = 0.0
    for t in range(1, n):
        y[t] = phi * y[t - 1] + sigma * rng.standard_normal()
    return y


def _assert_ordered(q: np.ndarray) -> None:
    assert q.ndim == 2
    assert np.all(np.isfinite(q))
    assert np.all(np.diff(q, axis=1) >= -1e-9)


def test_qar_fit_predict_shapes_ordered_finite() -> None:
    y = _ar1()
    m = QARDistribution(TAUS).fit(_x(y.size), y)
    q = m.predict(_x(1))
    assert q.shape == (1, len(TAUS))
    _assert_ordered(q)
    assert m.metadata().family == "distribution"
    assert m.metadata().name == "qar"


def test_qar_recovers_ar1_persistence() -> None:
    sigma = 0.01
    y = _ar1(phi=0.9, sigma=sigma)
    m = QARDistribution(TAUS).fit(_x(y.size), y)
    assert m.coef_ is not None
    # pure AR(1): a1(tau) ~ phi at every quantile, a0(tau) ~ sigma * Phi^-1(tau)
    assert np.allclose(m.coef_[:, 1], 0.9, atol=0.05)
    assert np.allclose(m.coef_[:, 0], sigma * norm.ppf(TAUS), atol=0.004)
    assert m.n_pairs_ == y.size - 1


def test_qar_predict_is_conditional_at_last_y() -> None:
    y = _ar1(phi=0.85)
    m = QARDistribution(TAUS).fit(_x(y.size), y)
    assert m.coef_ is not None
    raw = m.coef_[:, 0] + m.coef_[:, 1] * float(y[-1])
    q = m.predict(_x(1))
    assert np.allclose(np.sort(raw), q[0])
    with pytest.raises(ValueError, match="exactly one"):
        m.predict(_x(4))


def test_qar_coefficients_beat_unconditional_on_independent_planted_ar1() -> None:
    y = _ar1(phi=0.9, sigma=0.01)
    y_future = _ar1(phi=0.9, sigma=0.01, seed=5)
    m = QARDistribution(TAUS).fit(_x(y.size), y)
    assert m.coef_ is not None
    y_lag, y_cur = y_future[:-1], y_future[1:]
    emp = np.quantile(y, TAUS)
    for j, tau in enumerate(TAUS):
        q_cond = m.coef_[j, 0] + m.coef_[j, 1] * y_lag
        pin_qar = mean_pinball(y_cur, q_cond, tau)
        pin_emp = mean_pinball(y_cur, np.full(y_cur.size, emp[j]), tau)
        assert pin_qar < pin_emp


def test_qar_rejects_nonfinite_gap() -> None:
    y = _ar1()
    y[::97] = np.nan
    y[1] = np.inf
    with pytest.raises(ValueError, match="all finite"):
        QARDistribution(TAUS).fit(_x(y.size), y)


def test_qar_fail_closed_edges() -> None:
    with pytest.raises(RuntimeError, match="not been fitted"):
        QARDistribution(TAUS).predict(_x(2))
    with pytest.raises(ValueError, match=">= 30"):
        QARDistribution(TAUS).fit(_x(20), np.linspace(0, 1, 20))
    with pytest.raises(ValueError, match="non-constant"):
        QARDistribution(TAUS).fit(_x(100), np.full(100, 0.5))
    with pytest.raises(ValueError, match="0, 1"):
        QARDistribution([0.0, 0.5]).fit(_x(100), _ar1(n=100))


def test_train_distribution_rejects_qar_without_series_wiring() -> None:
    """Generic panel folds cannot supply a one-step single-series origin."""
    cfg = load_config("configs/research.yaml")
    with pytest.raises(ValueError, match="unknown distribution model"):
        train_distribution(cfg, "qar")
