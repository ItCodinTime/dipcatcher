"""RegimeDistribution (dip_regime): 2-state vol-regime mixture head —
fit contract, ordered quantiles, planted-regime recovery, fail-closed edges.
All data here is SYNTHETIC correctness material, not market evidence.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

import quant_fund.models.regime_dist as regime_dist_module
from quant_fund.config import load_config
from quant_fund.models.regime_dist import RegimeDistribution
from quant_fund.pipeline import train as train_module
from quant_fund.pipeline.train import train_distribution

TAUS = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]


def _x(n: int) -> np.ndarray:
    return np.ones((n, 2))


def _regime_switch(
    seed: int = 0, n_lo: int = 800, n_hi: int = 400, sig_lo: float = 0.005, sig_hi: float = 0.05
) -> np.ndarray:
    """SYNTHETIC two-regime series: low-vol block then high-vol block."""
    rng = np.random.default_rng(seed)
    return np.concatenate([rng.normal(0.0, sig_lo, n_lo), rng.normal(0.0, sig_hi, n_hi)])


def _assert_ordered(q: np.ndarray) -> None:
    assert q.ndim == 2
    assert np.all(np.isfinite(q))
    assert np.all(np.diff(q, axis=1) >= -1e-9)


def test_regime_fit_predict_ordered_and_finite() -> None:
    y = _regime_switch()
    m = RegimeDistribution(TAUS).fit(_x(y.size), y)
    q = m.predict(_x(5))
    assert q.shape == (5, len(TAUS))
    _assert_ordered(q)
    meta = m.metadata()
    assert meta.family == "distribution"
    assert meta.name == "regime"
    assert meta.extra["n_states_effective"] == 2
    assert meta.extra["state_estimator"] == "hmm"


def test_regime_recovers_planted_vol_states() -> None:
    y = _regime_switch()
    m = RegimeDistribution(TAUS).fit(_x(y.size), y)
    assert m.state_quantiles_ is not None and m.mix_weights_ is not None
    # high-vol state must have wider empirical quantile spread
    lo_spread = m.state_quantiles_[0, -1] - m.state_quantiles_[0, 0]
    hi_spread = m.state_quantiles_[1, -1] - m.state_quantiles_[1, 0]
    assert hi_spread > 3.0 * lo_spread
    # series ends in the high-vol block -> last-obs posterior puts most
    # weight there, so the mixed quantiles sit near the high-vol state CDF
    assert m.mix_weights_[1] > 0.5
    hi_q = np.quantile(y[-400:], TAUS)
    assert np.allclose(m.q_, hi_q, atol=0.02)


def test_regime_mixture_cdf_inversion_not_quantile_average(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Mixed quantile = generalized inverse of the weighted mixture CDF,
    not a weighted average of per-state quantiles (quantiles don't mix)."""
    rng = np.random.default_rng(4)
    y = np.concatenate([rng.normal(0.0, 0.01, 600), rng.normal(0.0, 0.05, 600)])
    n = y.size
    probs = np.empty((n, 2))
    probs[:600] = [0.9, 0.1]
    probs[600:] = [0.1, 0.9]
    probs[-1] = [0.5, 0.5]  # ambiguous last obs -> genuinely mixed weights
    monkeypatch.setattr(RegimeDistribution, "_state_probs", lambda self, yy: probs)
    m = RegimeDistribution(TAUS).fit(_x(n), y)
    assert m.mix_weights_ is not None and m.state_quantiles_ is not None
    np.testing.assert_allclose(m.mix_weights_, [0.5, 0.5], atol=1e-12)
    assign = np.argmax(probs, axis=1)
    grid = np.unique(y)
    cdf = np.zeros(grid.size)
    for s in range(2):
        ys = np.sort(y[assign == s])
        cdf += 0.5 * np.searchsorted(ys, grid, side="right") / ys.size
    expected = grid[np.clip(np.searchsorted(cdf, TAUS, side="left"), 0, grid.size - 1)]
    np.testing.assert_allclose(m.q_, expected, atol=1e-12)
    avg = 0.5 * (m.state_quantiles_[0] + m.state_quantiles_[1])
    assert not np.allclose(m.q_, avg, atol=1e-9)


def test_regime_deterministic_under_seed() -> None:
    y = _regime_switch(seed=7)
    q1 = RegimeDistribution(TAUS, seed=3).fit(_x(y.size), y).predict(_x(2))
    q2 = RegimeDistribution(TAUS, seed=3).fit(_x(y.size), y).predict(_x(2))
    np.testing.assert_array_equal(q1, q2)


def test_regime_constant_series_single_state() -> None:
    y = np.full(200, 0.01)
    m = RegimeDistribution(TAUS).fit(_x(y.size), y)
    assert m.n_states_effective_ == 1
    assert np.allclose(m.predict(_x(3))[0], 0.01)


def test_regime_hmm_failure_falls_back_to_threshold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _BrokenHMM:
        def __init__(self, *a: object, **k: object) -> None:
            pass

        def fit(self, *a: object, **k: object) -> object:
            raise RuntimeError("synthetic HMM failure")

    monkeypatch.setattr(regime_dist_module, "GaussianHMMRegime", _BrokenHMM)
    y = _regime_switch()
    m = RegimeDistribution(TAUS).fit(_x(y.size), y)
    assert m.state_estimator_ == "vol_threshold"
    _assert_ordered(m.predict(_x(2)))


def test_regime_too_short_fail_closed() -> None:
    with pytest.raises(ValueError, match=">= 60"):
        RegimeDistribution(TAUS).fit(_x(30), np.linspace(0, 1, 30))


def test_regime_mostly_nonfinite_fail_closed() -> None:
    y = np.full(100, np.nan)
    y[:50] = 0.01
    with pytest.raises(ValueError, match=">= 60"):
        RegimeDistribution(TAUS).fit(_x(y.size), y)


def test_regime_predict_before_fit_raises() -> None:
    with pytest.raises(RuntimeError, match="not been fitted"):
        RegimeDistribution(TAUS).predict(_x(2))


def test_train_distribution_accepts_regime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """'regime' passes _require_model; empty folds then fail closed."""
    cfg = load_config("configs/research.yaml")
    frame = pl.DataFrame(
        {
            "event_time": [0, 1],
            "security_id": ["a", "a"],
            cfg.train.distribution_target: [0.01, 0.02],
            "ret_1": [0.0, 0.0],
        }
    )
    monkeypatch.setattr(train_module, "panel", lambda *a, **k: frame)
    monkeypatch.setattr(train_module, "_walk_forward_splits", lambda *a, **k: [])
    with pytest.raises(ValueError, match="no trainable/evaluable fold"):
        train_distribution(cfg, "regime")
    with pytest.raises(ValueError, match="unknown distribution model"):
        train_distribution(cfg, "not_a_model")
