"""ConformalTDistribution (conf_t) — split-conformal shift on a skew-t base.

SYNTHETIC correctness tests only: shapes, monotone/finite output, recovery
of a planted trailing-window level shift, calibration grid behavior,
independent future scoring in a planted-shift scenario, and fail-closed edges.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from quant_fund.config import load_config
from quant_fund.models.conformal_dist import ConformalTDistribution
from quant_fund.pipeline import train as train_module
from quant_fund.pipeline.train import train_distribution

TAUS = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]


def _x(n: int) -> np.ndarray:
    return np.ones((n, 2))


def _y(seed: int = 0, n: int = 1200) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_t(df=5, size=n) * 0.01 + 0.001


def _assert_ordered(q: np.ndarray) -> None:
    assert q.ndim == 2
    assert np.all(np.isfinite(q))
    assert np.all(np.diff(q, axis=1) >= -1e-9)


def test_conf_t_fit_predict_ordered_and_finite() -> None:
    y = _y()
    m = ConformalTDistribution(TAUS).fit(_x(y.size), y)
    q = m.predict(_x(7))
    assert q.shape == (7, len(TAUS))
    _assert_ordered(q)
    assert m.n_cal_ == y.size - (2 * y.size) // 3
    assert m.metadata().name == "conf_t"
    assert m.metadata().family == "distribution"


def test_conf_t_recovers_planted_trailing_shift() -> None:
    rng = np.random.default_rng(3)
    n = 900
    delta = 0.05
    y = rng.normal(0.0, 0.01, n)
    y[(2 * n) // 3 :] += delta  # level shift confined to the calibration slice
    m = ConformalTDistribution(TAUS).fit(_x(n), y)
    assert m.shifts_ is not None and m.q_ is not None
    y_cal = y[(2 * n) // 3 :]
    # shift ≈ planted delta; corrected quantiles ≈ calibration-slice quantiles
    assert np.allclose(m.shifts_, delta, atol=0.01)
    assert np.allclose(m.q_, np.quantile(y_cal, TAUS), atol=0.01)


def test_conf_t_calibration_coverage_on_conformal_grid() -> None:
    y = _y(seed=7)
    m = ConformalTDistribution(TAUS).fit(_x(y.size), y)
    assert m.q_ is not None
    y_cal = y[(2 * y.size) // 3 :]
    n = y_cal.size
    for j, tau in enumerate(TAUS):
        k = min(max(int(np.ceil((n + 1) * tau)), 1), n)
        cov = float(np.mean(y_cal <= m.q_[j]))
        assert cov >= tau
        assert cov == pytest.approx(k / n, abs=2.0 / n)  # grid + rearrange slack


def test_conf_t_beats_base_on_independent_shifted_future() -> None:
    from quant_fund.metrics.scoring import mean_pinball
    from quant_fund.models.skew_t import skew_t_fit, skew_t_ppf

    rng = np.random.default_rng(9)
    n = 900
    y = rng.normal(0.0, 0.01, n)
    y[(2 * n) // 3 :] += 0.04
    y_future = rng.normal(0.04, 0.01, 300)
    cut = (2 * n) // 3
    m = ConformalTDistribution(TAUS).fit(_x(n), y)
    assert m.q_ is not None
    p = skew_t_fit(y[:cut])
    q_base = np.array([skew_t_ppf(t, p["nu"], p["lam"], p["mu"], p["sigma"]) for t in TAUS])
    for j, tau in enumerate(TAUS):
        pin_c = mean_pinball(y_future, np.full(y_future.size, m.q_[j]), tau)
        pin_b = mean_pinball(y_future, np.full(y_future.size, q_base[j]), tau)
        assert pin_c <= pin_b


def test_conf_t_deterministic() -> None:
    y = _y(seed=11)
    a = ConformalTDistribution(TAUS).fit(_x(y.size), y)
    b = ConformalTDistribution(TAUS).fit(_x(y.size), y)
    assert a.q_ is not None and b.q_ is not None
    assert np.array_equal(a.q_, b.q_)


def test_conf_t_predict_before_fit_raises() -> None:
    with pytest.raises(RuntimeError, match="not been fitted"):
        ConformalTDistribution(TAUS).predict(_x(3))


def test_conf_t_too_short_fail_closed() -> None:
    with pytest.raises(ValueError, match=">= 60"):
        ConformalTDistribution(TAUS).fit(_x(59), np.linspace(0, 1, 59))
    with pytest.raises(ValueError, match=">= 60"):
        ConformalTDistribution(TAUS).fit(_x(80), np.full(80, np.nan))


def test_conf_t_degenerate_fail_closed() -> None:
    with pytest.raises(ValueError, match="zero variance"):
        ConformalTDistribution(TAUS).fit(_x(200), np.full(200, 0.5))
    with pytest.raises(ValueError, match="0, 1"):
        ConformalTDistribution([0.5, 1.5]).fit(_x(200), _y(n=200))


def test_train_distribution_accepts_conf_t(monkeypatch: pytest.MonkeyPatch) -> None:
    """conf_t passes _require_model; empty folds then fail closed."""
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
        train_distribution(cfg, "conf_t")
    with pytest.raises(ValueError, match="unknown distribution model"):
        train_distribution(cfg, "not_a_model")
