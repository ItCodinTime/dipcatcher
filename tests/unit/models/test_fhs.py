"""FhsSkewDistribution (dip_fhs_skew) — fit contract, ordered quantiles,
planted vol-clustering/skew recovery, fail-closed edges, catalog wiring.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from quant_fund.config import load_config
from quant_fund.models.fhs import FhsSkewDistribution, _gjr_sigma_path
from quant_fund.pipeline import train as train_module
from quant_fund.pipeline.train import train_distribution

TAUS = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]
MID = TAUS.index(0.5)


def _assert_ordered(q: np.ndarray) -> None:
    assert q.ndim == 2
    assert np.all(np.isfinite(q))
    assert np.all(np.diff(q, axis=1) >= -1e-9)


def _x(n: int) -> np.ndarray:
    return np.ones((n, 2))


def test_fhs_skew_fit_predict_ordered_and_finite() -> None:
    rng = np.random.default_rng(0)
    y = rng.normal(0.0, 0.02, 500)
    m = FhsSkewDistribution(TAUS).fit(_x(y.size), y)
    q = m.predict(_x(5))
    assert q.shape == (5, len(TAUS))
    _assert_ordered(q)
    assert m.metadata().name == "fhs_skew"
    assert m.sigma_end_ > 0.0


def test_fhs_skew_wider_after_vol_burst() -> None:
    rng = np.random.default_rng(1)
    calm = rng.normal(0.0, 0.01, 400)
    burst = rng.normal(0.0, 0.08, 60)
    m_calm = FhsSkewDistribution(TAUS).fit(_x(calm.size), calm)
    y_burst = np.concatenate([calm, burst])
    m_burst = FhsSkewDistribution(TAUS).fit(_x(y_burst.size), y_burst)
    # one-step sigma at the window end must respond to the trailing burst
    assert m_burst.sigma_end_ > 2.0 * m_calm.sigma_end_
    q_calm = m_calm.predict(_x(1))[0]
    q_burst = m_burst.predict(_x(1))[0]
    assert (q_burst[-1] - q_burst[0]) > (q_calm[-1] - q_calm[0])


def test_fhs_skew_captures_left_skew() -> None:
    rng = np.random.default_rng(2)
    z = rng.standard_t(df=5, size=4000)
    y = np.where(z < 0, z * 2.2, z) * 0.01  # heavy left tail
    m = FhsSkewDistribution(TAUS).fit(_x(y.size), y)
    assert m.params_ is not None
    assert m.params_["lam"] < 0.0
    q = m.predict(_x(1))[0]
    assert (q[MID] - q[0]) > (q[-1] - q[MID])


def test_fhs_skew_deterministic() -> None:
    rng = np.random.default_rng(3)
    y = rng.standard_t(df=6, size=800) * 0.02
    q1 = FhsSkewDistribution(TAUS).fit(_x(y.size), y).predict(_x(3))
    q2 = FhsSkewDistribution(TAUS).fit(_x(y.size), y).predict(_x(3))
    np.testing.assert_array_equal(q1, q2)


def test_fhs_skew_fail_closed_edges() -> None:
    with pytest.raises(ValueError, match=">= 60"):
        FhsSkewDistribution(TAUS).fit(_x(30), np.linspace(0, 1, 30))
    with pytest.raises(ValueError, match="positive variance"):
        FhsSkewDistribution(TAUS).fit(_x(200), np.full(200, 1.0))
    with pytest.raises(RuntimeError, match="not been fitted"):
        FhsSkewDistribution(TAUS).predict(_x(2))


def test_fhs_skew_nan_rows_filtered() -> None:
    rng = np.random.default_rng(5)
    y = rng.normal(0.0, 0.02, 300)
    y[::7] = np.nan
    m = FhsSkewDistribution(TAUS).fit(_x(y.size), y)
    _assert_ordered(m.predict(_x(2)))


def test_gjr_sigma_path_degenerate_fails() -> None:
    e = np.tile(np.array([1e200, -1e200]), 50)
    with pytest.raises(ValueError, match="degenerate"):
        _gjr_sigma_path(e, 1.0)


def test_train_distribution_accepts_fhs_skew(monkeypatch: pytest.MonkeyPatch) -> None:
    """fhs_skew passes _require_model; empty folds then fail closed."""
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
        train_distribution(cfg, "fhs_skew")


def test_train_distribution_rejects_pooled_fhs_series(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = load_config("configs/research.yaml")
    frame = pl.DataFrame(
        {
            "event_time": [0, 0],
            "security_id": ["a", "b"],
            cfg.train.distribution_target: [0.01, 0.02],
            "ret_1": [0.0, 0.0],
        }
    )
    monkeypatch.setattr(train_module, "panel", lambda *a, **k: frame)
    with pytest.raises(ValueError, match="one security"):
        train_distribution(cfg, "fhs_skew")
