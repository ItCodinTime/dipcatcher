"""LGBMQ2Distribution (dip_lgbm_q2, P1.8) — LightGBM quantile boosters on the
full causal feature matrix. SYNTHETIC data only: correctness checks, never
market evidence.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from quant_fund.config import load_config
from quant_fund.metrics.scoring import mean_pinball
from quant_fund.models.distribution import GaussianDistribution
from quant_fund.models.lgbm_q2 import LGBMQ2Distribution
from quant_fund.pipeline import train as train_module
from quant_fund.pipeline.train import train_distribution

TAUS = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]
MID = TAUS.index(0.5)


def _planted(n: int = 2400, k: int = 8, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """y = g(x) + sigma(x) * eps: nonlinear level + heteroskedastic scale."""
    rng = np.random.default_rng(seed)
    x = rng.normal(0.0, 1.0, (n, k))
    g = 0.04 * np.sin(2.0 * x[:, 0]) + 0.05 * np.abs(x[:, 1]) - 0.03 * x[:, 2] * x[:, 3]
    scale = 0.005 * (1.0 + np.abs(x[:, 4]) + x[:, 5] ** 2)
    y = g + scale * rng.standard_t(df=5, size=n)
    return x, y


def _assert_ordered(q: np.ndarray) -> None:
    assert q.ndim == 2
    assert np.all(np.isfinite(q))
    assert np.all(np.diff(q, axis=1) >= -1e-9)


def test_lgbm_q2_shapes_monotone_finite() -> None:
    x, y = _planted()
    m = LGBMQ2Distribution(TAUS).fit(x, y)
    rng = np.random.default_rng(1)
    q = m.predict(rng.normal(0.0, 1.0, (17, x.shape[1])))
    assert q.shape == (17, len(TAUS))
    _assert_ordered(q)


def test_lgbm_q2_beats_gaussian_pinball_on_planted() -> None:
    x, y = _planted(seed=3)
    cut = int(0.75 * x.shape[0])
    xtr, xte, ytr, yte = x[:cut], x[cut:], y[:cut], y[cut:]
    lgbm = LGBMQ2Distribution(TAUS, seed=7).fit(xtr, ytr).predict(xte)
    gauss = GaussianDistribution(TAUS).fit(xtr, ytr).predict(xte)
    for j, tau in enumerate(TAUS):
        pin_l = mean_pinball(yte, lgbm[:, j], tau)
        pin_g = mean_pinball(yte, gauss[:, j], tau)
        assert pin_l < pin_g


def test_lgbm_q2_uses_all_feature_columns() -> None:
    x, y = _planted(k=11)
    m = LGBMQ2Distribution(TAUS).fit(x, y)
    assert m.n_features_ == 11
    meta = m.metadata()
    assert meta.family == "distribution" and meta.name == "lgbm_q2"
    assert meta.extra["n_train"] == x.shape[0]
    assert meta.extra["n_features"] == 11
    assert meta.extra["n_estimators"] == m.n_estimators


def test_lgbm_q2_deterministic_under_seed() -> None:
    x, y = _planted(n=800)
    xq = x[:64]
    q1 = LGBMQ2Distribution(TAUS, seed=9).fit(x, y).predict(xq)
    q2 = LGBMQ2Distribution(TAUS, seed=9).fit(x, y).predict(xq)
    assert np.array_equal(q1, q2)


def test_lgbm_q2_fail_closed_edges() -> None:
    with pytest.raises(RuntimeError, match="not been fitted"):
        LGBMQ2Distribution(TAUS).predict(np.ones((3, 4)))
    with pytest.raises(ValueError, match=">= 50"):
        LGBMQ2Distribution(TAUS).fit(np.ones((49, 3)), np.linspace(0, 1, 49))
    rng = np.random.default_rng(0)
    x_nan = np.concatenate([rng.normal(size=(40, 3)), np.full((80, 3), np.nan)])
    y_ok = rng.normal(0.0, 0.01, 120)
    with pytest.raises(ValueError, match=">= 50"):
        LGBMQ2Distribution(TAUS).fit(x_nan, y_ok)
    with pytest.raises(ValueError, match=">= 1 feature"):
        LGBMQ2Distribution(TAUS).fit(np.empty((200, 0)), rng.normal(size=200))
    with pytest.raises(ValueError, match="nonzero variance"):
        LGBMQ2Distribution(TAUS).fit(rng.normal(size=(100, 3)), np.ones(100))
    with pytest.raises(ValueError, match="same number of rows"):
        LGBMQ2Distribution(TAUS).fit(rng.normal(size=(100, 3)), rng.normal(size=99))


def test_lgbm_q2_nan_rows_dropped_then_fits() -> None:
    rng = np.random.default_rng(5)
    x, y = _planted(n=900)
    x[:60] = np.nan
    m = LGBMQ2Distribution(TAUS).fit(x, y)
    assert m.n_train_ == 840
    xq = rng.normal(0.0, 1.0, (4, x.shape[1]))
    xq[0, 0] = np.nan
    _assert_ordered(m.predict(xq))


def test_train_distribution_accepts_lgbm_q2(monkeypatch: pytest.MonkeyPatch) -> None:
    """Catalog name passes _require_model; empty folds then fail closed."""
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
        train_distribution(cfg, "lgbm_q2")
    with pytest.raises(ValueError, match="unknown distribution model"):
        train_distribution(cfg, "not_a_model")
