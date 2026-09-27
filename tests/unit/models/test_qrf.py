"""Tests for quant_fund.models.qrf — quantile regression forests (Meinshausen 2006). SYNTHETIC."""

import numpy as np
import pytest
from scipy.stats import kstest

from quant_fund.metrics.scoring import mean_pinball
from quant_fund.models.qrf import QuantileRegressionForest, weighted_quantiles

TAUS = np.array([0.1, 0.5, 0.9])


def _hetero(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """y = sin(2 pi x1) + (0.2 + x2) * eps, eps ~ N(0,1); scale grows with x2."""
    rng = np.random.default_rng(seed)
    X = rng.uniform(0.0, 1.0, size=(n, 2))
    y = np.sin(2 * np.pi * X[:, 0]) + (0.2 + X[:, 1]) * rng.normal(size=n)
    return X, y


def test_weighted_quantiles_matches_empirical_on_uniform_weights() -> None:
    v = np.arange(1.0, 11.0)
    w = np.full((1, 10), 0.1)
    q = weighted_quantiles(v, w, np.array([0.1, 0.5, 1.0 - 1e-9]))
    assert q.shape == (1, 3)
    assert q[0, 0] == 1.0 and q[0, 1] == 5.0 and q[0, 2] == 10.0


def test_weighted_quantiles_point_mass() -> None:
    v = np.array([3.0, 1.0, 2.0])
    w = np.array([[0.0, 1.0, 0.0]])
    assert np.all(weighted_quantiles(v, w, TAUS) == 1.0)


def test_weighted_quantiles_fail_closed() -> None:
    v = np.arange(3.0)
    with pytest.raises(ValueError):
        weighted_quantiles(v, np.ones((1, 3)), np.array([0.0]))
    with pytest.raises(ValueError):
        weighted_quantiles(v, np.ones((1, 2)), TAUS)
    with pytest.raises(ValueError):
        weighted_quantiles(v, np.zeros((1, 3)), TAUS)
    with pytest.raises(ValueError):
        weighted_quantiles(v, -np.ones((1, 3)), TAUS)


def test_weights_sum_to_one_and_mean_matches_weights() -> None:
    X, y = _hetero(400, 0)
    qrf = QuantileRegressionForest(n_estimators=50, min_samples_leaf=10, seed=0).fit(X, y)
    Xt, _ = _hetero(20, 1)
    w = qrf.weights(Xt)
    assert w.shape == (20, 400)
    np.testing.assert_allclose(w.sum(axis=1), 1.0, atol=1e-12)
    np.testing.assert_allclose(qrf.predict_mean(Xt), w @ y)


def test_quantiles_monotone_and_track_heteroskedastic_scale() -> None:
    X, y = _hetero(1500, 2)
    qrf = QuantileRegressionForest(n_estimators=100, min_samples_leaf=10, seed=1).fit(X, y)
    Xt = np.column_stack([np.full(200, 0.25), np.linspace(0.0, 1.0, 200)])
    q = qrf.predict_quantiles(Xt, TAUS)
    assert np.all(np.diff(q, axis=1) >= 0.0)
    width = q[:, 2] - q[:, 0]
    assert np.mean(width[-50:]) > 1.5 * np.mean(width[:50])


def test_oos_coverage_and_pinball_beat_unconditional() -> None:
    X, y = _hetero(1500, 3)
    Xt, yt = _hetero(600, 4)
    qrf = QuantileRegressionForest(n_estimators=100, min_samples_leaf=10, seed=2).fit(X, y)
    q = qrf.predict_quantiles(Xt, TAUS)
    cov = np.mean((yt >= q[:, 0]) & (yt <= q[:, 2]))
    assert 0.72 <= cov <= 0.88
    for j, tau in enumerate(TAUS):
        uncond = np.full(yt.size, np.quantile(y, tau))
        assert mean_pinball(yt, q[:, j], float(tau)) < mean_pinball(yt, uncond, float(tau))


def test_oob_pit_is_approximately_uniform() -> None:
    X, y = _hetero(1200, 5)
    qrf = QuantileRegressionForest(
        n_estimators=200, min_samples_leaf=10, leaf_mode="oob", seed=3
    ).fit(X, y)
    pit = qrf.pit(X, y)
    assert np.all((pit >= 0.0) & (pit <= 1.0))
    assert kstest(pit, "uniform").pvalue > 0.01


def test_in_sample_all_mode_is_overconfident_relative_to_oob() -> None:
    X, y = _hetero(600, 6)
    q_all = QuantileRegressionForest(n_estimators=60, min_samples_leaf=5, seed=4).fit(X, y)
    q_oob = QuantileRegressionForest(
        n_estimators=60, min_samples_leaf=5, leaf_mode="oob", seed=4
    ).fit(X, y)
    w_all = q_all.predict_quantiles(X, TAUS)
    w_oob = q_oob.predict_quantiles(X, TAUS)
    assert np.mean(w_all[:, 2] - w_all[:, 0]) < np.mean(w_oob[:, 2] - w_oob[:, 0])


def test_predict_cdf_monotone_bounded() -> None:
    X, y = _hetero(300, 7)
    qrf = QuantileRegressionForest(n_estimators=30, seed=0).fit(X, y)
    grid = np.linspace(-3, 3, 25)
    cdf = qrf.predict_cdf(X[:5], grid)
    assert cdf.shape == (5, 25)
    assert np.all(np.diff(cdf, axis=1) >= -1e-12)
    assert np.all(cdf >= 0.0) and np.all(cdf <= 1.0 + 1e-12)


def test_fail_closed_edges() -> None:
    with pytest.raises(ValueError):
        QuantileRegressionForest(n_estimators=0)
    with pytest.raises(ValueError):
        QuantileRegressionForest(leaf_mode="bag")
    qrf = QuantileRegressionForest(n_estimators=5)
    with pytest.raises(RuntimeError):
        qrf.predict_quantiles(np.zeros((2, 2)), TAUS)
    X, y = _hetero(100, 0)
    with pytest.raises(ValueError):
        qrf.fit(X, np.r_[np.inf, y[1:]])
    with pytest.raises(ValueError):
        qrf.fit(X[:5], y[:5])
    qrf.fit(X, y)
    with pytest.raises(ValueError):
        qrf.predict_quantiles(np.zeros((2, 3)), TAUS)
    with pytest.raises(ValueError):
        qrf.predict_quantiles(X[:2], np.array([1.5]))
    with pytest.raises(ValueError):
        qrf.predict_cdf(X[:2], np.array([]))
