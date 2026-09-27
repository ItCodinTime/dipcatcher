"""Tests for quant_fund.models.enbpi — EnbPI (Xu & Xie 2021/2023). SYNTHETIC only."""

import numpy as np
import pytest
from sklearn.linear_model import Ridge

from quant_fund.models.enbpi import EnbPI, circular_block_bootstrap_indices


def _ar1_regression(n: int, seed: int, sigma: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = np.zeros(n)
    eps = rng.normal(0.0, sigma, n)
    for t in range(1, n):
        x[t] = 0.6 * x[t - 1] + eps[t]
    X = np.column_stack([np.roll(x, 1), np.roll(x, 2)])[2:]
    y = x[2:]
    return X, y


def _fit(n_train: int = 300, seed: int = 0, **kw: object) -> tuple[EnbPI, np.ndarray, np.ndarray]:
    X, y = _ar1_regression(n_train + 300, seed)
    model = EnbPI(lambda: Ridge(alpha=1e-3), n_estimators=25, alpha=0.1, seed=seed, **kw)  # type: ignore[arg-type]
    model.fit(X[:n_train], y[:n_train])
    return model, X[n_train:], y[n_train:]


def test_block_bootstrap_indices_shape_and_range() -> None:
    rng = np.random.default_rng(0)
    idx = circular_block_bootstrap_indices(50, 7, rng)
    assert idx.shape == (50,)
    assert idx.min() >= 0 and idx.max() < 50
    iid = circular_block_bootstrap_indices(50, 1, rng)
    assert iid.shape == (50,)
    with pytest.raises(ValueError):
        circular_block_bootstrap_indices(0, 1, rng)
    with pytest.raises(ValueError):
        circular_block_bootstrap_indices(5, 0, rng)


def test_block_bootstrap_preserves_contiguity() -> None:
    rng = np.random.default_rng(1)
    idx = circular_block_bootstrap_indices(40, 5, rng)
    for start in range(0, 40, 5):
        block = idx[start : start + 5]
        assert np.all(np.diff(block) % 40 == 1)


def test_online_coverage_near_nominal_on_stationary_ar1() -> None:
    covs = []
    for seed in range(4):
        model, Xt, yt = _fit(seed=seed)
        res = model.predict_online(Xt, yt)
        covs.append(res.coverage)
    assert 0.85 <= float(np.mean(covs)) <= 0.95
    assert res.mean_width > 0.0
    assert np.all(res.lower <= res.point) and np.all(res.point <= res.upper)


def test_residual_window_slides_and_tracks_variance_shift() -> None:
    model, Xt, yt = _fit(seed=3)
    _, _, _ = model.predict_interval(Xt[:1])
    w_before = float(np.quantile(model.residuals, 0.9))
    # Reveal labels with 4x noise; window should widen
    rng = np.random.default_rng(9)
    y_big = yt + rng.normal(0.0, 4.0, yt.size)
    model.update(Xt, y_big)
    w_after = float(np.quantile(model.residuals, 0.9))
    assert w_after > w_before
    assert model.residuals.size == 300  # window length == n_train


def test_median_aggregate_and_block_bootstrap_run() -> None:
    model, Xt, yt = _fit(seed=5, aggregate="median", block_size=10)
    res = model.predict_online(Xt, yt, batch_size=25)
    assert 0.8 <= res.coverage <= 0.98


def test_fail_closed_edges() -> None:
    with pytest.raises(ValueError):
        EnbPI(lambda: Ridge(), n_estimators=1)
    with pytest.raises(ValueError):
        EnbPI(lambda: Ridge(), alpha=1.0)
    with pytest.raises(ValueError):
        EnbPI(lambda: Ridge(), aggregate="max")
    m = EnbPI(lambda: Ridge(), n_estimators=5)
    with pytest.raises(RuntimeError):
        m.predict_point(np.zeros((2, 2)))
    with pytest.raises(ValueError):
        m.fit(np.zeros((5, 2)), np.zeros(5))
    X = np.random.default_rng(0).normal(size=(40, 2))
    with pytest.raises(ValueError):
        m.fit(X, np.r_[np.nan, np.zeros(39)])
    with pytest.raises(ValueError):
        m.fit(X, np.zeros(39))


def test_every_point_must_be_out_of_bag_somewhere() -> None:
    X = np.random.default_rng(0).normal(size=(12, 1))
    y = X[:, 0]
    m = EnbPI(lambda: Ridge(), n_estimators=2, block_size=12, alpha=0.2)
    # Two circular blocks of length 12 always cover all points -> no LOO set.
    with pytest.raises(ValueError):
        m.fit(X, y)
