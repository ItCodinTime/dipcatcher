"""Tests for quant_fund.models.ngboost_lite — NGBoostGaussian (Duan et al. 2020). SYNTHETIC."""

import numpy as np
import pytest
from scipy.stats import kstest, norm

from quant_fund.metrics.scoring import crps_gaussian
from quant_fund.models.ngboost_lite import NGBoostGaussian, _gradients


def _hetero(n: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1.0, 1.0, size=(n, 2))
    mu = 2.0 * X[:, 0] + X[:, 1] ** 2
    sigma = 0.3 + 1.2 * (X[:, 1] + 1.0) / 2.0
    y = mu + sigma * rng.normal(size=n)
    return X, y, mu, sigma


def test_logscore_natural_gradient_closed_form() -> None:
    y = np.array([1.0, -2.0, 0.5])
    mu = np.zeros(3)
    ls = np.log(np.array([1.0, 2.0, 0.5]))
    g_mu, g_ls = _gradients(y, mu, ls, "logscore")
    np.testing.assert_allclose(g_mu, -(y - mu))
    z = (y - mu) / np.exp(ls)
    np.testing.assert_allclose(g_ls, 0.5 * (1.0 - z**2))


def test_crps_gradient_matches_finite_difference() -> None:
    y = np.array([0.7, -1.3])
    mu = np.array([0.1, 0.2])
    ls = np.array([0.0, 0.4])
    g_mu, g_ls = _gradients(y, mu, ls, "crps")
    sigma = np.exp(ls)
    metric_mu = 1.0 / (np.sqrt(np.pi) * sigma)
    metric_ls = sigma / (2.0 * np.sqrt(np.pi))
    h = 1e-6
    fd_mu = (crps_gaussian(y, mu + h, sigma) - crps_gaussian(y, mu - h, sigma)) / (2 * h)
    fd_ls = (crps_gaussian(y, mu, np.exp(ls + h)) - crps_gaussian(y, mu, np.exp(ls - h))) / (2 * h)
    np.testing.assert_allclose(g_mu * metric_mu, fd_mu, rtol=1e-5, atol=1e-7)
    np.testing.assert_allclose(g_ls * metric_ls, fd_ls, rtol=1e-5, atol=1e-7)


@pytest.mark.parametrize("score", ["logscore", "crps"])
def test_training_score_monotone_and_sigma_tracks_heteroskedasticity(score: str) -> None:
    X, y, mu_true, sigma_true = _hetero(1500, 0)
    model = NGBoostGaussian(n_estimators=120, learning_rate=0.1, score=score, seed=0).fit(X, y)
    info = model.fit_info
    assert info is not None and info.n_rounds == 120
    hist = np.asarray(info.train_score)
    assert hist[-1] < hist[0]
    assert np.all(np.diff(hist) <= 1e-9)
    Xt, yt, mu_t, sigma_t = _hetero(800, 1)
    mu_hat, sigma_hat = model.predict_params(Xt)
    assert np.corrcoef(mu_hat, mu_t)[0, 1] > 0.95
    assert np.corrcoef(sigma_hat, sigma_t)[0, 1] > 0.6
    # Beats an unconditional Gaussian on CRPS out of sample
    uncond = float(
        np.mean(crps_gaussian(yt, np.full(yt.size, y.mean()), np.full(yt.size, y.std())))
    )
    assert model.crps(Xt, yt) < 0.7 * uncond


def test_pit_uniform_out_of_sample_with_early_stopping() -> None:
    """Early-stopped fit is calibrated OOS; the unstopped fit is overconfident (sigma shrinks)."""
    X, y, _, _ = _hetero(2000, 2)
    Xv, yv, _, _ = _hetero(500, 9)
    Xt, yt, _, _ = _hetero(1000, 3)
    model = NGBoostGaussian(
        n_estimators=500, learning_rate=0.05, early_stopping_rounds=20, seed=1
    ).fit(X, y, Xv, yv)
    pit = model.pit(Xt, yt)
    assert kstest(pit, "uniform").pvalue > 0.01
    mu, sigma = model.predict_params(Xt)
    assert 0.9 <= np.std((yt - mu) / sigma) <= 1.15
    over = NGBoostGaussian(n_estimators=150, learning_rate=0.1, seed=1).fit(X, y)
    mu_o, sigma_o = over.predict_params(Xt)
    assert np.std((yt - mu_o) / sigma_o) > 1.2
    assert model.crps(Xt, yt) < over.crps(Xt, yt)
    q = model.predict_quantiles(Xt, np.array([0.05, 0.5, 0.95]))
    cov = np.mean((yt >= q[:, 0]) & (yt <= q[:, 2]))
    assert 0.86 <= cov <= 0.94
    assert np.all(np.diff(q, axis=1) > 0.0)


def test_early_stopping_truncates_learners() -> None:
    X, y, _, _ = _hetero(400, 4)
    Xv, yv, _, _ = _hetero(200, 5)
    model = NGBoostGaussian(
        n_estimators=400,
        learning_rate=0.3,
        max_depth=6,
        min_samples_leaf=1,
        early_stopping_rounds=10,
        seed=2,
    ).fit(X, y, Xv, yv)
    info = model.fit_info
    assert info is not None
    assert info.n_rounds < 400
    assert info.n_rounds == info.best_round
    assert len(info.val_score) >= info.best_round
    assert min(info.val_score) == pytest.approx(info.val_score[info.best_round - 1])


def test_quantiles_consistent_with_params() -> None:
    X, y, _, _ = _hetero(300, 6)
    model = NGBoostGaussian(n_estimators=20, seed=0).fit(X, y)
    mu, sigma = model.predict_params(X[:5])
    q = model.predict_quantiles(X[:5], np.array([0.25]))
    np.testing.assert_allclose(q[:, 0], mu + sigma * norm.ppf(0.25))
    assert model.predict(X[:5]).shape == (5,)


def test_fail_closed_edges() -> None:
    for kw in (
        {"n_estimators": 0},
        {"learning_rate": 0.0},
        {"score": "mse"},
        {"early_stopping_rounds": 0},
    ):
        with pytest.raises(ValueError):
            NGBoostGaussian(**kw)  # type: ignore[arg-type]
    m = NGBoostGaussian(n_estimators=5)
    with pytest.raises(RuntimeError):
        m.predict(np.zeros((2, 2)))
    X, y, _, _ = _hetero(100, 0)
    with pytest.raises(ValueError):
        m.fit(X, np.r_[np.nan, y[1:]])
    with pytest.raises(ValueError):
        m.fit(X, np.zeros(100))
    with pytest.raises(ValueError):
        NGBoostGaussian(n_estimators=5, early_stopping_rounds=2).fit(X, y)
    with pytest.raises(ValueError):
        m.fit(X, y, X_val=X)
    m.fit(X, y)
    with pytest.raises(ValueError):
        m.predict(np.zeros((2, 3)))
    with pytest.raises(ValueError):
        m.predict_quantiles(X[:2], np.array([0.0, 0.5]))
