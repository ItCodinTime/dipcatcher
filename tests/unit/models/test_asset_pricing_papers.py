"""Kelly–Malamud–Zhou RFF, Kozak–Nagel–Santosh SDF ridge, Kelly–Pruitt–Su IPCA."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.config import load_config
from quant_fund.config.models import TrainConfig
from quant_fund.models.asset_pricing import (
    IPCARanker,
    RandomFourierRanker,
    SDFRidgeRanker,
    _voc_ridge_dual,
    _voc_ridge_primal,
    characteristic_managed_portfolios,
    random_fourier_features,
    sdf_ridge_loadings,
    voc_ridge,
)
from quant_fund.pipeline.train import RANKING_MODEL_NAMES, _fit_ranker, _make_ranker


def test_voc_ridge_primal_matches_dual() -> None:
    rng = np.random.default_rng(0)
    n_obs, n_features = 24, 10
    signals = rng.normal(size=(n_obs, n_features))
    y = rng.normal(size=n_obs)
    z = 0.7
    sample_t = float(n_obs)
    primal = _voc_ridge_primal(signals, y, z, sample_t)
    dual = _voc_ridge_dual(signals, y, z, sample_t)
    assert primal == pytest.approx(dual, rel=1e-8, abs=1e-8)
    auto = voc_ridge(signals, y, z)
    assert auto == pytest.approx(primal, rel=1e-8, abs=1e-8)


def test_voc_ridge_p_gt_t_uses_dual_and_is_finite() -> None:
    rng = np.random.default_rng(1)
    n_obs, n_features = 12, 40
    signals = rng.normal(size=(n_obs, n_features))
    y = rng.normal(size=n_obs)
    beta = voc_ridge(signals, y, z=1.0)
    assert beta.shape == (n_features,)
    assert np.all(np.isfinite(beta))
    dual = _voc_ridge_dual(signals, y, 1.0, float(n_obs))
    assert beta == pytest.approx(dual, rel=1e-8, abs=1e-8)


def test_rff_is_sin_cos_pairs_gamma_two() -> None:
    rng = np.random.default_rng(2)
    x = rng.normal(size=(8, 3))
    omega = rng.normal(size=(5, 3))
    s = random_fourier_features(x, omega, bandwidth=2.0)
    assert s.shape == (8, 10)
    proj = 2.0 * (x @ omega.T)
    assert s[:, :5] == pytest.approx(np.sin(proj))
    assert s[:, 5:] == pytest.approx(np.cos(proj))


def test_random_fourier_ranker_recovers_trig_signal() -> None:
    rng = np.random.default_rng(3)
    n = 200
    x = rng.normal(size=(n, 4))
    y = np.sin(2.0 * x[:, 0]) + 0.25 * np.cos(2.0 * x[:, 1]) + 0.05 * rng.normal(size=n)
    ranker = RandomFourierRanker(n_features=64, bandwidth=2.0, z=0.1, seed=3)
    pred = ranker.fit(x, y).predict(x)
    assert pred.shape == (n,)
    assert float(np.corrcoef(pred, y)[0, 1]) > 0.7
    meta = ranker.metadata()
    assert meta.name == "rff"
    assert meta.extra["bandwidth_gamma"] == 2.0
    assert "sharpe" not in meta.extra
    assert meta.extra["p_over_t"] < 1.0


def test_random_fourier_ranker_rejects_odd_p() -> None:
    with pytest.raises(ValueError, match="even"):
        RandomFourierRanker(n_features=3)


def test_sdf_ridge_shrinks_low_eigenvalue_pcs_more() -> None:
    rng = np.random.default_rng(4)
    t = 80
    # Two PCs: large variance vs tiny variance, equal PC-space means.
    f_large = rng.normal(loc=0.05, scale=1.0, size=t)
    f_small = rng.normal(loc=0.05, scale=0.05, size=t)
    managed = np.column_stack([f_large, f_small])
    z = 1.0
    b, mu, evals = sdf_ridge_loadings(managed, z)
    cov = np.cov(managed, rowvar=False, ddof=1)
    evals_c, evecs = np.linalg.eigh(cov)
    order = np.argsort(evals_c)[::-1]
    evals_c = evals_c[order]
    evecs = evecs[:, order]
    b_pc = evecs.T @ b
    mu_pc = evecs.T @ mu
    ols_pc = mu_pc / evals_c
    shrink = b_pc / ols_pc
    expected = evals_c / (evals_c + z)
    assert shrink == pytest.approx(expected, rel=1e-6, abs=1e-6)
    assert shrink[1] < shrink[0]
    assert evals[0] > evals[1]


def test_sdf_ridge_ranker_scores_are_z_dot_b() -> None:
    rng = np.random.default_rng(5)
    n_dates, n_names, n_char = 16, 12, 4
    x = rng.normal(size=(n_dates * n_names, n_char))
    dates = np.repeat(np.arange(n_dates), n_names)
    true_b = np.array([0.4, -0.2, 0.1, 0.0])
    y = x @ true_b + 0.05 * rng.normal(size=x.shape[0])
    ranker = SDFRidgeRanker(z=0.2)
    with pytest.raises(ValueError, match="dates"):
        ranker.fit(x, y)
    pred = ranker.fit(x, y, dates=dates).predict(x)
    assert float(np.corrcoef(pred, x @ true_b)[0, 1]) > 0.9
    meta = ranker.metadata()
    assert meta.name == "sdf_ridge"
    assert "sharpe" not in meta.extra
    managed = characteristic_managed_portfolios(x, y, dates)
    assert managed.shape == (n_dates, n_char)


def test_ipca_als_recovers_planted_expected_return() -> None:
    rng = np.random.default_rng(6)
    n_dates, n_names, n_char, k = 40, 28, 6, 2
    raw = rng.normal(size=(n_char, k))
    gamma_true, _ = np.linalg.qr(raw, mode="reduced")
    mu_f = np.array([0.15, 0.06])
    xs = []
    ys = []
    dates = []
    for t in range(n_dates):
        z_t = rng.normal(size=(n_names, n_char))
        f_t = mu_f + 0.15 * rng.normal(size=k)
        r_t = z_t @ (gamma_true @ f_t) + 0.02 * rng.normal(size=n_names)
        xs.append(z_t)
        ys.append(r_t)
        dates.append(np.full(n_names, t))
    x = np.vstack(xs)
    y = np.concatenate(ys)
    d = np.concatenate(dates)
    ranker = IPCARanker(n_factors=k, max_iter=40, tol=1e-6)
    with pytest.raises(ValueError, match="dates"):
        ranker.fit(x, y)
    pred = ranker.fit(x, y, dates=d).predict(x)
    true_er = x @ (gamma_true @ mu_f)
    assert float(np.corrcoef(pred, true_er)[0, 1]) > 0.85
    gram = ranker.gamma.T @ ranker.gamma
    assert gram == pytest.approx(np.eye(k), abs=1e-6)
    meta = ranker.metadata()
    assert meta.name == "ipca"
    assert "sharpe" not in meta.extra


def test_paper_rankers_in_train_catalog() -> None:
    cfg = load_config("configs/research.yaml")
    assert {"rff", "sdf_ridge", "ipca"} <= RANKING_MODEL_NAMES
    rff = _make_ranker("rff", cfg)
    sdf = _make_ranker("sdf_ridge", cfg)
    ipca = _make_ranker("ipca", cfg)
    assert rff.metadata().name == "rff"
    assert sdf.metadata().name == "sdf_ridge"
    assert ipca.metadata().name == "ipca"
    rng = np.random.default_rng(7)
    n_dates, n_names = 10, 8
    x = rng.normal(size=(n_dates * n_names, 3))
    y = rng.normal(size=x.shape[0])
    dates = np.repeat(np.arange(n_dates), n_names)
    _fit_ranker(rff, "rff", x, y, dates)
    _fit_ranker(sdf, "sdf_ridge", x, y, dates)
    _fit_ranker(ipca, "ipca", x, y, dates)
    for model in (rff, sdf, ipca):
        out = model.predict(x)
        assert out.shape == (x.shape[0],)
        assert np.all(np.isfinite(out))


def test_train_config_rejects_invalid_paper_ranker_hparams() -> None:
    with pytest.raises(ValueError, match="rff_n_features"):
        TrainConfig.model_validate({"rff_n_features": 3})
    with pytest.raises(ValueError, match="rff_gamma"):
        TrainConfig.model_validate({"rff_gamma": 0.0})
    with pytest.raises(ValueError, match="ipca_n_factors"):
        TrainConfig.model_validate({"ipca_n_factors": 0})
    with pytest.raises(ValueError, match="ipca_tol"):
        TrainConfig.model_validate({"ipca_tol": 0.0})
    with pytest.raises(ValueError):
        TrainConfig.model_validate({"sdf_ridge_z": float("nan")})
