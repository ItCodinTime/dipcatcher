"""RP-PCA, FNW group LASSO, Giglio–Xiu 3-pass, FGX double-selection, KNS EN."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.config.models import TrainConfig
from quant_fund.models.asset_pricing import IPCARanker
from quant_fund.models.cs_papers import (
    PAPER_RANKER_NAMES,
    AdaptiveLassoRanker,
    ClassicRanker,
    ClassicShortRanker,
    CombinationRanker,
    DoubleSelectionRanker,
    FamaMacBethRanker,
    FamaMacBethRidgeRanker,
    FNWRanker,
    GBRTRanker,
    GXThreePassRanker,
    ICWeightedCombinationRanker,
    MSFECombinationRanker,
    PCRRanker,
    PLSRanker,
    PrincipalPortfolioRanker,
    RPPCARanker,
    ReversalRanker,
    SDFElasticNetRanker,
    ThreePassFilterRanker,
    gx_three_pass_premia,
    quadratic_spline_basis,
    rp_pca_loadings,
    sdf_elastic_net_loadings,
)
from quant_fund.pipeline.train import RANKING_MODEL_NAMES, _fit_ranker, _make_ranker, _predict_ranker
from quant_fund.config import load_config


def test_rp_pca_gamma_minus_one_is_uncentered_covariance() -> None:
    rng = np.random.default_rng(0)
    managed = rng.normal(size=(40, 5))
    mu = managed.mean(axis=0)
    second = (managed.T @ managed) / managed.shape[0]
    cov0 = second - np.outer(mu, mu)
    lam, _, evals = rp_pca_loadings(managed, gamma=-1.0, n_factors=2)
    evals_c, evecs = np.linalg.eigh(cov0)
    assert evals[:2] == pytest.approx(np.sort(evals_c)[::-1][:2], rel=1e-6, abs=1e-6)
    assert lam.shape == (5, 2)


def test_rp_pca_overweights_priced_weak_factor() -> None:
    rng = np.random.default_rng(1)
    t = 120
    noisy = rng.normal(scale=1.0, size=t)
    priced = 0.12 + rng.normal(scale=0.12, size=t)
    managed = np.column_stack([noisy, priced])
    lam_pca, _, _ = rp_pca_loadings(managed, gamma=-1.0, n_factors=1)
    lam_rp, _, _ = rp_pca_loadings(managed, gamma=10.0, n_factors=1)
    assert abs(float(lam_rp[1, 0])) > abs(float(lam_pca[1, 0]))


def test_sdf_elastic_net_can_zero_a_coefficient() -> None:
    rng = np.random.default_rng(2)
    t = 60
    f0 = rng.normal(loc=0.04, scale=0.3, size=t)
    f1 = rng.normal(loc=0.0, scale=0.3, size=t)
    managed = np.column_stack([f0, f1])
    dense = sdf_elastic_net_loadings(managed, l2=0.2, l1=0.0)
    sparse = sdf_elastic_net_loadings(managed, l2=0.2, l1=2.0)
    assert np.count_nonzero(np.abs(sparse) > 1e-8) <= np.count_nonzero(np.abs(dense) > 1e-8)
    assert "sharpe" not in SDFElasticNetRanker(l1=0.1).metadata().extra


def test_gx_three_pass_finite_and_ranker_scores() -> None:
    rng = np.random.default_rng(3)
    n_dates, n_names, n_char = 20, 12, 4
    x = rng.normal(size=(n_dates * n_names, n_char))
    dates = np.repeat(np.arange(n_dates), n_names)
    y = x @ np.array([0.3, -0.1, 0.0, 0.05]) + 0.05 * rng.normal(size=x.shape[0])
    premia = gx_three_pass_premia(
        np.stack([x[dates == t].T @ y[dates == t] / n_names for t in range(n_dates)]),
        n_factors=2,
    )
    assert premia.shape == (n_char,)
    assert np.all(np.isfinite(premia))
    ranker = GXThreePassRanker(n_factors=2)
    pred = ranker.fit(x, y, dates=dates).predict(x)
    assert pred.shape == (x.shape[0],)
    assert "sharpe" not in ranker.metadata().extra


def test_fnw_spline_basis_and_recovers_nonlinear_characteristic() -> None:
    grid = np.linspace(0.0, 1.0, 11)
    basis = quadratic_spline_basis(grid, n_intervals=4)
    assert basis.shape == (11, 6)
    rng = np.random.default_rng(4)
    n_dates, n_names = 24, 16
    x = rng.normal(size=(n_dates * n_names, 3))
    dates = np.repeat(np.arange(n_dates), n_names)
    planted = (x[:, 0] ** 2) - 0.3 * x[:, 1]
    y = planted + 0.05 * rng.normal(size=x.shape[0])
    ranker = FNWRanker(n_intervals=3, lam=0.001)
    with pytest.raises(ValueError, match="dates"):
        ranker.fit(x, y)
    pred = ranker.fit(x, y, dates=dates).predict(x, dates=dates)
    assert float(np.corrcoef(pred, planted)[0, 1]) > 0.4
    assert ranker.metadata().name == "fnw"
    assert "sharpe" not in ranker.metadata().extra


def test_double_selection_keeps_planted_column() -> None:
    rng = np.random.default_rng(5)
    x = rng.normal(size=(200, 6))
    y = 1.4 * x[:, 2] - 0.8 * x[:, 0] + 0.05 * rng.normal(size=200)
    ranker = DoubleSelectionRanker(alpha=0.02)
    pred = ranker.fit(x, y).predict(x)
    assert 2 in ranker.selected
    assert float(np.corrcoef(pred, y)[0, 1]) > 0.7
    assert "sharpe" not in ranker.metadata().extra


def test_double_selection_survives_mixed_column_scale() -> None:
    """sklearn L1 is not scale-invariant; FGX columns are standardized first.

    A 1e8 noise column used to make post-selection OLS a constant
    (max |coef| ~ 1e-13 on the 5-day file tape).
    """
    rng = np.random.default_rng(5)
    x = rng.normal(size=(200, 6))
    y = 1.4 * x[:, 2] - 0.8 * x[:, 0] + 0.05 * rng.normal(size=200)
    x[:, 1] *= 1e8
    x[:, 4] *= 1e-8
    ranker = DoubleSelectionRanker(alpha=0.02)
    pred = ranker.fit(x, y).predict(x)
    assert 2 in ranker.selected
    assert 0 in ranker.selected
    assert float(np.corrcoef(pred, y)[0, 1]) > 0.7


def test_ipca_unrestricted_differs_when_alpha_present() -> None:
    rng = np.random.default_rng(6)
    n_dates, n_names, n_char, k = 24, 18, 5, 2
    raw = rng.normal(size=(n_char, k))
    gamma, _ = np.linalg.qr(raw, mode="reduced")
    alpha = np.array([0.12, 0.0, -0.08, 0.0, 0.0])
    xs, ys, dates = [], [], []
    for t in range(n_dates):
        z_t = rng.normal(size=(n_names, n_char))
        f_t = np.array([0.05, 0.02]) + 0.1 * rng.normal(size=k)
        r_t = z_t @ alpha + z_t @ (gamma @ f_t) + 0.02 * rng.normal(size=n_names)
        xs.append(z_t)
        ys.append(r_t)
        dates.append(np.full(n_names, t))
    x = np.vstack(xs)
    y = np.concatenate(ys)
    d = np.concatenate(dates)
    rest = IPCARanker(n_factors=k, unrestricted=False).fit(x, y, dates=d)
    unres = IPCARanker(n_factors=k, unrestricted=True).fit(x, y, dates=d)
    assert unres.metadata().name == "ipca_alpha"
    assert float(np.linalg.norm(rest.gamma_alpha)) == pytest.approx(0.0, abs=1e-12)
    assert float(np.linalg.norm(unres.gamma_alpha)) > 1e-3
    assert float(np.corrcoef(unres.predict(x), y)[0, 1]) > float(np.corrcoef(rest.predict(x), y)[0, 1])
    assert float(np.corrcoef(unres.gamma_alpha, alpha)[0, 1]) > 0.5


def test_catalog_includes_all_paper_rankers() -> None:
    cfg = load_config("configs/research.yaml")
    assert set(PAPER_RANKER_NAMES) <= RANKING_MODEL_NAMES
    rng = np.random.default_rng(8)
    n_dates, n_names = 12, 10
    x = rng.normal(size=(n_dates * n_names, 4))
    y = rng.normal(size=x.shape[0])
    dates = np.repeat(np.arange(n_dates), n_names)
    ids = np.tile(np.arange(n_names), n_dates)
    for name in PAPER_RANKER_NAMES:
        model = _make_ranker(name, cfg)
        _fit_ranker(model, name, x, y, dates, ids)
        out = _predict_ranker(model, name, x, dates, ids)
        assert out.shape == (x.shape[0],)
        assert np.all(np.isfinite(out))
        assert float(np.std(out)) > 0.0
        assert "sharpe" not in (model.metadata().extra or {})


def test_train_config_rejects_unknown_paper_ranker() -> None:
    with pytest.raises(ValueError, match="paper_rankers"):
        TrainConfig.model_validate({"paper_rankers": ["ridge"]})
    with pytest.raises(ValueError, match="rp_pca_gamma"):
        TrainConfig.model_validate({"rp_pca_gamma": -1.5})
    with pytest.raises(ValueError, match="tprf_n_factors"):
        TrainConfig.model_validate({"tprf_n_factors": 0})
    with pytest.raises(ValueError, match="gbrt_learning_rate"):
        TrainConfig.model_validate({"gbrt_learning_rate": 0.0})
    with pytest.raises(ValueError, match="alasso_alpha"):
        TrainConfig.model_validate({"alasso_alpha": -0.01})


def test_fnw_high_lambda_is_not_constant() -> None:
    rng = np.random.default_rng(11)
    n_dates, n_names = 16, 12
    x = rng.normal(size=(n_dates * n_names, 3))
    dates = np.repeat(np.arange(n_dates), n_names)
    y = 0.2 * x[:, 0] ** 2 + 0.02 * rng.normal(size=x.shape[0])
    pred = FNWRanker(n_intervals=3, lam=5.0).fit(x, y, dates=dates).predict(x, dates=dates)
    assert np.all(np.isfinite(pred))
    assert float(np.std(pred)) > 1e-8


def test_sdf_en_does_not_collapse_to_zero_on_small_means() -> None:
    rng = np.random.default_rng(12)
    t = 40
    x = rng.normal(size=(t * 8, 4))
    y = 0.001 * rng.normal(size=x.shape[0]) + 0.002 * x[:, 0]
    dates = np.repeat(np.arange(t), 8)
    ranker = SDFElasticNetRanker(l2=1.0, l1=0.05)
    pred = ranker.fit(x, y, dates=dates).predict(x)
    assert np.all(np.isfinite(pred))
    assert float(np.max(np.abs(ranker.b))) > 0.0
    assert "sharpe" not in ranker.metadata().extra


def test_fama_macbeth_recovers_cs_slope() -> None:
    rng = np.random.default_rng(13)
    n_dates, n_names, n_char = 20, 14, 3
    x = rng.normal(size=(n_dates * n_names, n_char))
    dates = np.repeat(np.arange(n_dates), n_names)
    truth = np.array([0.4, -0.2, 0.0])
    y = x @ truth + 0.05 * rng.normal(size=x.shape[0])
    ranker = FamaMacBethRanker().fit(x, y, dates=dates)
    assert float(np.corrcoef(ranker.lambda_bar, truth)[0, 1]) > 0.8
    pred = ranker.predict(x)
    assert float(np.corrcoef(pred, y)[0, 1]) > 0.6
    assert ranker.metadata().name == "fm"


def test_pcr_pls_tprf_recover_linear_signal() -> None:
    rng = np.random.default_rng(14)
    factor = rng.normal(size=(400, 1))
    loadings = np.array([[1.2, 0.9, 0.4, 0.05, 0.0, -0.1]])
    x = factor @ loadings + 0.25 * rng.normal(size=(400, 6))
    y = factor[:, 0] + 0.05 * rng.normal(size=400)
    for cls, kwargs in (
        (PCRRanker, {"n_factors": 2}),
        (PLSRanker, {"n_factors": 2}),
        (ThreePassFilterRanker, {"n_factors": 2}),
    ):
        ranker = cls(**kwargs).fit(x, y)
        pred = ranker.predict(x)
        assert float(np.corrcoef(pred, y)[0, 1]) > 0.7
        assert "sharpe" not in ranker.metadata().extra


def test_gbrt_fits_nonlinear_characteristic() -> None:
    rng = np.random.default_rng(15)
    x = rng.normal(size=(500, 4))
    planted = (x[:, 0] ** 2) - 0.8 * x[:, 1]
    y = planted + 0.05 * rng.normal(size=500)
    ranker = GBRTRanker(n_estimators=30, max_depth=2, learning_rate=0.1, seed=15).fit(x, y)
    pred = ranker.predict(x)
    assert float(np.corrcoef(pred, planted)[0, 1]) > 0.5
    assert ranker.metadata().name == "gbrt"


def test_principal_portfolios_uses_cross_predictability() -> None:
    rng = np.random.default_rng(16)
    n_dates, n_names = 30, 10
    ids = np.tile(np.arange(n_names), n_dates)
    dates = np.repeat(np.arange(n_dates), n_names)
    x = rng.normal(size=(n_dates * n_names, 3))
    signal = x[:, 0]
    y = np.zeros(x.shape[0])
    for t in range(n_dates):
        sl = slice(t * n_names, (t + 1) * n_names)
        s = signal[sl]
        y[sl] = 0.4 * s + 0.35 * np.roll(s, 1) + 0.05 * rng.normal(size=n_names)
    ranker = PrincipalPortfolioRanker(n_factors=6).fit(x, y, dates=dates, ids=ids)
    pred = ranker.predict(x, dates=dates, ids=ids)
    own = x @ ranker.beta
    assert float(np.corrcoef(pred, y)[0, 1]) > float(np.corrcoef(own, y)[0, 1])
    assert float(np.linalg.norm(ranker.pi_k - np.diag(np.diag(ranker.pi_k)))) > 1e-8
    assert ranker.metadata().name == "pp"


def test_combo_and_alasso_recover_linear_signal() -> None:
    rng = np.random.default_rng(17)
    x = rng.normal(size=(300, 5))
    y = 0.9 * x[:, 0] - 0.4 * x[:, 2] + 0.05 * rng.normal(size=300)
    combo = CombinationRanker().fit(x, y).predict(x)
    alasso = AdaptiveLassoRanker(alpha=0.01).fit(x, y).predict(x)
    assert float(np.corrcoef(combo, y)[0, 1]) > 0.5
    assert float(np.corrcoef(alasso, y)[0, 1]) > 0.7
    assert "sharpe" not in CombinationRanker().metadata().extra
    assert AdaptiveLassoRanker().metadata().name == "alasso"


def test_classic_uses_a_priori_signs() -> None:
    rng = np.random.default_rng(18)
    names = ["cs_z_reversal_1", "cs_z_max_ret_20", "noise"]
    x = rng.normal(size=(80, 3))
    y = x[:, 0] - x[:, 1] + 0.05 * rng.normal(size=80)
    pred = ClassicRanker().fit(x, y, features=names).predict(x)
    assert float(np.corrcoef(pred, y)[0, 1]) > 0.7
    assert ClassicRanker().metadata().name == "classic"


def test_fm_ridge_fits_when_ols_fm_cannot() -> None:
    rng = np.random.default_rng(19)
    n_dates, n_names, n_char = 16, 10, 9
    x = rng.normal(size=(n_dates * n_names, n_char))
    dates = np.repeat(np.arange(n_dates), n_names)
    y = 0.4 * x[:, 0] - 0.3 * x[:, 1] + 0.05 * rng.normal(size=x.shape[0])
    with pytest.raises(ValueError, match="enough names"):
        FamaMacBethRanker().fit(x, y, dates=dates)
    pred = FamaMacBethRidgeRanker(alpha=1.0).fit(x, y, dates=dates).predict(x)
    assert float(np.corrcoef(pred, y)[0, 1]) > 0.4
    assert FamaMacBethRidgeRanker().metadata().name == "fm_ridge"


def test_combo_ic_uses_nonnegative_train_ic_weights() -> None:
    rng = np.random.default_rng(20)
    n_dates, n_names = 20, 12
    x = rng.normal(size=(n_dates * n_names, 3))
    dates = np.repeat(np.arange(n_dates), n_names)
    y = 0.8 * x[:, 0] + 0.05 * rng.normal(size=x.shape[0])
    ranker = ICWeightedCombinationRanker().fit(x, y, dates=dates)
    assert ranker.weights is not None
    assert float(np.min(ranker.weights)) >= -1e-15
    assert float(np.sum(ranker.weights)) == pytest.approx(1.0)
    assert ranker.weights[0] == pytest.approx(float(np.max(ranker.weights)))
    pred = ranker.predict(x)
    assert float(np.corrcoef(pred, y)[0, 1]) > 0.5
    assert ranker.metadata().name == "combo_ic"


def test_classic_st_uses_daily_cs_signs() -> None:
    rng = np.random.default_rng(21)
    names = ["cs_z_reversal_1", "cs_z_ret_5", "cs_z_mom_skip_5_20", "noise"]
    x = rng.normal(size=(100, 4))
    y = x[:, 0] - x[:, 1] + x[:, 2] + 0.05 * rng.normal(size=100)
    pred = ClassicShortRanker().fit(x, y, features=names).predict(x)
    assert float(np.corrcoef(pred, y)[0, 1]) > 0.7
    assert ClassicShortRanker().metadata().name == "classic_st"
    rev = ReversalRanker().fit(x, y, features=names).predict(x)
    assert float(np.corrcoef(rev, x[:, 0])[0, 1]) > 0.99


def test_combo_msfe_upweights_low_error_univariate() -> None:
    rng = np.random.default_rng(22)
    n_dates, n_names = 40, 12
    x = rng.normal(size=(n_dates * n_names, 3))
    dates = np.repeat(np.arange(n_dates), n_names)
    y = 0.9 * x[:, 0] + 0.05 * rng.normal(size=x.shape[0])
    ranker = MSFECombinationRanker(theta=0.99).fit(x, y, dates=dates)
    assert ranker.weights is not None
    assert float(np.sum(ranker.weights)) == pytest.approx(1.0)
    assert ranker.weights[0] == pytest.approx(float(np.max(ranker.weights)))
    pred = ranker.predict(x)
    assert float(np.corrcoef(pred, y)[0, 1]) > 0.5
    assert ranker.metadata().name == "combo_msfe"


def test_ridge_st_masks_to_short_horizon_columns() -> None:
    from quant_fund.models.cs_papers import SHORT_HORIZON_FEATURES, make_ridge_st

    rng = np.random.default_rng(23)
    names = ["cs_z_reversal_1", "cs_z_amihud", "noise"]
    n_dates, n_names = 24, 10
    x = rng.normal(size=(n_dates * n_names, 3))
    dates = np.repeat(np.arange(n_dates), n_names)
    y = 0.8 * x[:, 0] + 0.05 * rng.normal(size=x.shape[0])
    ranker = make_ridge_st(0.3).fit(x, y, dates=dates, features=names)
    assert ranker._idx is not None
    assert list(ranker._idx) == [0]
    pred = ranker.predict(x)
    assert float(np.corrcoef(pred, y)[0, 1]) > 0.5
    assert ranker.metadata().name == "ridge_st"
    assert "cs_z_reversal_1" in SHORT_HORIZON_FEATURES


def test_hedge_lab_uses_rolling_daily_cs_window() -> None:
    cfg = load_config("configs/hedge_lab.yaml")
    assert cfg.validation.scheme == "rolling"
    assert cfg.validation.train_bars == 252
    wide = load_config("configs/hedge_lab_wide.yaml")
    assert wide.validation.scheme == "rolling"
