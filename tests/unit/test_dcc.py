"""Day Wave 116: DCC stage-1 uses arch GARCH, not EWMA.

Day Wave 122: named DCC specs; ADCC was fail-closed until Wave 128.
Day Wave 123: Gaussian DCC returns one-step-ahead H_{t+1}, not last in-sample H_t.
Day Wave 124: named optimize_asof / /risk/portfolio path for dcc_gaussian.
Day Wave 125: trailing contiguous complete-case window; incomplete terminal fails closed.
Day Wave 126: Student-t DCC two-stage likelihood.
Day Wave 127: named optimize_asof / /risk/portfolio path for dcc_student_t.
Day Wave 128: Cappiello–Engle–Sheppard scalar ADCC catalog estimator; optimizer unwired.
Day Wave 133: Bollerslev (1990) CCC catalog estimator; optimizer unwired.
Day Wave 134: named optimize_asof / /risk/portfolio path for ccc.
Day Wave 135: diagonal CES AG-DCC catalog estimator; optimizer unwired.
Day Wave 136: named optimize_asof / /risk/portfolio path for agdcc.
Day Wave 137: unrestricted CES AG-DCC catalog estimator; optimizer unwired.
Day Wave 138: named optimize_asof / /risk/portfolio path for agdcc_full.
"""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.covariance import (
    DCC_COVARIANCE_OBJECT_ONE_STEP,
    DCC_FAMILY_ADCC,
    DCC_FAMILY_AGDCC,
    DCC_FAMILY_AGDCC_FULL,
    DCC_FAMILY_CCC,
    DCC_FAMILY_GAUSSIAN,
    DCC_FAMILY_STUDENT_T,
    DCC_PARAMETERIZATION_DIAGONAL,
    DCC_PARAMETERIZATION_FULL,
    DCC_SAMPLE_TRAILING_COMPLETE,
    DCC_SPEC_ADCC,
    DCC_SPEC_AGDCC,
    DCC_SPEC_AGDCC_FULL,
    DCC_SPEC_CCC,
    DCC_SPEC_ENGLE_2002,
    DCC_SPEC_STUDENT_T,
    DCC_STAGE1_MIN_OBS,
    OPTIMIZER_COVARIANCE_EWMA,
    OPTIMIZER_COVARIANCE_LEDOIT_WOLF,
    OPTIMIZER_COVARIANCE_LEDOIT_WOLF_NONLINEAR,
    OPTIMIZER_COVARIANCE_OAS,
    OPTIMIZER_COVARIANCE_SAMPLE,
    _adcc_kappa,
    _adcc_negative_shocks,
    _adcc_step_q,
    _agdcc_full_intercept,
    _agdcc_full_step_q,
    _agdcc_intercept,
    _agdcc_step_q,
    _dcc_qbar,
    adcc,
    agdcc,
    agdcc_full,
    ccc,
    dcc_gaussian,
    dcc_student_t,
    dcc_trailing_complete_window,
    ewma_variance_1d,
    min_eigenvalue,
    require_implemented_dcc_spec,
    require_implemented_optimizer_covariance,
    student_t_corr_nll,
)
from quant_fund.models.volatility import GARCHVol


def _simulate_ccc_garch(n: int = 240, *, rho: float = 0.7, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    omega, alpha, beta = 1e-6, 0.08, 0.88
    h1 = omega / (1.0 - alpha - beta)
    h2 = h1
    chol = np.linalg.cholesky(np.array([[1.0, rho], [rho, 1.0]]))
    returns = np.empty((n, 2))
    for t in range(n):
        z = chol @ rng.normal(size=2)
        returns[t, 0] = np.sqrt(h1) * z[0]
        returns[t, 1] = np.sqrt(h2) * z[1]
        h1 = omega + alpha * returns[t, 0] ** 2 + beta * h1
        h2 = omega + alpha * returns[t, 1] ** 2 + beta * h2
    return returns


def test_dcc_psd_and_params() -> None:
    rng = np.random.default_rng(0)
    e = rng.normal(size=(240, 4))
    h, params = dcc_gaussian(e * 0.01)
    assert min_eigenvalue(h) >= -1e-8
    assert float(params["a"]) >= 0
    assert float(params["b"]) >= 0
    assert float(params["a"]) + float(params["b"]) < 1.0 + 1e-6
    assert params["stage1"] == "garch"
    assert params["stage1_vol"] == "garch"
    assert params["stage1_dist"] == "normal"
    assert params["family"] == DCC_FAMILY_GAUSSIAN
    assert params["spec"] == DCC_SPEC_ENGLE_2002
    assert params["dist"] == "normal"
    assert params["asymmetric"] == "false"
    assert params["covariance_object"] == DCC_COVARIANCE_OBJECT_ONE_STEP
    assert float(params["horizon"]) == 1.0
    assert params["sample"] == DCC_SAMPLE_TRAILING_COMPLETE
    assert float(params["n_prefix_dropped"]) == 0.0


def test_dcc_stage1_uses_garch_not_ewma() -> None:
    returns = _simulate_ccc_garch()
    h, params = dcc_gaussian(returns)
    assert params["stage1"] == "garch"
    one_step = np.empty(returns.shape[1])
    last_in_sample = np.empty(returns.shape[1])
    last_ewma = np.empty(returns.shape[1])
    for j in range(returns.shape[1]):
        model = GARCHVol(
            min_obs=DCC_STAGE1_MIN_OBS, mean="Constant", series_scope="dcc_stage1_univariate"
        )
        model.fit_returns(returns[:, j])
        sigma, _z = model.in_sample_sigma_and_z()
        last_in_sample[j] = float(sigma[-1] ** 2)
        one_step[j] = float(np.asarray(model.forecast(horizon=1)["variance"], dtype=float)[0])
        last_ewma[j] = float(ewma_variance_1d(returns[:, j])[-1])
    np.testing.assert_allclose(np.diag(h), one_step, rtol=1e-8, atol=1e-16)
    assert not np.allclose(one_step, last_ewma, rtol=1e-3, atol=1e-12)
    assert not np.allclose(last_in_sample, last_ewma, rtol=1e-3, atol=1e-12)


def test_dcc_gaussian_is_one_step_ahead_not_in_sample_last() -> None:
    """A terminal shock must enter D_{t+1} / Q_{t+1}, not be dropped as H_t."""
    returns = _simulate_ccc_garch(n=240, seed=19)
    shocked = returns.copy()
    shocked[-1] = np.array([0.04, -0.035])
    h, params = dcc_gaussian(shocked)
    one_step = np.empty(shocked.shape[1])
    last_in_sample = np.empty(shocked.shape[1])
    for j in range(shocked.shape[1]):
        model = GARCHVol(
            min_obs=DCC_STAGE1_MIN_OBS, mean="Constant", series_scope="dcc_stage1_univariate"
        )
        model.fit_returns(shocked[:, j])
        sigma, _z = model.in_sample_sigma_and_z()
        last_in_sample[j] = float(sigma[-1] ** 2)
        one_step[j] = float(np.asarray(model.forecast(horizon=1)["variance"], dtype=float)[0])
    np.testing.assert_allclose(np.diag(h), one_step, rtol=1e-8, atol=1e-16)
    assert np.all(one_step > last_in_sample)
    assert not np.allclose(np.diag(h), last_in_sample, rtol=1e-3, atol=1e-16)
    assert params["covariance_object"] == DCC_COVARIANCE_OBJECT_ONE_STEP
    assert float(params["horizon"]) == 1.0


def test_dcc_recovers_synthetic_ccc_correlation() -> None:
    coupled = _simulate_ccc_garch(rho=0.7, seed=11)
    independent = _simulate_ccc_garch(rho=0.0, seed=13)
    h_c, params_c = dcc_gaussian(coupled)
    h_i, _params_i = dcc_gaussian(independent)
    d_c = np.sqrt(np.clip(np.diag(h_c), 1e-16, None))
    d_i = np.sqrt(np.clip(np.diag(h_i), 1e-16, None))
    corr_c = float((h_c / np.outer(d_c, d_c))[0, 1])
    corr_i = float((h_i / np.outer(d_i, d_i))[0, 1])
    assert corr_c > 0.35
    assert abs(corr_c) > abs(corr_i)
    assert 0.0 <= float(params_c["a"]) + float(params_c["b"]) < 1.0 + 1e-6
    assert min_eigenvalue(h_c) >= -1e-8
    assert min_eigenvalue(h_i) >= -1e-8


def test_dcc_stage1_fail_closed_on_zero_variance() -> None:
    with pytest.raises(ValueError, match="stage-1 GARCH"):
        dcc_gaussian(np.ones((80, 2)) * 0.01)


def test_dcc_stage1_fail_closed_on_short_history() -> None:
    rng = np.random.default_rng(3)
    with pytest.raises(ValueError, match=f"at least {DCC_STAGE1_MIN_OBS}"):
        dcc_gaussian(rng.normal(size=(20, 3)) * 0.01)


def test_dcc_refuses_incomplete_terminal_row() -> None:
    returns = _simulate_ccc_garch(n=80, seed=23)
    returns[-1, 0] = np.nan
    with pytest.raises(ValueError, match="incomplete_terminal_row"):
        dcc_gaussian(returns)
    window = dcc_trailing_complete_window(np.vstack([returns[:-1], [[0.01, -0.01]]]))
    assert window.shape[0] == 80
    assert np.isfinite(window).all()


def test_dcc_does_not_concatenate_across_interior_holes() -> None:
    """Pre-hole shocks must not enter H_{t+1}; n_obs is the suffix, not the stitch."""
    base = _simulate_ccc_garch(n=120, seed=29)
    hole_i = 40
    shocked = base.copy()
    shocked[hole_i - 1] = np.array([0.2, -0.2])
    holed = shocked.copy()
    holed[hole_i] = np.array([np.nan, np.nan])
    suffix = base[hole_i + 1 :]
    h_holed, params_holed = dcc_gaussian(holed)
    h_suffix, params_suffix = dcc_gaussian(suffix)
    np.testing.assert_allclose(h_holed, h_suffix, rtol=1e-10, atol=1e-16)
    assert float(params_holed["n_obs"]) == float(suffix.shape[0])
    assert float(params_holed["n_prefix_dropped"]) == float(hole_i + 1)
    assert params_holed["sample"] == DCC_SAMPLE_TRAILING_COMPLETE
    assert float(params_suffix["n_prefix_dropped"]) == 0.0
    stitched = np.vstack([shocked[:hole_i], shocked[hole_i + 1 :]])
    h_stitched, params_stitched = dcc_gaussian(stitched)
    assert float(params_stitched["n_obs"]) == float(stitched.shape[0])
    assert float(params_stitched["n_obs"]) != float(params_holed["n_obs"])
    assert not np.allclose(h_holed, h_stitched, rtol=1e-8, atol=1e-16)


def test_dcc_leading_incomplete_rows_are_unused() -> None:
    returns = _simulate_ccc_garch(n=80, seed=31)
    padded = np.vstack([np.full((5, 2), np.nan), returns])
    h_padded, params_padded = dcc_gaussian(padded)
    h_plain, params_plain = dcc_gaussian(returns)
    np.testing.assert_allclose(h_padded, h_plain, rtol=1e-10, atol=1e-16)
    assert float(params_padded["n_obs"]) == float(returns.shape[0])
    assert float(params_padded["n_prefix_dropped"]) == 5.0
    assert float(params_plain["n_prefix_dropped"]) == 0.0


def test_dcc_short_contiguous_suffix_fails_closed() -> None:
    returns = _simulate_ccc_garch(n=80, seed=37)
    returns[-(DCC_STAGE1_MIN_OBS - 1)] = np.array([np.nan, np.nan])
    with pytest.raises(ValueError, match="insufficient_contiguous_rows"):
        dcc_gaussian(returns)


def test_dcc_student_t_is_catalog_estimator_not_gaussian() -> None:
    returns = _simulate_ccc_garch(n=120, seed=41)
    h, params = dcc_student_t(returns)
    assert min_eigenvalue(h) >= -1e-8
    assert params["family"] == DCC_FAMILY_STUDENT_T
    assert params["spec"] == DCC_SPEC_STUDENT_T
    assert params["dist"] == "student_t"
    assert params["stage1"] == "garch"
    assert params["stage1_dist"] == "t"
    assert params["asymmetric"] == "false"
    assert params["covariance_object"] == DCC_COVARIANCE_OBJECT_ONE_STEP
    assert params["sample"] == DCC_SAMPLE_TRAILING_COMPLETE
    assert float(params["horizon"]) == 1.0
    assert float(params["nu"]) > 2.0
    assert np.isfinite(float(params["nu"]))
    assert float(params["a"]) >= 0.0
    assert float(params["b"]) >= 0.0
    assert float(params["a"]) + float(params["b"]) < 1.0 + 1e-6
    one_step = np.empty(returns.shape[1])
    for j in range(returns.shape[1]):
        model = GARCHVol(
            dist="t",
            min_obs=DCC_STAGE1_MIN_OBS,
            mean="Constant",
            series_scope="dcc_stage1_univariate",
        )
        model.fit_returns(returns[:, j])
        one_step[j] = float(np.asarray(model.forecast(horizon=1)["variance"], dtype=float)[0])
    np.testing.assert_allclose(np.diag(h), one_step, rtol=1e-8, atol=1e-16)


def test_dcc_student_t_does_not_call_gaussian(monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom(*_args: object, **_kwargs: object) -> tuple[object, dict[str, str]]:
        raise AssertionError("student-t DCC must not silently run dcc_gaussian")

    monkeypatch.setattr("quant_fund.models.covariance.dcc_gaussian", _boom)
    returns = _simulate_ccc_garch(n=80, seed=43)
    h, params = dcc_student_t(returns)
    assert params["family"] == DCC_FAMILY_STUDENT_T
    assert min_eigenvalue(h) >= -1e-8


def test_adcc_is_catalog_estimator_not_gaussian() -> None:
    returns = _simulate_ccc_garch(n=120, seed=67)
    h, params = adcc(returns)
    assert min_eigenvalue(h) >= -1e-8
    assert params["family"] == DCC_FAMILY_ADCC
    assert params["spec"] == DCC_SPEC_ADCC
    assert params["dist"] == "normal"
    assert params["stage1"] == "garch"
    assert params["stage1_dist"] == "normal"
    assert params["asymmetric"] == "true"
    assert params["covariance_object"] == DCC_COVARIANCE_OBJECT_ONE_STEP
    assert params["sample"] == DCC_SAMPLE_TRAILING_COMPLETE
    assert float(params["horizon"]) == 1.0
    assert float(params["g"]) >= 0.0
    assert float(params["kappa"]) >= 0.0
    assert np.isfinite(float(params["kappa"]))
    assert float(params["a"]) >= 0.0
    assert float(params["b"]) >= 0.0
    assert (
        float(params["a"]) + float(params["b"]) + float(params["kappa"]) * float(params["g"])
        < 1.0 + 1e-6
    )
    one_step = np.empty(returns.shape[1])
    for j in range(returns.shape[1]):
        model = GARCHVol(
            min_obs=DCC_STAGE1_MIN_OBS, mean="Constant", series_scope="dcc_stage1_univariate"
        )
        model.fit_returns(returns[:, j])
        one_step[j] = float(np.asarray(model.forecast(horizon=1)["variance"], dtype=float)[0])
    np.testing.assert_allclose(np.diag(h), one_step, rtol=1e-8, atol=1e-16)


def test_adcc_does_not_call_gaussian_or_student_t(monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom(*_args: object, **_kwargs: object) -> tuple[object, dict[str, str]]:
        raise AssertionError("ADCC must not silently run an Engle DCC fitter")

    monkeypatch.setattr("quant_fund.models.covariance.dcc_gaussian", _boom)
    monkeypatch.setattr("quant_fund.models.covariance.dcc_student_t", _boom)
    monkeypatch.setattr("quant_fund.models.covariance.agdcc", _boom)
    monkeypatch.setattr("quant_fund.models.covariance.agdcc_full", _boom)
    returns = _simulate_ccc_garch(n=80, seed=71)
    h, params = adcc(returns)
    assert params["family"] == DCC_FAMILY_ADCC
    assert params["asymmetric"] == "true"
    assert min_eigenvalue(h) >= -1e-8


def test_adcc_q_step_uses_negative_outer_product() -> None:
    qbar = np.eye(2)
    nbar = 0.25 * np.eye(2)
    q = qbar.copy()
    a, b, g = 0.05, 0.90, 0.04
    z_pos = np.array([1.0, 1.0])
    z_neg = np.array([-1.0, -1.0])
    q_pos = _adcc_step_q(q, qbar, nbar, z_pos, _adcc_negative_shocks(z_pos), a, b, g)
    q_neg = _adcc_step_q(q, qbar, nbar, z_neg, _adcc_negative_shocks(z_neg), a, b, g)
    assert float(q_neg[0, 1]) == pytest.approx(a + g)
    assert float(q_pos[0, 1]) == pytest.approx(a)
    assert float(q_neg[0, 1]) > float(q_pos[0, 1])


def test_adcc_kappa_is_largest_generalized_eigenvalue() -> None:
    qbar = np.eye(2)
    nbar = np.array([[0.50, 0.10], [0.10, 0.40]])
    expected = float(np.max(np.linalg.eigvalsh(nbar)))
    assert _adcc_kappa(qbar, nbar) == pytest.approx(expected)
    with pytest.raises(ValueError, match="matching square"):
        _adcc_kappa(np.eye(2), np.eye(3))


def test_adcc_refuses_negative_g0() -> None:
    returns = _simulate_ccc_garch(n=80, seed=73)
    with pytest.raises(ValueError, match="g0 must be finite and non-negative"):
        adcc(returns, g0=-0.1)


def test_adcc_refuses_incomplete_terminal_row() -> None:
    returns = _simulate_ccc_garch(n=80, seed=79)
    returns[-1, 0] = np.nan
    with pytest.raises(ValueError, match="incomplete_terminal_row"):
        adcc(returns)


def test_adcc_does_not_concatenate_across_interior_holes() -> None:
    base = _simulate_ccc_garch(n=120, seed=83)
    hole_i = 40
    holed = base.copy()
    holed[hole_i] = np.array([np.nan, np.nan])
    suffix = base[hole_i + 1 :]
    h_holed, params_holed = adcc(holed)
    h_suffix, params_suffix = adcc(suffix)
    np.testing.assert_allclose(h_holed, h_suffix, rtol=1e-10, atol=1e-16)
    assert float(params_holed["n_obs"]) == float(suffix.shape[0])
    assert float(params_holed["n_prefix_dropped"]) == float(hole_i + 1)
    assert params_holed["sample"] == DCC_SAMPLE_TRAILING_COMPLETE
    assert float(params_suffix["n_prefix_dropped"]) == 0.0


def test_ccc_is_catalog_estimator_not_dcc() -> None:
    returns = _simulate_ccc_garch(n=120, seed=91)
    h, params = ccc(returns)
    assert min_eigenvalue(h) >= -1e-8
    assert params["family"] == DCC_FAMILY_CCC
    assert params["spec"] == DCC_SPEC_CCC
    assert params["dist"] == "normal"
    assert params["stage1"] == "garch"
    assert params["stage1_dist"] == "normal"
    assert params["asymmetric"] == "false"
    assert params["dynamic_correlation"] == "false"
    assert "a" not in params
    assert "b" not in params
    assert params["covariance_object"] == DCC_COVARIANCE_OBJECT_ONE_STEP
    assert params["sample"] == DCC_SAMPLE_TRAILING_COMPLETE
    assert float(params["horizon"]) == 1.0
    z = np.zeros_like(returns)
    one_step = np.empty(returns.shape[1])
    for j in range(returns.shape[1]):
        model = GARCHVol(
            min_obs=DCC_STAGE1_MIN_OBS, mean="Constant", series_scope="dcc_stage1_univariate"
        )
        model.fit_returns(returns[:, j])
        _sigma, z_j = model.in_sample_sigma_and_z()
        z[:, j] = z_j
        one_step[j] = float(np.asarray(model.forecast(horizon=1)["variance"], dtype=float)[0])
    np.testing.assert_allclose(np.diag(h), one_step, rtol=1e-8, atol=1e-16)
    qbar = _dcc_qbar(z)
    d = np.sqrt(np.clip(np.diag(h), 1e-16, None))
    r = h / np.outer(d, d)
    np.fill_diagonal(r, 1.0)
    np.testing.assert_allclose(r, qbar, rtol=1e-8, atol=1e-10)


def test_ccc_does_not_call_dcc_or_adcc(monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom(*_args: object, **_kwargs: object) -> tuple[object, dict[str, str]]:
        raise AssertionError("CCC must not silently run a DCC or ADCC fitter")

    monkeypatch.setattr("quant_fund.models.covariance.dcc_gaussian", _boom)
    monkeypatch.setattr("quant_fund.models.covariance.dcc_student_t", _boom)
    monkeypatch.setattr("quant_fund.models.covariance.adcc", _boom)
    monkeypatch.setattr("quant_fund.models.covariance.agdcc", _boom)
    monkeypatch.setattr("quant_fund.models.covariance.agdcc_full", _boom)
    returns = _simulate_ccc_garch(n=120, seed=97)
    h, params = ccc(returns)
    assert params["family"] == DCC_FAMILY_CCC
    assert params["dynamic_correlation"] == "false"
    assert min_eigenvalue(h) >= -1e-8


def test_ccc_recovers_synthetic_constant_correlation() -> None:
    coupled = _simulate_ccc_garch(rho=0.7, seed=101)
    independent = _simulate_ccc_garch(rho=0.0, seed=103)
    h_c, _params_c = ccc(coupled)
    h_i, _params_i = ccc(independent)
    d_c = np.sqrt(np.clip(np.diag(h_c), 1e-16, None))
    d_i = np.sqrt(np.clip(np.diag(h_i), 1e-16, None))
    corr_c = float((h_c / np.outer(d_c, d_c))[0, 1])
    corr_i = float((h_i / np.outer(d_i, d_i))[0, 1])
    assert corr_c > 0.35
    assert abs(corr_c) > abs(corr_i)
    assert min_eigenvalue(h_c) >= -1e-8
    assert min_eigenvalue(h_i) >= -1e-8


def test_ccc_is_one_step_ahead_not_in_sample_last() -> None:
    returns = _simulate_ccc_garch(n=240, seed=107)
    shocked = returns.copy()
    shocked[-1] = np.array([0.04, -0.035])
    h, params = ccc(shocked)
    one_step = np.empty(shocked.shape[1])
    last_in_sample = np.empty(shocked.shape[1])
    for j in range(shocked.shape[1]):
        model = GARCHVol(
            min_obs=DCC_STAGE1_MIN_OBS, mean="Constant", series_scope="dcc_stage1_univariate"
        )
        model.fit_returns(shocked[:, j])
        sigma, _z = model.in_sample_sigma_and_z()
        last_in_sample[j] = float(sigma[-1] ** 2)
        one_step[j] = float(np.asarray(model.forecast(horizon=1)["variance"], dtype=float)[0])
    np.testing.assert_allclose(np.diag(h), one_step, rtol=1e-8, atol=1e-16)
    assert np.all(one_step > last_in_sample)
    assert not np.allclose(np.diag(h), last_in_sample, rtol=1e-3, atol=1e-16)
    assert params["covariance_object"] == DCC_COVARIANCE_OBJECT_ONE_STEP
    assert float(params["horizon"]) == 1.0


def test_ccc_refuses_incomplete_terminal_row() -> None:
    returns = _simulate_ccc_garch(n=80, seed=109)
    returns[-1, 0] = np.nan
    with pytest.raises(ValueError, match="incomplete_terminal_row"):
        ccc(returns)


def test_ccc_does_not_concatenate_across_interior_holes() -> None:
    base = _simulate_ccc_garch(n=120, seed=113)
    hole_i = 40
    holed = base.copy()
    holed[hole_i] = np.array([np.nan, np.nan])
    suffix = base[hole_i + 1 :]
    h_holed, params_holed = ccc(holed)
    h_suffix, params_suffix = ccc(suffix)
    np.testing.assert_allclose(h_holed, h_suffix, rtol=1e-10, atol=1e-16)
    assert float(params_holed["n_obs"]) == float(suffix.shape[0])
    assert float(params_holed["n_prefix_dropped"]) == float(hole_i + 1)
    assert params_holed["sample"] == DCC_SAMPLE_TRAILING_COMPLETE
    assert float(params_suffix["n_prefix_dropped"]) == 0.0


def test_ccc_requires_two_return_series() -> None:
    rng = np.random.default_rng(127)
    with pytest.raises(ValueError, match="at least two return series"):
        ccc(rng.normal(size=(80, 1)) * 0.01)


def test_agdcc_equal_diagonal_recovers_scalar_adcc_step() -> None:
    qbar = np.array([[1.0, 0.3], [0.3, 1.0]])
    nbar = np.array([[0.40, 0.05], [0.05, 0.35]])
    q = qbar.copy()
    a_s, b_s, g_s = 0.05, 0.90, 0.04
    a = np.full(2, np.sqrt(a_s))
    b = np.full(2, np.sqrt(b_s))
    g = np.full(2, np.sqrt(g_s))
    z = np.array([-1.2, 0.8])
    n_shock = _adcc_negative_shocks(z)
    intercept = _agdcc_intercept(qbar, nbar, a, b, g)
    expected_intercept = (1.0 - a_s - b_s) * qbar - g_s * nbar
    np.testing.assert_allclose(intercept, expected_intercept, rtol=1e-12, atol=1e-16)
    q_scalar = _adcc_step_q(q, qbar, nbar, z, n_shock, a_s, b_s, g_s)
    q_diag = _agdcc_step_q(q, intercept, z, n_shock, a, b, g)
    np.testing.assert_allclose(q_diag, q_scalar, rtol=1e-12, atol=1e-16)


def test_agdcc_heterogeneous_diagonal_is_not_scalar() -> None:
    qbar = np.eye(2)
    nbar = 0.25 * np.eye(2)
    q = qbar.copy()
    z = np.array([-1.0, -1.0])
    n_shock = _adcc_negative_shocks(z)
    a = np.array([0.10, 0.40])
    b = np.array([0.90, 0.80])
    g = np.array([0.20, 0.05])
    intercept = _agdcc_intercept(qbar, nbar, a, b, g)
    q_diag = _agdcc_step_q(q, intercept, z, n_shock, a, b, g)
    mean_a = float(np.mean(a) ** 2)
    mean_b = float(np.mean(b) ** 2)
    mean_g = float(np.mean(g) ** 2)
    q_scalar = _adcc_step_q(q, qbar, nbar, z, n_shock, mean_a, mean_b, mean_g)
    assert float(q_diag[0, 1]) != pytest.approx(float(q_scalar[0, 1]))


def test_agdcc_is_catalog_estimator_not_scalar_adcc() -> None:
    returns = _simulate_ccc_garch(n=120, seed=131)
    h, params = agdcc(returns)
    assert min_eigenvalue(h) >= -1e-8
    assert params["family"] == DCC_FAMILY_AGDCC
    assert params["spec"] == DCC_SPEC_AGDCC
    assert params["parameterization"] == DCC_PARAMETERIZATION_DIAGONAL
    assert params["dist"] == "normal"
    assert params["stage1"] == "garch"
    assert params["stage1_dist"] == "normal"
    assert params["asymmetric"] == "true"
    assert params["dynamic_correlation"] == "true"
    assert params["covariance_object"] == DCC_COVARIANCE_OBJECT_ONE_STEP
    assert params["sample"] == DCC_SAMPLE_TRAILING_COMPLETE
    assert float(params["horizon"]) == 1.0
    assert "a" not in params
    assert "b" not in params
    assert "g" not in params
    assert float(params["a_mean"]) >= 0.0
    assert float(params["b_mean"]) >= 0.0
    assert float(params["g_mean"]) >= 0.0
    assert float(params["intercept_eig_min"]) >= -1e-8
    one_step = np.empty(returns.shape[1])
    for j in range(returns.shape[1]):
        model = GARCHVol(
            min_obs=DCC_STAGE1_MIN_OBS, mean="Constant", series_scope="dcc_stage1_univariate"
        )
        model.fit_returns(returns[:, j])
        one_step[j] = float(np.asarray(model.forecast(horizon=1)["variance"], dtype=float)[0])
    np.testing.assert_allclose(np.diag(h), one_step, rtol=1e-8, atol=1e-16)


def test_agdcc_does_not_call_scalar_or_engle(monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom(*_args: object, **_kwargs: object) -> tuple[object, dict[str, str]]:
        raise AssertionError("AG-DCC must not silently run scalar ADCC or Engle DCC")

    monkeypatch.setattr("quant_fund.models.covariance.dcc_gaussian", _boom)
    monkeypatch.setattr("quant_fund.models.covariance.dcc_student_t", _boom)
    monkeypatch.setattr("quant_fund.models.covariance.adcc", _boom)
    monkeypatch.setattr("quant_fund.models.covariance.ccc", _boom)
    monkeypatch.setattr("quant_fund.models.covariance.agdcc_full", _boom)
    returns = _simulate_ccc_garch(n=80, seed=137)
    h, params = agdcc(returns)
    assert params["family"] == DCC_FAMILY_AGDCC
    assert params["parameterization"] == DCC_PARAMETERIZATION_DIAGONAL
    assert min_eigenvalue(h) >= -1e-8


def test_agdcc_is_one_step_ahead_not_in_sample_last() -> None:
    returns = _simulate_ccc_garch(n=240, seed=139)
    shocked = returns.copy()
    shocked[-1] = shocked[-1] * 8.0
    h, params = agdcc(shocked)
    assert params["covariance_object"] == DCC_COVARIANCE_OBJECT_ONE_STEP
    assert float(params["horizon"]) == 1.0
    h_base, _params_base = agdcc(returns)
    assert not np.allclose(h, h_base)


def test_agdcc_refuses_negative_g0() -> None:
    returns = _simulate_ccc_garch(n=80, seed=149)
    with pytest.raises(ValueError, match="g0 must be finite and non-negative"):
        agdcc(returns, g0=-0.1)


def test_agdcc_refuses_incomplete_terminal_row() -> None:
    returns = _simulate_ccc_garch(n=80, seed=151)
    returns[-1, 0] = np.nan
    with pytest.raises(ValueError, match="incomplete_terminal_row"):
        agdcc(returns)


def test_agdcc_does_not_concatenate_across_interior_holes() -> None:
    base = _simulate_ccc_garch(n=120, seed=157)
    hole_i = 40
    holed = base.copy()
    holed[hole_i] = np.array([np.nan, np.nan])
    suffix = base[hole_i + 1 :]
    h_holed, params_holed = agdcc(holed)
    h_suffix, params_suffix = agdcc(suffix)
    np.testing.assert_allclose(h_holed, h_suffix, rtol=1e-10, atol=1e-16)
    assert float(params_holed["n_obs"]) == float(suffix.shape[0])
    assert float(params_holed["n_prefix_dropped"]) == float(hole_i + 1)
    assert params_holed["sample"] == DCC_SAMPLE_TRAILING_COMPLETE
    assert float(params_suffix["n_prefix_dropped"]) == 0.0


def test_agdcc_requires_two_return_series() -> None:
    rng = np.random.default_rng(163)
    with pytest.raises(ValueError, match="at least two return series"):
        agdcc(rng.normal(size=(80, 1)) * 0.01)


def test_agdcc_full_diagonal_recovers_diagonal_agdcc_step() -> None:
    qbar = np.array([[1.0, 0.3], [0.3, 1.0]])
    nbar = np.array([[0.40, 0.05], [0.05, 0.35]])
    q = qbar.copy()
    a = np.array([0.22, 0.18])
    b = np.array([0.90, 0.88])
    g = np.array([0.12, 0.08])
    z = np.array([-1.2, 0.8])
    n_shock = _adcc_negative_shocks(z)
    intercept_diag = _agdcc_intercept(qbar, nbar, a, b, g)
    intercept_full = _agdcc_full_intercept(qbar, nbar, np.diag(a), np.diag(b), np.diag(g))
    np.testing.assert_allclose(intercept_full, intercept_diag, rtol=1e-12, atol=1e-16)
    q_diag = _agdcc_step_q(q, intercept_diag, z, n_shock, a, b, g)
    q_full = _agdcc_full_step_q(q, intercept_full, z, n_shock, np.diag(a), np.diag(b), np.diag(g))
    np.testing.assert_allclose(q_full, q_diag, rtol=1e-12, atol=1e-16)


def test_agdcc_full_offdiag_is_not_diagonal() -> None:
    qbar = np.eye(2)
    nbar = 0.25 * np.eye(2)
    q = qbar.copy()
    z = np.array([-1.0, -1.0])
    n_shock = _adcc_negative_shocks(z)
    a_diag = np.array([0.20, 0.20])
    b_diag = np.array([0.90, 0.90])
    g_diag = np.array([0.10, 0.10])
    a_full = np.array([[0.20, 0.05], [0.04, 0.20]])
    b_full = np.diag(b_diag)
    g_full = np.diag(g_diag)
    intercept_diag = _agdcc_intercept(qbar, nbar, a_diag, b_diag, g_diag)
    intercept_full = _agdcc_full_intercept(qbar, nbar, a_full, b_full, g_full)
    q_diag = _agdcc_step_q(q, intercept_diag, z, n_shock, a_diag, b_diag, g_diag)
    q_full = _agdcc_full_step_q(q, intercept_full, z, n_shock, a_full, b_full, g_full)
    assert float(q_full[0, 1]) != pytest.approx(float(q_diag[0, 1]))


def test_agdcc_full_is_catalog_estimator_not_diagonal() -> None:
    returns = _simulate_ccc_garch(n=120, seed=131)
    h, params = agdcc_full(returns)
    assert min_eigenvalue(h) >= -1e-8
    assert params["family"] == DCC_FAMILY_AGDCC_FULL
    assert params["spec"] == DCC_SPEC_AGDCC_FULL
    assert params["parameterization"] == DCC_PARAMETERIZATION_FULL
    assert params["dist"] == "normal"
    assert params["stage1"] == "garch"
    assert params["stage1_dist"] == "normal"
    assert params["asymmetric"] == "true"
    assert params["dynamic_correlation"] == "true"
    assert params["covariance_object"] == DCC_COVARIANCE_OBJECT_ONE_STEP
    assert params["sample"] == DCC_SAMPLE_TRAILING_COMPLETE
    assert float(params["horizon"]) == 1.0
    assert "a" not in params
    assert "b" not in params
    assert "g" not in params
    assert float(params["a_mean"]) >= 0.0
    assert float(params["b_mean"]) >= 0.0
    assert float(params["g_mean"]) >= 0.0
    assert float(params["a_offdiag_maxabs"]) >= 0.0
    assert float(params["kronecker_radius"]) < 1.0 + 1e-8
    assert float(params["intercept_eig_min"]) >= -1e-8
    one_step = np.empty(returns.shape[1])
    for j in range(returns.shape[1]):
        model = GARCHVol(
            min_obs=DCC_STAGE1_MIN_OBS, mean="Constant", series_scope="dcc_stage1_univariate"
        )
        model.fit_returns(returns[:, j])
        one_step[j] = float(np.asarray(model.forecast(horizon=1)["variance"], dtype=float)[0])
    np.testing.assert_allclose(np.diag(h), one_step, rtol=1e-8, atol=1e-16)


def test_agdcc_full_does_not_call_diagonal_or_engle(monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom(*_args: object, **_kwargs: object) -> tuple[object, dict[str, str]]:
        raise AssertionError("full AG-DCC must not silently run diagonal AG-DCC or Engle DCC")

    monkeypatch.setattr("quant_fund.models.covariance.dcc_gaussian", _boom)
    monkeypatch.setattr("quant_fund.models.covariance.dcc_student_t", _boom)
    monkeypatch.setattr("quant_fund.models.covariance.adcc", _boom)
    monkeypatch.setattr("quant_fund.models.covariance.ccc", _boom)
    monkeypatch.setattr("quant_fund.models.covariance.agdcc", _boom)
    returns = _simulate_ccc_garch(n=80, seed=173)
    h, params = agdcc_full(returns)
    assert params["family"] == DCC_FAMILY_AGDCC_FULL
    assert params["parameterization"] == DCC_PARAMETERIZATION_FULL
    assert min_eigenvalue(h) >= -1e-8


def test_agdcc_full_is_one_step_ahead_not_in_sample_last() -> None:
    returns = _simulate_ccc_garch(n=120, seed=179)
    shocked = returns.copy()
    shocked[-1] = shocked[-1] * 8.0
    h, params = agdcc_full(shocked)
    assert params["covariance_object"] == DCC_COVARIANCE_OBJECT_ONE_STEP
    assert float(params["horizon"]) == 1.0
    h_base, _params_base = agdcc_full(returns)
    assert not np.allclose(h, h_base)


def test_agdcc_full_refuses_negative_g0() -> None:
    returns = _simulate_ccc_garch(n=80, seed=181)
    with pytest.raises(ValueError, match="g0 must be finite and non-negative"):
        agdcc_full(returns, g0=-0.1)


def test_agdcc_full_refuses_incomplete_terminal_row() -> None:
    returns = _simulate_ccc_garch(n=80, seed=191)
    returns[-1, 0] = np.nan
    with pytest.raises(ValueError, match="incomplete_terminal_row"):
        agdcc_full(returns)


def test_agdcc_full_does_not_concatenate_across_interior_holes() -> None:
    base = _simulate_ccc_garch(n=120, seed=193)
    hole_i = 40
    holed = base.copy()
    holed[hole_i] = np.array([np.nan, np.nan])
    suffix = base[hole_i + 1 :]
    h_holed, params_holed = agdcc_full(holed)
    h_suffix, params_suffix = agdcc_full(suffix)
    np.testing.assert_allclose(h_holed, h_suffix, rtol=1e-10, atol=1e-16)
    assert float(params_holed["n_obs"]) == float(suffix.shape[0])
    assert float(params_holed["n_prefix_dropped"]) == float(hole_i + 1)
    assert params_holed["sample"] == DCC_SAMPLE_TRAILING_COMPLETE
    assert float(params_suffix["n_prefix_dropped"]) == 0.0


def test_agdcc_full_requires_two_return_series() -> None:
    rng = np.random.default_rng(197)
    with pytest.raises(ValueError, match="at least two return series"):
        agdcc_full(rng.normal(size=(80, 1)) * 0.01)


def test_student_t_corr_nll_matches_covariance_t_logpdf() -> None:
    from scipy.stats import multivariate_t

    rng = np.random.default_rng(17)
    nu = 6.0
    corr = np.array([[1.0, 0.4], [0.4, 1.0]])
    residual = rng.normal(size=2)
    ours = student_t_corr_nll(residual, corr, nu)
    scale = ((nu - 2.0) / nu) * corr
    ref = -2.0 * float(multivariate_t.logpdf(residual, loc=np.zeros(2), shape=scale, df=nu))
    assert ours + residual.size * np.log(np.pi) == pytest.approx(ref, rel=1e-9, abs=1e-9)
    with pytest.raises(ValueError, match="greater than 2"):
        student_t_corr_nll(residual, corr, 1.5)


def test_dcc_student_t_refuses_nu_at_or_below_two() -> None:
    returns = _simulate_ccc_garch(n=80, seed=47)
    with pytest.raises(ValueError, match="greater than 2"):
        dcc_student_t(returns, nu=1.5)


def test_dcc_student_t_refuses_incomplete_terminal_row() -> None:
    returns = _simulate_ccc_garch(n=80, seed=53)
    returns[-1, 0] = np.nan
    with pytest.raises(ValueError, match="incomplete_terminal_row"):
        dcc_student_t(returns)


def test_dcc_student_t_does_not_concatenate_across_interior_holes() -> None:
    base = _simulate_ccc_garch(n=120, seed=59)
    hole_i = 40
    holed = base.copy()
    holed[hole_i] = np.array([np.nan, np.nan])
    suffix = base[hole_i + 1 :]
    h_holed, params_holed = dcc_student_t(holed)
    h_suffix, params_suffix = dcc_student_t(suffix)
    np.testing.assert_allclose(h_holed, h_suffix, rtol=1e-10, atol=1e-16)
    assert float(params_holed["n_obs"]) == float(suffix.shape[0])
    assert float(params_holed["n_prefix_dropped"]) == float(hole_i + 1)
    assert params_holed["sample"] == DCC_SAMPLE_TRAILING_COMPLETE
    assert float(params_suffix["n_prefix_dropped"]) == 0.0


def test_require_implemented_dcc_spec_accepts_gaussian_aliases() -> None:
    assert require_implemented_dcc_spec("dcc_gaussian") == DCC_FAMILY_GAUSSIAN
    assert require_implemented_dcc_spec("Engle_2002") == DCC_FAMILY_GAUSSIAN
    assert require_implemented_dcc_spec("normal") == DCC_FAMILY_GAUSSIAN


def test_require_implemented_dcc_spec_accepts_student_t_and_adcc() -> None:
    assert require_implemented_dcc_spec("dcc_student_t") == DCC_FAMILY_STUDENT_T
    assert require_implemented_dcc_spec("student_t") == DCC_FAMILY_STUDENT_T
    assert require_implemented_dcc_spec("t") == DCC_FAMILY_STUDENT_T
    assert require_implemented_dcc_spec("adcc") == DCC_FAMILY_ADCC
    assert require_implemented_dcc_spec("asymmetric_dcc") == DCC_FAMILY_ADCC
    assert require_implemented_dcc_spec("cappiello_engle_sheppard_2006") == DCC_FAMILY_ADCC
    assert require_implemented_dcc_spec("ccc") == DCC_FAMILY_CCC
    assert require_implemented_dcc_spec("bollerslev_1990_ccc") == DCC_FAMILY_CCC
    assert require_implemented_dcc_spec("constant_conditional_correlation") == DCC_FAMILY_CCC
    assert require_implemented_dcc_spec("agdcc") == DCC_FAMILY_AGDCC
    assert require_implemented_dcc_spec("ag_dcc") == DCC_FAMILY_AGDCC
    assert require_implemented_dcc_spec("diagonal_agdcc") == DCC_FAMILY_AGDCC
    assert (
        require_implemented_dcc_spec("cappiello_engle_sheppard_2006_diagonal_agdcc")
        == DCC_FAMILY_AGDCC
    )
    assert require_implemented_dcc_spec("agdcc_full") == DCC_FAMILY_AGDCC_FULL
    assert require_implemented_dcc_spec("full_agdcc") == DCC_FAMILY_AGDCC_FULL
    assert (
        require_implemented_dcc_spec("cappiello_engle_sheppard_2006_full_agdcc")
        == DCC_FAMILY_AGDCC_FULL
    )
    with pytest.raises(ValueError, match="DCC family must be a non-empty string"):
        require_implemented_dcc_spec("  ")


def test_models_catalog_does_not_advertise_generic_dcc() -> None:
    from fastapi.testclient import TestClient

    from quant_fund.api.app import app
    from quant_fund.models.covariance import (
        IMPLEMENTED_COVARIANCE_SPECS,
        IMPLEMENTED_OPTIMIZER_COVARIANCE_SPECS,
        OPTIMIZER_COVARIANCE_EWMA,
        OPTIMIZER_COVARIANCE_OAS,
        OPTIMIZER_COVARIANCE_SAMPLE,
        UNSPECIFIED_COVARIANCE_SPECS,
    )

    payload = TestClient(app).get("/models").json()
    assert payload["covariance"] == list(IMPLEMENTED_COVARIANCE_SPECS)
    assert DCC_FAMILY_GAUSSIAN in payload["covariance"]
    assert DCC_FAMILY_STUDENT_T in payload["covariance"]
    assert DCC_FAMILY_ADCC in payload["covariance"]
    assert DCC_FAMILY_AGDCC in payload["covariance"]
    assert DCC_FAMILY_AGDCC_FULL in payload["covariance"]
    assert DCC_FAMILY_CCC in payload["covariance"]
    assert OPTIMIZER_COVARIANCE_OAS in payload["covariance"]
    assert OPTIMIZER_COVARIANCE_LEDOIT_WOLF_NONLINEAR in payload["covariance"]
    assert "dcc" not in payload["covariance"]
    assert payload["covariance_unspecified"] == list(UNSPECIFIED_COVARIANCE_SPECS)
    assert payload["covariance_unspecified"] == []
    assert DCC_FAMILY_AGDCC_FULL not in payload["covariance_unspecified"]
    assert DCC_FAMILY_STUDENT_T not in payload["covariance_unspecified"]
    assert DCC_FAMILY_ADCC not in payload["covariance_unspecified"]
    assert DCC_FAMILY_AGDCC not in payload["covariance_unspecified"]
    assert DCC_FAMILY_CCC not in payload["covariance_unspecified"]
    assert payload["optimizer_covariance"] == list(IMPLEMENTED_OPTIMIZER_COVARIANCE_SPECS)
    assert payload["optimizer_covariance"] == [
        OPTIMIZER_COVARIANCE_LEDOIT_WOLF,
        DCC_FAMILY_GAUSSIAN,
        DCC_FAMILY_STUDENT_T,
        DCC_FAMILY_ADCC,
        DCC_FAMILY_CCC,
        DCC_FAMILY_AGDCC,
        DCC_FAMILY_AGDCC_FULL,
        OPTIMIZER_COVARIANCE_EWMA,
        OPTIMIZER_COVARIANCE_OAS,
        OPTIMIZER_COVARIANCE_LEDOIT_WOLF_NONLINEAR,
        OPTIMIZER_COVARIANCE_SAMPLE,
    ]
    assert "factor" not in payload["optimizer_covariance"]
    assert OPTIMIZER_COVARIANCE_EWMA in payload["optimizer_covariance"]
    assert OPTIMIZER_COVARIANCE_OAS in payload["optimizer_covariance"]
    assert OPTIMIZER_COVARIANCE_LEDOIT_WOLF_NONLINEAR in payload["optimizer_covariance"]
    assert OPTIMIZER_COVARIANCE_SAMPLE in payload["optimizer_covariance"]
    assert DCC_FAMILY_STUDENT_T in payload["optimizer_covariance"]
    assert DCC_FAMILY_ADCC in payload["optimizer_covariance"]
    assert DCC_FAMILY_CCC in payload["optimizer_covariance"]
    assert DCC_FAMILY_AGDCC in payload["optimizer_covariance"]
    assert DCC_FAMILY_AGDCC_FULL in payload["optimizer_covariance"]
    assert "factor" not in payload["optimizer_covariance"]
    assert "dcc" not in payload["optimizer_covariance"]


def test_require_implemented_optimizer_covariance_named_paths() -> None:
    assert (
        require_implemented_optimizer_covariance("ledoit_wolf") == OPTIMIZER_COVARIANCE_LEDOIT_WOLF
    )
    assert require_implemented_optimizer_covariance("DCC_GAUSSIAN") == DCC_FAMILY_GAUSSIAN
    assert require_implemented_optimizer_covariance("engle_2002") == DCC_FAMILY_GAUSSIAN
    assert require_implemented_optimizer_covariance("dcc_student_t") == DCC_FAMILY_STUDENT_T
    assert (
        require_implemented_optimizer_covariance("engle_2002_student_t_dcc")
        == DCC_FAMILY_STUDENT_T
    )
    assert require_implemented_optimizer_covariance("adcc") == DCC_FAMILY_ADCC
    assert (
        require_implemented_optimizer_covariance("cappiello_engle_sheppard_2006")
        == DCC_FAMILY_ADCC
    )
    assert require_implemented_optimizer_covariance("asymmetric_dcc") == DCC_FAMILY_ADCC
    assert require_implemented_optimizer_covariance("ccc") == DCC_FAMILY_CCC
    assert require_implemented_optimizer_covariance("bollerslev_1990_ccc") == DCC_FAMILY_CCC
    assert (
        require_implemented_optimizer_covariance("constant_conditional_correlation")
        == DCC_FAMILY_CCC
    )
    assert require_implemented_optimizer_covariance("agdcc") == DCC_FAMILY_AGDCC
    assert require_implemented_optimizer_covariance("ag_dcc") == DCC_FAMILY_AGDCC
    assert require_implemented_optimizer_covariance("diagonal_agdcc") == DCC_FAMILY_AGDCC
    assert (
        require_implemented_optimizer_covariance("cappiello_engle_sheppard_2006_diagonal_agdcc")
        == DCC_FAMILY_AGDCC
    )
    assert require_implemented_optimizer_covariance("agdcc_full") == DCC_FAMILY_AGDCC_FULL
    assert require_implemented_optimizer_covariance("full_agdcc") == DCC_FAMILY_AGDCC_FULL
    assert (
        require_implemented_optimizer_covariance("cappiello_engle_sheppard_2006_full_agdcc")
        == DCC_FAMILY_AGDCC_FULL
    )
    assert require_implemented_optimizer_covariance("ewma") == OPTIMIZER_COVARIANCE_EWMA
    assert require_implemented_optimizer_covariance("riskmetrics") == OPTIMIZER_COVARIANCE_EWMA
    assert require_implemented_optimizer_covariance("oas") == OPTIMIZER_COVARIANCE_OAS
    assert (
        require_implemented_optimizer_covariance("oracle_approximating_shrinkage")
        == OPTIMIZER_COVARIANCE_OAS
    )
    assert require_implemented_optimizer_covariance("chen_2010") == OPTIMIZER_COVARIANCE_OAS
    assert (
        require_implemented_optimizer_covariance("ledoit_wolf_nonlinear")
        == OPTIMIZER_COVARIANCE_LEDOIT_WOLF_NONLINEAR
    )
    assert (
        require_implemented_optimizer_covariance("nlshrink")
        == OPTIMIZER_COVARIANCE_LEDOIT_WOLF_NONLINEAR
    )
    assert (
        require_implemented_optimizer_covariance("ledoit_wolf_2020_analytical")
        == OPTIMIZER_COVARIANCE_LEDOIT_WOLF_NONLINEAR
    )
    assert require_implemented_optimizer_covariance("sample") == OPTIMIZER_COVARIANCE_SAMPLE
    assert require_implemented_optimizer_covariance("unbiased_sample") == OPTIMIZER_COVARIANCE_SAMPLE
    with pytest.raises(ValueError, match="unknown_optimizer_covariance"):
        require_implemented_optimizer_covariance("t")
    with pytest.raises(ValueError, match="unknown_optimizer_covariance"):
        require_implemented_optimizer_covariance("student_t")
    with pytest.raises(ValueError, match="unknown_dcc_spec:dcc"):
        require_implemented_optimizer_covariance("dcc")
    with pytest.raises(ValueError, match="unknown_dcc_spec:shrinkage"):
        require_implemented_optimizer_covariance("shrinkage")
    with pytest.raises(ValueError, match="analytical 2020, not QuEST 2017"):
        require_implemented_optimizer_covariance("ledoit_wolf_2017")
    with pytest.raises(ValueError, match="analytical 2020, not QuEST 2017"):
        require_implemented_optimizer_covariance("quest")
    with pytest.raises(ValueError, match="unwired_optimizer_covariance:factor"):
        require_implemented_optimizer_covariance("factor")
    with pytest.raises(ValueError, match="unknown_optimizer_covariance:normal"):
        require_implemented_optimizer_covariance("normal")
    with pytest.raises(ValueError, match="optimizer covariance must be a non-empty string"):
        require_implemented_optimizer_covariance("  ")
