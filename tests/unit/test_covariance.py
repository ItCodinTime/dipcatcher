import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from quant_fund.models.covariance import (
    dcc_gaussian,
    ewma_cov,
    ewma_variance_1d,
    factor_cov,
    is_symmetric,
    ledoit_wolf_cov,
    min_eigenvalue,
    oracle_approximating_shrinkage_cov,
    repair_psd,
    sample_cov,
)


def test_ledoit_wolf_psd() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(80, 6))
    s = ledoit_wolf_cov(x)
    assert is_symmetric(s)
    assert min_eigenvalue(s) >= -1e-8


def test_oas_covariance_is_finite_psd_in_high_dimensional_regime() -> None:
    rng = np.random.default_rng(123)
    sigma = oracle_approximating_shrinkage_cov(rng.normal(size=(12, 40)))
    assert sigma.shape == (40, 40)
    assert np.isfinite(sigma).all()
    assert is_symmetric(sigma)
    assert min_eigenvalue(sigma) >= -1e-10


def test_oas_covariance_rejects_insufficient_finite_rows() -> None:
    with pytest.raises(ValueError, match="at least 2"):
        oracle_approximating_shrinkage_cov(np.array([[1.0, 2.0], [np.nan, 3.0]]))


def test_repair_indefinite() -> None:
    s = np.array([[1.0, 2.0], [2.0, 1.0]])
    r, info = repair_psd(s)
    assert info["repaired"] == 1.0
    assert min_eigenvalue(r) >= -1e-8
    assert is_symmetric(r)


def test_repair_rejects_malformed_covariance() -> None:
    with np.testing.assert_raises(ValueError):
        repair_psd(np.ones((2, 3)))
    with np.testing.assert_raises(ValueError):
        repair_psd(np.array([[1.0, np.nan], [np.nan, 1.0]]))


def test_factor_covariance_rejects_invalid_inputs() -> None:
    assert factor_cov(np.ones((2, 1)), np.array([[0.04]]), np.array([0.01, 0.02])).shape == (2, 2)
    with np.testing.assert_raises(ValueError):
        factor_cov(np.ones((2, 1)), np.eye(2), np.ones(2))
    with np.testing.assert_raises(ValueError):
        factor_cov(np.ones((2, 1)), np.array([[0.04]]), np.array([-0.01, 0.02]))
    with np.testing.assert_raises(ValueError):
        factor_cov(np.eye(2), np.array([[1.0, 2.0], [2.0, 1.0]]), np.ones(2))


def test_ewma_rejects_invalid_lambda_and_dcc_is_finite() -> None:
    with np.testing.assert_raises(ValueError):
        ewma_variance_1d(np.ones(3), lam=1.1)
    with np.testing.assert_raises(ValueError):
        ewma_cov(np.ones((3, 2)), lam=-0.1)

    rng = np.random.default_rng(11)
    sigma, params = dcc_gaussian(rng.normal(size=(60, 4)) * 0.01)
    assert np.isfinite(sigma).all()
    assert min_eigenvalue(sigma) >= -1e-8
    assert 0.0 <= params["a"] + params["b"] < 0.999


@given(st.integers(3, 8), st.integers(20, 40))
@settings(max_examples=10, deadline=None)
def test_sample_cov_symmetric(k: int, n: int) -> None:
    rng = np.random.default_rng(k * 100 + n)
    x = rng.normal(size=(n, k))
    s = sample_cov(x)
    assert is_symmetric(s)


def test_repair_already_psd_noop() -> None:
    s = np.array([[2.0, 0.5], [0.5, 1.0]])
    r, info = repair_psd(s)
    assert info["repaired"] == 0.0
    assert info["eig_min_before"] == pytest.approx(info["eig_min_after"])
    assert min_eigenvalue(r) >= -1e-10
    np.testing.assert_allclose(r, 0.5 * (s + s.T))


def test_repair_near_singular_becomes_psd() -> None:
    # Ill-conditioned indefinite: large off-diagonal relative to tiny diag.
    s = np.array([[1e-8, 1.0], [1.0, 1e-8]])
    assert min_eigenvalue(s) < 0.0
    r, info = repair_psd(s, tol=1e-12)
    assert info["repaired"] == 1.0
    assert np.isfinite(r).all()
    assert min_eigenvalue(r) >= -1e-8
    assert is_symmetric(r)


def test_sample_cov_drops_nan_rows() -> None:
    x = np.array(
        [
            [0.01, 0.02],
            [np.nan, 0.03],
            [0.04, 0.05],
            [0.06, np.nan],
            [0.07, 0.08],
            [0.09, 0.10],
        ]
    )
    s = sample_cov(x)
    assert s.shape == (2, 2)
    assert is_symmetric(s)
    assert np.isfinite(s).all()


def test_sample_cov_rejects_too_few_finite_rows() -> None:
    x = np.array([[1.0, 2.0], [np.nan, 3.0], [4.0, np.nan]])
    with np.testing.assert_raises(ValueError):
        sample_cov(x)


def test_condition_number_identity_is_one() -> None:
    from quant_fund.models.covariance import condition_number

    assert condition_number(np.eye(3)) == pytest.approx(1.0)


def test_condition_number_is_infinite_for_singular_or_indefinite() -> None:
    from quant_fund.models.covariance import condition_number

    assert np.isinf(condition_number(np.diag([1.0, 0.0])))
    assert np.isinf(condition_number(np.array([[1.0, 2.0], [2.0, 1.0]])))
    with pytest.raises(ValueError, match="square matrix"):
        condition_number(np.ones((2, 3)))
    with pytest.raises(ValueError, match="finite"):
        condition_number(np.array([[1.0, np.nan], [np.nan, 1.0]]))


def test_factor_cov_closed_form_and_psd() -> None:
    """B F B' + diag(idio) closed form; result PSD when inputs valid."""
    betas = np.array([[1.0], [2.0]])
    f = np.array([[0.04]])
    idio = np.array([0.01, 0.02])
    sigma = factor_cov(betas, f, idio)
    expected = np.array([[0.05, 0.08], [0.08, 0.18]])
    np.testing.assert_allclose(sigma, expected)
    assert is_symmetric(sigma)
    assert min_eigenvalue(sigma) >= -1e-10


def test_factor_cov_multi_factor_closed_form() -> None:
    betas = np.array([[1.0, 0.0], [0.5, 1.0], [0.0, 0.5]])
    f = np.array([[0.04, 0.01], [0.01, 0.09]])
    idio = np.array([0.01, 0.02, 0.03])
    sigma = factor_cov(betas, f, idio)
    expected = betas @ f @ betas.T + np.diag(idio)
    np.testing.assert_allclose(sigma, expected)
    assert min_eigenvalue(sigma) >= -1e-10


def test_factor_cov_rejects_nan_and_nonsquare_factor() -> None:
    with np.testing.assert_raises(ValueError):
        factor_cov(np.ones((2, 1)), np.array([[np.nan]]), np.ones(2))
    with np.testing.assert_raises(ValueError):
        factor_cov(np.ones((2, 1)), np.ones((1, 2)), np.ones(2))


def test_dcc_rejects_too_few_rows_and_bad_a0() -> None:
    with np.testing.assert_raises(ValueError):
        dcc_gaussian(np.ones((2, 3)) * 0.01)  # min_rows=3
    with np.testing.assert_raises(ValueError):
        dcc_gaussian(np.ones((10, 2)) * 0.01, a0=-0.1)
    with np.testing.assert_raises(ValueError):
        dcc_gaussian(np.ones((10, 2)) * 0.01, b0=float("nan"))
    with pytest.raises(ValueError, match="correlation target is non-finite"):
        dcc_gaussian(np.ones((10, 2)) * 0.01)


def test_dcc_finite_psd_on_constant_plus_noise() -> None:
    rng = np.random.default_rng(21)
    x = 0.001 + rng.normal(scale=0.01, size=(50, 3))
    h, params = dcc_gaussian(x)
    assert h.shape == (3, 3)
    assert np.isfinite(h).all()
    assert is_symmetric(h)
    assert min_eigenvalue(h) >= -1e-8
    assert np.isfinite(params["a"]) and np.isfinite(params["b"])
    assert params["a"] >= 0.0 and params["b"] >= 0.0
