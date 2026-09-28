"""KATs and contracts for the covariance estimator catalog.

Every estimator must return a symmetric PSD covariance with a positive
diagonal, deterministic output for a fixed input, and honest metadata.
DCC-family estimators need >= DCC_STAGE1_MIN_OBS complete trailing rows.
"""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models import covariance as cov


def _shared_factor_returns(t: int = 80, n: int = 3, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    factor = rng.standard_normal(t)
    return factor[:, None] * np.array([0.8, 0.5, -0.3]) + 0.5 * rng.standard_normal((t, n))


def _assert_covariance(sigma: np.ndarray, n: int = 3) -> None:
    assert sigma.shape == (n, n)
    np.testing.assert_allclose(sigma, sigma.T, atol=1e-10)
    assert np.all(np.diag(sigma) > 0.0)
    assert np.min(np.linalg.eigvalsh(sigma)) >= -1e-9


def _corr01(sigma: np.ndarray) -> float:
    return float(sigma[0, 1] / np.sqrt(sigma[0, 0] * sigma[1, 1]))


class TestRepairPsd:
    def test_negative_eigenvalue_is_clipped(self) -> None:
        # [[1, 2], [2, 1]] has eigenvalues 3 and -1.
        sigma = np.array([[1.0, 2.0], [2.0, 1.0]])
        repaired, info = cov.repair_psd(sigma)
        assert info["repaired"] == 1.0
        assert info["eig_min_before"] == pytest.approx(-1.0)
        assert info["eig_min_after"] >= 0.0
        assert info["frobenius"] > 0.0
        _assert_covariance(repaired, n=2)

    def test_psd_input_passes_through(self) -> None:
        sigma = np.array([[1.0, 0.3], [0.3, 1.0]])
        repaired, info = cov.repair_psd(sigma)
        assert info["repaired"] == 0.0
        np.testing.assert_allclose(repaired, sigma)

    def test_validation(self) -> None:
        with pytest.raises(ValueError, match="square"):
            cov.repair_psd(np.ones((2, 3)))
        with pytest.raises(ValueError, match="finite"):
            cov.repair_psd(np.array([[1.0, np.nan], [0.0, 1.0]]))


class TestEstimatorContracts:
    @pytest.mark.parametrize(
        "name",
        ["dcc_gaussian", "dcc_student_t", "adcc", "agdcc", "agdcc_full", "ccc", "ewma"],
    )
    def test_estimator_returns_psd_and_family(self, name: str) -> None:
        fn = getattr(cov, name)
        x = _shared_factor_returns()
        sigma, meta = fn(x)
        _assert_covariance(sigma)
        # Assets 0 and 1 share a positive factor loading — must be detected.
        assert _corr01(sigma) > 0.2
        # Assets 0 and 2 have opposite signs on the shared factor.
        corr02 = float(sigma[0, 2] / np.sqrt(sigma[0, 0] * sigma[2, 2]))
        assert corr02 < 0.0

    @pytest.mark.parametrize("name", ["dcc_gaussian", "adcc", "agdcc", "ccc"])
    def test_deterministic_across_calls(self, name: str) -> None:
        fn = getattr(cov, name)
        x = _shared_factor_returns()
        a, _ = fn(x)
        b, _ = fn(x)
        np.testing.assert_array_equal(a, b)

    def test_agdcc_full_and_gaussian_agree_on_isotropic_news(self) -> None:
        # With scalar parameters broadcast to diagonals, agdcc reduces to
        # adcc-style dynamics; both must land in a similar correlation ballpark.
        x = _shared_factor_returns()
        diag, _ = cov.agdcc(x)
        full, _ = cov.agdcc_full(x)
        assert abs(_corr01(diag) - _corr01(full)) < 0.3

    def test_anticorrelated_pair_detected(self) -> None:
        rng = np.random.default_rng(1)
        factor = rng.standard_normal(100)
        x = np.column_stack([factor, -factor]) + 0.2 * rng.standard_normal((100, 2))
        sigma, _ = cov.dcc_gaussian(x)
        _assert_covariance(sigma, n=2)
        assert sigma[0, 1] < 0.0


class TestLinearAndShrinkageEstimators:
    def test_sample_cov_matches_numpy(self) -> None:
        x = _shared_factor_returns(t=60)
        sigma = cov.sample_cov(x)
        np.testing.assert_allclose(sigma, np.cov(x.T, ddof=1), rtol=1e-10)

    def test_ewma_cov_psd_and_deterministic(self) -> None:
        x = _shared_factor_returns(t=60)
        sigma = cov.ewma_cov(x, lam=0.94)
        _assert_covariance(sigma)
        np.testing.assert_array_equal(sigma, cov.ewma_cov(x, lam=0.94))

    def test_ewma_rejects_bad_lambda(self) -> None:
        with pytest.raises(ValueError):
            cov.ewma_cov(_shared_factor_returns(t=60), lam=1.5)

    def test_shrinkage_estimators_stay_psd(self) -> None:
        x = _shared_factor_returns(t=60)
        for fn in (cov.ledoit_wolf_cov, cov.oracle_approximating_shrinkage_cov):
            sigma = fn(x)
            _assert_covariance(sigma)
        sigma_nl = cov.ledoit_wolf_nonlinear_cov(x)
        _assert_covariance(sigma_nl)

    def test_factor_cov_kat(self) -> None:
        betas = np.array([[1.0, 0.0], [0.0, 1.0]])
        factors = np.array([[0.04, 0.01], [0.01, 0.09]])
        idio = np.array([0.01, 0.02])
        expected = betas @ factors @ betas.T + np.diag(idio)
        np.testing.assert_allclose(cov.factor_cov(betas, factors, idio), expected)
        with pytest.raises(ValueError, match="incompatible"):
            cov.factor_cov(np.ones((2, 2)), np.ones((3, 3)), np.ones(2))
        with pytest.raises(ValueError, match="non-negative"):
            cov.factor_cov(betas, factors, np.array([-1.0, 0.0]))


class TestTrailingWindow:
    def test_incomplete_last_row_fails_closed(self) -> None:
        x = _shared_factor_returns(t=60)
        x[-1, 0] = np.nan
        with pytest.raises(ValueError, match="incomplete_terminal_row"):
            cov.dcc_trailing_complete_window(x)

    def test_interior_hole_keeps_trailing_contiguous_block(self) -> None:
        x = _shared_factor_returns(t=70)
        x[5, 0] = np.nan
        window = cov.dcc_trailing_complete_window(x)
        assert window.shape == (64, 3)
        np.testing.assert_array_equal(window, x[6:])

    def test_insufficient_rows_fail(self) -> None:
        with pytest.raises(ValueError, match="insufficient_contiguous_rows"):
            cov.dcc_trailing_complete_window(_shared_factor_returns(t=40))


class TestNameResolution:
    @pytest.mark.parametrize(
        ("alias", "expected"),
        [
            ("dcc_gaussian", cov.DCC_FAMILY_GAUSSIAN),
            ("gaussian", cov.DCC_FAMILY_GAUSSIAN),
            ("engle_2002", cov.DCC_FAMILY_GAUSSIAN),
            ("adcc", cov.DCC_FAMILY_ADCC),
            ("cappiello_engle_sheppard", cov.DCC_FAMILY_ADCC),
            ("agdcc", cov.DCC_FAMILY_AGDCC),
            ("agdcc_full", cov.DCC_FAMILY_AGDCC_FULL),
        ],
    )
    def test_dcc_family_aliases(self, alias: str, expected: str) -> None:
        assert cov.require_implemented_dcc_spec(alias) == expected

    def test_generic_dcc_stays_unknown(self) -> None:
        with pytest.raises(ValueError, match="unknown_dcc_spec"):
            cov.require_implemented_dcc_spec("dcc")
        with pytest.raises(ValueError):
            cov.require_implemented_dcc_spec("")

    def test_optimizer_covariance_resolution(self) -> None:
        # Named paths resolve; ambiguous aliases must not.
        resolved = cov.require_implemented_optimizer_covariance("ledoit_wolf")
        assert isinstance(resolved, str) and resolved
        for ambiguous in ("gaussian", "normal", "t", "student_t", "shrinkage", "dcc"):
            with pytest.raises(ValueError):
                cov.require_implemented_optimizer_covariance(ambiguous)


class TestHelpers:
    def test_ewma_variance_1d(self) -> None:
        r = np.array([0.1, -0.1, 0.05])
        out = cov.ewma_variance_1d(r, lam=0.94)
        assert out.shape == r.shape
        assert np.all(np.isfinite(out))

    def test_condition_number(self) -> None:
        sigma = np.diag([4.0, 1.0])
        assert cov.condition_number(sigma) == pytest.approx(4.0)

    def test_is_symmetric_and_min_eigenvalue(self) -> None:
        assert cov.is_symmetric(np.array([[1.0, 0.5], [0.5, 1.0]]))
        assert not cov.is_symmetric(np.array([[1.0, 0.5], [0.6, 1.0]]))
        assert cov.min_eigenvalue(np.diag([1.0, 2.0])) == pytest.approx(1.0)
